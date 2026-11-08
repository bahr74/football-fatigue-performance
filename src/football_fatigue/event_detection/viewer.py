"""Match viewer: one self-contained HTML page comparing detected events with the PFF labels for one match.

Four tabs: results in plain language (with a match timeline and where the mistakes come from), an event explorer (real
versus detected, field by field, with a position replay), how each event is detected (the real thresholds of this run),
and the gaps of this match with ways to fix them. All numbers are computed from the detection and the labels.

The page embeds tracking positions and player names, so it is a local report (``reports/`` is ignored by Git); do not share
it outside the team until the dataset licence is verified (docs/data_policy.md).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import evaluate as E
from .classify import DetectedEvent
from .config import DetectorConfig
from .features import FEATURES
from .inputs import MatchPositions
from .touches import Touch

CLIP_BEFORE_S, CLIP_AFTER_S, CLIP_STEP = 2.5, 3.5, 3       # every third frame: about 10 Hz
MAX_PASS_CLIPS = 150                                        # replays stored for ordinary passes, per result category
TYPE_NAMES = {"PA": "Pass", "CR": "Cross", "SH": "Shot", "CL": "Clearance"}


def _clip(pos: MatchPositions, t: float, period: int) -> dict:
    sel = np.flatnonzero((pos.period == period) & (pos.t >= t - CLIP_BEFORE_S) & (pos.t <= t + CLIP_AFTER_S))[::CLIP_STEP]
    dm = lambda a: None if np.isnan(a) else int(round(float(a) * 10))     # noqa: E731  decimetres keep the page small
    players = []
    for j in range(pos.px.shape[1]):
        xs, ys = pos.px[sel, j], pos.py[sel, j]
        if np.isnan(xs).all():
            continue
        players.append([j, [dm(v) for v in xs], [dm(v) for v in ys]])
    return {"t": [round(float(pos.t[i] - t), 2) for i in sel], "bx": [dm(pos.ball[i, 0]) for i in sel], "by": [dm(pos.ball[i, 1]) for i in sel],
            "players": players}


def _clock(pos: MatchPositions, t: float, period: int) -> str:
    starts = {1: 0, 2: 45, 3: 90, 4: 105}
    first = pos.t[pos.period == period]
    sec = int(t - (first[0] if first.size else t)) + starts.get(period, 0) * 60
    return f"{sec // 60}'{sec % 60:02d}"


def _round(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 1)


def build_data(pos: MatchPositions, detected: Sequence[DetectedEvent], truth: list[E.LabelledEvent], outside: list[E.LabelledEvent],
               touches: list[Touch], score: dict, names: dict[str, Any], match: dict[str, Any], cfg: DetectorConfig, version: str,
               model_info: dict | None) -> dict:
    pred = E.detected_as_labelled(pos, detected)
    T = [e for e in truth if e.type in TYPE_NAMES]
    pairs = E.match_events(pred, T, E.PRIMARY_TOLERANCE_S)
    side_of_team = {pos.home_team_id: 0, pos.away_team_id: 1}
    by_key, by_id = names["by_key"], names["by_id"]
    touch_t = np.array([t.t_s for t in touches])
    touch_p = np.array([t.period for t in touches])

    def det_info(i: int) -> dict:
        d = detected[i]
        actor = pos.player_key[d.touch.player]
        rec = pos.player_key[d.receiver] if d.receiver is not None else None
        return dict(type=d.type, outcome=d.outcome, key=d.key_pass, goal=d.is_goal, actor=actor, actorName=by_key.get(actor),
                    receiver=rec, receiverName=by_key.get(rec) if rec else None, side=d.team_side, conf=d.confidence,
                    reason=d.reason, x=_round(d.touch.x), y=_round(d.touch.y), t=round(d.touch.t_s, 2))

    def truth_info(j: int) -> dict:
        e = T[j]
        return dict(type=e.type, outcome=e.outcome, key=e.key_pass, goal=e.type == "SH" and e.outcome == "G", playerId=e.player_id,
                    playerName=by_id.get(e.player_id), receiverId=e.receiver_id, receiverName=by_id.get(e.receiver_id),
                    side=side_of_team.get(e.team_id), x=_round(e.x), y=_round(e.y), t=round(e.t_s, 2))

    def has_candidate(e: E.LabelledEvent) -> bool:
        if not touch_t.size:
            return False
        return bool(np.any((touch_p == e.period) & (np.abs(touch_t - e.t_s) <= E.PRIMARY_TOLERANCE_S)))

    items = []
    used_p, used_t = set(), set()
    for i, j, dt in pairs:
        used_p.add(i)
        used_t.add(j)
        d, t = det_info(i), truth_info(j)
        pid = pos.player_id[detected[i].touch.player]
        checks = dict(type=d["type"] == t["type"], team=d["side"] == t["side"], dt=round(dt, 2))
        if t["playerId"] is not None and pid >= 0:
            checks["player"] = int(pid) == t["playerId"]
        if t["type"] in ("PA", "CR") and d["type"] in ("PA", "CR") and t["outcome"] and d["outcome"]:
            checks["outcome"] = (t["outcome"] == "C") == (d["outcome"] == "C")
        if t["type"] == "SH" and d["type"] == "SH":
            checks["outcome"] = t["goal"] == d["goal"]
        if d["receiver"] and t["receiverId"] is not None:
            rid = pos.player_id[detected[i].receiver]
            checks["receiver"] = int(rid) == t["receiverId"]
        if t["x"] is not None and d["x"] is not None:
            checks["dist"] = round(float(np.hypot(t["x"] - d["x"], t["y"] - d["y"])), 1)
        items.append(dict(status="hit" if checks["type"] else "type_differs", t=T[j].t_s, period=T[j].period, det=d, truth=t, checks=checks))
    for j, e in enumerate(T):
        if j not in used_t:
            items.append(dict(status="missed", why="rejected" if has_candidate(e) else "no_touch", t=e.t_s, period=e.period, det=None,
                              truth=truth_info(j), checks={}))
    for i, p in enumerate(pred):
        if i not in used_p:
            items.append(dict(status="false_alarm", t=p.t_s, period=p.period, det=det_info(i), truth=None, checks={}))
    items.sort(key=lambda x: x["t"])

    def important(it: dict) -> bool:
        return any(e and (e["type"] in ("SH", "CR") or e["key"] or e["goal"]) for e in (it["det"], it["truth"]))

    per_status: dict[str, int] = {}
    for n, it in enumerate(items):
        it["id"] = n
        it["clock"] = _clock(pos, it["t"], it["period"])
        keep = important(it)
        if not keep:
            k = it["status"] + it.get("why", "")
            if per_status.get(k, 0) < MAX_PASS_CLIPS:
                per_status[k] = per_status.get(k, 0) + 1
                keep = True
        if keep:
            it["clip"] = _clip(pos, it["t"], it["period"])

    counts = {k: score["counts"][f"{k}@{E.PRIMARY_TOLERANCE_S}"] for k in E.SCORED_TYPES}
    periods = {int(p): {"start": float(pos.t[pos.period == p][0]), "end": float(pos.t[pos.period == p][-1])} for p in np.unique(pos.period)}
    ball_present = float(np.mean(~np.isnan(pos.ball[:, 0])))
    outside_counts: dict[str, int] = {}
    for e in outside:
        key = "Goal" if e.type == "SH" and e.outcome == "G" else TYPE_NAMES.get(e.type, e.type)
        outside_counts[key] = outside_counts.get(key, 0) + 1
    return dict(match=match, version=version, kind="ml" if model_info else "rules", model=model_info, counts=counts,
                attributes=score["attributes"], confusion=score["confusion"], dts=score["timing_dt_s"].get("PASS_ANY", []), items=items,
                periods=periods, sides=[int(s) for s in pos.player_side], keys=pos.player_key, names=by_key,
                ballPresent=round(ball_present, 3), excludedPeriods=list(pos.excluded_periods), outsideCounts=outside_counts,
                config=cfg.to_dict())


# ---------------------------------------------------------------- static tabs (thresholds and numbers filled in)

def _rules_html(cfg: DetectorConfig, kind: str, model_info: dict | None) -> str:
    c = cfg
    ml = kind == "ml"
    thr = (model_info or {}).get("event_threshold")
    steps = [
        ("Positions in", "Every frame (29.97 per second) gives the raw x/y of the 22 players and of the ball. Nothing from the PFF event "
         "file goes in. PFF's smoothed ball is left out on purpose, because PFF builds it from the event labels."),
        ("Clean the ball", f"The source repeats each ball position for two frames (about 15 updates per second), so repeats are ignored. "
         f"A position that would need the ball to fly faster than {c.max_ball_speed_mps:g} m/s in and out is a glitch and is dropped. "
         "Missing ball stays missing; nothing is filled in."),
        ("Find touches", f"The ball's velocity is measured over about {c.velocity_half_window / 30:.1f} s before and after every frame. A "
         f"<b>touch</b> is a sudden change of at least {c.touch_dv_mps:g} m/s, the biggest within ±{c.touch_peak_window} frames, with a player "
         f"within {c.touch_radius_m:g} m of the ball. The nearest player is the candidate actor"
         + (", and the nearest opponent in reach is scored too." if ml else ".")),
        ("Decide the event", (f"A trained model reads {len(FEATURES)} numbers about each touch (below) and gives a probability for pass, cross, shot, "
                              f"clearance or nothing. A touch becomes an event when the chance of <i>some</i> event is at least {thr:g}, "
                              "chosen on 8 development matches only.") if ml and thr is not None else
         "Fixed rules, below, decide pass, cross or shot for every touch that sends the ball away."),
        ("Finish the details", "Who received it, whether it was completed, whether a shot was a goal, and which pass set up a shot. "
         "The result is written in the same shape as the PFF event file."),
    ]
    flow = "".join(f"<li><b>{h}</b><span>{b}</span></li>" for h, b in steps)
    rules = [
        ("Pass", "PA", [
            f"A touch sends the ball away at {c.min_release_speed_mps:g} m/s or more, and the next player to touch it is someone else.",
            f"Completed when a teammate touches it next; defended when an opponent does; out when the ball is last seen outside the lines. "
            f"If the ball disappears for more than {c.dead_ball_gap_s:g} s with no receiver, no pass is claimed.",
            f"The same player touching again while the ball stays within {c.carry_radius_m:g} m is a dribble, not a pass."]),
        ("Cross", "CR", [
            f"A pass from a wide position (at least {c.cross_min_abs_y_m:g} m sideways from the middle of the pitch), within "
            f"{c.cross_max_goal_distance_m:g} m of the goal line it attacks.",
            f"The ball moves inwards at {c.cross_min_speed_mps:g} m/s or more and ends in or near the penalty box."]),
        ("Shot", "SH", [
            f"Struck within {c.shot_max_distance_m:g} m of the goal it attacks.",
            f"Either the ball's direction crosses the goal line between the posts (± {c.shot_post_margin_m:g} m) at {c.shot_min_speed_mps:g} m/s "
            f"or more, or the ball is seen within {c.shot_reach_goal_m:g} m of the goal mouth."]),
        ("Goal", "G", [
            f"A shot, and within {c.goal_line_window_s:g} s the ball reaches the goal mouth (crosses the line between the posts, or comes "
            f"within {c.goal_mouth_max_m:g} m of the frame).",
            f"And play then stops for {c.goal_min_stoppage_s:g} to {c.goal_max_stoppage_s:g} s (celebration and kickoff), starting within "
            f"{c.goal_stoppage_start_s:g} s of the shot.",
            "A long pause alone (injury, VAR, time-wasting) or a ball near the goal alone (a save) is not a goal. Only the last shot "
            "before a pause can be the goal."]),
        ("Key pass", "KP", [
            "The last completed pass or cross by the shooting team before a shot, with no opponent action in between, "
            f"within {c.key_pass_window_s:g} s. The same rule is applied to the PFF labels, so both sides are compared fairly."]),
    ]
    used = {"PA", "CR", "SH"} if not ml else set()
    cards = []
    for name, code, lines in rules:
        note = "" if (code in used or not ml or code in ("G", "KP")) else (
            "<p class='mlnote'>In this run the model decides this event type; the rule is the starting point it was compared against.</p>")
        cards.append(f"<article class='rule'><header><span class='chip t-{code}'>{code}</span><h3>{name}</h3></header><ul>"
                     + "".join(f"<li>{x}</li>" for x in lines) + f"</ul>{note}</article>")
    features = ("Where the ball is (in the attacking team's direction), distance and angle to goal; how sharply the ball changed "
                "velocity and how fast it left; ball height; the direction it flew and where that line meets the goal line; where it "
                "ended up; how long until the next touch and whether that was a teammate, an opponent or the same player; whether it "
                "went out; how close it came to the goal mouth and how long play stopped afterwards; defenders within 3 and 6 m, the "
                "nearest opponent and teammate; how much of the last second the ball was visible.")
    ml_box = (f"<section class='box'><h3>The model ({html.escape(model_info.get('model', ''))})</h3><p>Gradient-boosted decision trees "
              f"(scikit-learn). It learned from {model_info.get('rows', 0):,} candidate touches (one row per candidate player) in the other "
              "World Cup matches, each labelled with the PFF "
              "event at that moment. This match was <b>not</b> in the training data of the model that scored it. A second model "
              f"predicts whether a pass was completed.</p><p><b>What it looks at:</b> {features}</p></section>") if ml else ""
    return (f"<section><h2>From positions to events</h2><ol class='flow'>{flow}</ol></section>{ml_box}"
            f"<section><h2>Event rules</h2><div class='rules'>{''.join(cards)}</div></section>"
            "<section class='box'><h3>How a detection is marked correct</h3><p>A detected event is correct when the PFF file has an event "
            f"of the same type within ±{E.PRIMARY_TOLERANCE_S:g} s, in the same half. Each label can be used once. Team, player, receiver and "
            "completed-or-not are then checked separately (Event explorer).</p></section>")


def _gaps_html(data: dict) -> str:
    items = data["items"]
    cnt = lambda f: sum(1 for it in items if f(it))       # noqa: E731
    miss_no = cnt(lambda it: it["status"] == "missed" and it.get("why") == "no_touch")
    miss_rej = cnt(lambda it: it["status"] == "missed" and it.get("why") == "rejected")
    fa = cnt(lambda it: it["status"] == "false_alarm")
    cl_as_pass = data["confusion"].get("CL->PA", 0)
    cr_pa = data["confusion"].get("CR->PA", 0) + data["confusion"].get("PA->CR", 0)
    sh_miss = cnt(lambda it: it["status"] == "missed" and it["truth"]["type"] == "SH")
    goals_t = data["counts"]["GOAL"]["n_truth"]
    goals_hit = data["counts"]["GOAL"]["tp"]
    att = data["attributes"].get("PASS_ANY", {})
    wrong_team = att.get("matched", 0) - att.get("team_agree", 0)
    wrong_player = att.get("player_comparable", 0) - att.get("player_agree", 0)
    wrong_out = att.get("outcome_comparable", 0) - att.get("outcome_agree", 0)
    outside = data["outsideCounts"]
    excl = data["excludedPeriods"]
    excl_txt = ""
    if excl:
        plural = {"Pass": "passes", "Cross": "crosses", "Shot": "shots", "Goal": "goals", "Clearance": "clearances"}
        parts = ", ".join(f"{v} {(plural.get(k, k.lower() + 's') if v != 1 else k.lower())}" for k, v in sorted(outside.items()))
        excl_txt = (f"Periods {', '.join(map(str, excl))} (extra time) of this match are left out: {parts or 'events'} there (shoot-out kicks included) are not scored. "
                    "The audit found that the home/away player lists look swapped in that period of the tracking data, so team "
                    "attribution there cannot be trusted.")
    gaps = [
        ("Events the touch finder never sees", miss_no, "labelled passes, crosses, shots or clearances had no candidate touch within 1 s",
         "The ball is missing in {:.0%} of this match's frames and only updates about 15 times a second, so a quick one-touch pass "
         "often leaves no clear change in ball velocity.".format(1 - data["ballPresent"]),
         ["Rebuild a continuous ball path between observations with a physics model (a Kalman filter or ballistic fit), stored as a "
          "separate derived layer and never using the PFF smoothed ball, which leaks the labels.",
          "Detect possession changes from players too: who is closest to the ball over the next half second, not only the ball's velocity.",
          "Use a sequence model over frame windows instead of a single peak, so two quick touches are not merged."], "High", "Medium"),
        ("Touches seen but judged not to be an event", miss_rej, "labelled events had a candidate touch the detector rejected",
         "The model prefers precision: the threshold was chosen on development matches to balance passes, crosses and shots.",
         ["Give the model the context of the whole move: the previous and next 3 to 5 touches (a sequence model such as a GRU or a "
          "temporal convolution).", "Calibrate a separate threshold per event type.",
          "Add more training data (other competitions with tracking and events)."], "High", "Medium"),
        ("Detections with no event behind them", fa, "detections had no labelled pass, cross, shot or clearance within 1 s",
         "Deflections, blocks, headers in duels and goalkeeper actions look like touches.",
         ["Add the other PFF event types as classes (challenge, clearance, ball carry, rebound) so the model has somewhere to put them.",
          "Add player speed and body orientation features from the tracking data."], "Medium", "Low"),
        ("Clearances counted as passes", cl_as_pass, "PFF clearances were reported as passes",
         "The detector has no clearance output; a long kick away from goal looks like a long pass.",
         ["Export the clearance class the ML model already learns (it is currently dropped).",
          "Use a pressure feature (defenders within 3 m) and distance from own goal more strongly."], "Medium", "Low"),
        ("Passes and crosses mixed up", cr_pa, "events were found but given the other of pass/cross",
         "The line between a cutback, a low cross and a pass from the wing is fuzzy, also for human annotators.",
         ["Add ball height over the flight and the receiving zone as features.", "Report pass and cross together when the user only needs passes."],
         "Low", "Low"),
        ("Shots not found", sh_miss, "labelled shots were missed",
         "Blocked shots change direction within a metre, so the flight to goal is never seen; headers happen in crowds where the "
         "nearest player is often a defender.",
         ["Look at the ball's path over the next second, not only the first 0.4 s.",
          "Score more than two candidate players per touch (the three nearest)."], "High", "Medium"),
        ("Wrong team or player on a correct event", wrong_team + wrong_player,
         f"times on correctly timed passes ({wrong_team} wrong team, {wrong_player} wrong player)",
         "Tracking identifies players by shirt number; most shirt reads have low confidence, and the nearest player at the touch "
         "is not always the one who played the ball.",
         ["Track identity over several frames before and after the touch, weighted by jersey confidence.",
          "In the video pipeline, re-identify players by appearance as well as by number."], "Medium", "Medium"),
        ("Completed-or-not wrong", wrong_out, "correctly found passes have the wrong completed/not completed result",
         "The next detected touch can be a deflection or a challenge rather than the real receiver.",
         ["Predict the receiver directly (which teammate is the ball moving towards) instead of taking the next touch."], "Medium", "Low"),
        ("Goals", goals_t - goals_hit, f"of {goals_t} goals in the scored periods were not detected",
         "The ball is rarely tracked into the net, so a goal needs the stoppage after it as evidence, and a goal can only be found "
         "when its shot was found.",
         ["Detect the kickoff that follows a goal from player positions (both teams in their own half, ball on the centre spot).",
          "In the video pipeline: read the scoreboard (text recognition) or detect the ball crossing the goal line in the image."],
         "High", "Medium"),
    ]
    cards = []
    for title, n, unit, cause, fixes, impact, effort in gaps:
        cards.append(f"<article class='gap'><div class='gnum'>{n:,}</div><div class='gbody'><h3>{title}</h3><p class='gunit'>{unit} in this match.</p>"
                     f"<p><b>Why:</b> {cause}</p><p><b>How to fix:</b></p><ul>{''.join(f'<li>{f}</li>' for f in fixes)}</ul>"
                     f"<p class='tags'><span>Impact: {impact}</span><span>Effort: {effort}</span></p></div></article>")
    general = [
        ("From tracking to video", "The model learned PFF tracking. Positions from our video pipeline will be noisier, miss players "
         "off camera and swap identities. Before trusting results on video: feed the detector PFF data made worse on purpose (dropped "
         "ball frames, position noise, swapped ids) to measure how fast accuracy falls, then retrain on a mix."),
        ("Labels are not ground truth", "PFF events are human annotations with their own timing conventions. A sample of disputed "
         "events (misses and false alarms) should be checked against match video to separate detector errors from label noise."),
        ("Extra time and penalties", excl_txt or "This match has no excluded periods."),
        ("More event types", "PFF also labels challenges, ball carries, rebounds, touches and set pieces. Adding them as classes gives "
         "a fuller match story and removes false alarms that are really one of these."),
    ]
    gen = "".join(f"<article class='box'><h3>{h}</h3><p>{b}</p></article>" for h, b in general)
    return (f"<section><h2>What it misses in this match, and how to fix it</h2><p class='lede'>Each card names a failure, how often it "
            f"happened in this match, why, and what would fix it. Impact and effort are estimates.</p><div class='gaps'>{''.join(cards)}</div></section>"
            f"<section><h2>Bigger limits of the approach</h2><div class='grid2'>{gen}</div></section>")


def write_viewer(path: Path, pos: MatchPositions, detected: Sequence[DetectedEvent], truth: list[E.LabelledEvent], outside: list[E.LabelledEvent],
                 touches: list[Touch], score: dict, names: dict[str, Any], match: dict[str, Any], cfg: DetectorConfig, version: str,
                 model_info: dict | None = None) -> Path:
    data = build_data(pos, detected, truth, outside, touches, score, names, match, cfg, version, model_info)
    title = f"{match['home']} v {match['away']} events"
    page = (_HTML.replace("__TITLE__", html.escape(title)).replace("__RULES__", _rules_html(cfg, data["kind"], model_info))
            .replace("__GAPS__", _gaps_html(data)).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* A match-centre layout: tab bar, then wide panels; pitch green accent, result colours carry meaning. */
:root{--bg:#f4f6f3;--panel:#ffffff;--panel2:#eef2ee;--ink:#141a16;--muted:#5e6a62;--line:#d8dfd9;--accent:#18794e;
--hit:#1a7f4b;--diff:#7a4fd1;--miss:#c26a00;--fa:#c0362c;--home:#2a62d4;--away:#d23c3c;--pitch:#2c7a4b;--pitch-line:rgba(255,255,255,.7);
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Inter Tight",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1311;--panel:#171d19;--panel2:#1d2520;--ink:#e7ece8;--muted:#98a59c;--line:#2b352f;--accent:#4cc38a;
--hit:#3fbf7f;--diff:#a585ff;--miss:#f0a030;--fa:#ff6b5e;--home:#6b9bff;--away:#ff7272;--pitch:#1f5a37;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1311;--panel:#171d19;--panel2:#1d2520;--ink:#e7ece8;--muted:#98a59c;--line:#2b352f;--accent:#4cc38a;
--hit:#3fbf7f;--diff:#a585ff;--miss:#f0a030;--fa:#ff6b5e;--home:#6b9bff;--away:#ff7272;--pitch:#1f5a37;color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body)}
.wrap{max-width:1280px;margin:0 auto;padding:0 max(16px,env(safe-area-inset-left,0px)) 48px max(16px,env(safe-area-inset-right,0px))}
header.top{padding:22px 0 8px;display:grid;gap:6px}
.eyebrow{font:500 12px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
h1{font:700 clamp(32px,5vw,52px)/1 var(--display);margin:0;letter-spacing:.01em}
h2{font:700 26px/1.1 var(--display);margin:0 0 12px;letter-spacing:.02em}
h3{font:600 19px/1.2 var(--display);margin:0 0 6px;letter-spacing:.02em}
p{margin:0 0 10px}.lede{color:var(--muted);max-width:75ch}
.sub{color:var(--muted)}.badge{display:inline-block;font:500 12px var(--mono);padding:2px 8px;border:1px solid var(--line);border-radius:99px;margin-right:6px;background:var(--panel)}
nav.tabs{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;gap:4px;border-bottom:1px solid var(--line);margin:14px 0 20px;overflow-x:auto}
nav.tabs button{font:600 16px var(--display);letter-spacing:.04em;background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);padding:10px 14px;cursor:pointer;white-space:nowrap}
nav.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent)}
nav.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
section{margin-bottom:30px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:grid;gap:8px;min-width:0}
.card h3{display:flex;justify-content:space-between;align-items:baseline}
.card .f1{font:500 12px var(--mono);color:var(--muted)}
.big{font:700 34px/1 var(--display)}.big small{font:500 15px var(--body);color:var(--muted);margin-left:4px}
.meter{display:grid;gap:3px}.meter span{font-size:13px;color:var(--muted)}.meter div{height:8px;background:var(--panel2);border-radius:99px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--accent);border-radius:99px}
.chip{display:inline-block;font:600 12px var(--mono);padding:2px 7px;border-radius:5px;color:#fff;background:var(--muted)}
.t-PA{background:#3f6fb5}.t-CR{background:#8a5cc7}.t-SH{background:#c0632b}.t-G{background:#18794e}.t-KP{background:#a0802a}.t-CL{background:#6b7a70}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.s-hit{background:var(--hit)}.s-type_differs{background:var(--diff)}.s-missed{background:var(--miss)}.s-false_alarm{background:var(--fa)}
.timeline{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
.timeline svg{display:block;min-width:720px;width:100%;height:auto}
.timeline text{font:12px var(--mono);fill:var(--muted)}.timeline .rowlab{font:600 14px var(--display);fill:var(--ink)}
.timeline .mark{cursor:pointer}.timeline .mark:hover{stroke:var(--ink);stroke-width:2}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.bars{display:grid;gap:12px}.barrow{display:grid;grid-template-columns:150px 1fr 80px;gap:10px;align-items:center;font-size:14px}
.stack{display:flex;height:22px;border-radius:5px;overflow:hidden;background:var(--panel2)}.stack div{height:100%}
.barrow .n{font:13px var(--mono);color:var(--muted);text-align:right}
.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
/* explorer */
.explorer{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:14px;align-items:start}
@media (max-width:900px){.explorer{grid-template-columns:1fr}}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
select,input[type=search]{font:inherit;font-size:14px;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);min-width:0}
input[type=search]{flex:1 1 140px}
#list{max-height:72vh;overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.row{display:grid;grid-template-columns:10px 46px 1fr;gap:8px;align-items:center;padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.row:hover{background:var(--panel2)}.row.sel{background:var(--panel2);box-shadow:inset 3px 0 0 var(--accent)}
.row .dot{width:10px;height:10px;border-radius:50%}.row .clk{font:12px var(--mono);color:var(--muted)}.row .txt{font-size:13px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.count{font:12px var(--mono);color:var(--muted);margin:6px 2px}
.detail{display:grid;gap:12px;min-width:0}
.verdict{border-radius:10px;padding:12px 16px;color:#fff;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}
.verdict b{font:700 22px var(--display);letter-spacing:.03em}.verdict span{font-size:14px;opacity:.95}
.v-hit{background:var(--hit)}.v-type_differs{background:var(--diff)}.v-missed{background:var(--miss)}.v-false_alarm{background:var(--fa)}
.compare{display:grid;grid-template-columns:120px 1fr 1fr 34px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}
.compare>div{padding:8px 12px;border-bottom:1px solid var(--line);min-width:0}
.compare .h{font:600 13px var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);background:var(--panel2)}
.compare .k{color:var(--muted);font-size:13px}.compare .ok{color:var(--hit);font-weight:700;text-align:center}.compare .bad{color:var(--fa);font-weight:700;text-align:center}.compare .na{color:var(--muted);text-align:center}
@media (max-width:560px){.compare{grid-template-columns:90px 1fr 1fr 26px;font-size:13px}}
.why ul{margin:6px 0 0;padding-left:18px}.why li{margin-bottom:4px}
.probs{display:grid;gap:4px;margin-top:6px}.prob{display:grid;grid-template-columns:80px 1fr 46px;gap:8px;align-items:center;font-size:13px}
.prob div{height:8px;background:var(--panel2);border-radius:99px;overflow:hidden}.prob i{display:block;height:100%;background:var(--accent)}
.prob .v{font:12px var(--mono);text-align:right;color:var(--muted)}
.replay{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
canvas{width:100%;height:auto;display:block;border-radius:6px;background:var(--pitch)}
.controls{display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap}
.controls button{font:600 14px var(--body);padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);cursor:pointer}
.controls input[type=range]{flex:1 1 200px}.controls .tm{font:12px var(--mono);color:var(--muted);min-width:64px}
.noclip{padding:30px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:10px}
/* rules + gaps */
.flow{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;counter-reset:s}
.flow li{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:grid;gap:4px;counter-increment:s}
.flow li b{font:700 18px var(--display);letter-spacing:.03em}.flow li>b::before{content:counter(s) "  ";color:var(--accent)}
.flow li span{font-size:14px;color:var(--muted)}
.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.rule{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.rule header{display:flex;gap:10px;align-items:center;margin-bottom:6px}
.rule h3{margin:0}.rule ul{margin:0;padding-left:18px}.rule li{margin-bottom:6px}.mlnote{font-size:13px;color:var(--muted);margin-top:8px}
.gaps{display:grid;gap:12px}.gap{display:grid;grid-template-columns:110px 1fr;gap:16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
@media (max-width:560px){.gap{grid-template-columns:1fr}}
.gnum{font:700 44px/1 var(--display);color:var(--miss)}.gunit{color:var(--muted)}.gap ul{margin:0 0 8px;padding-left:18px}.gap li{margin-bottom:4px}
.tags span{display:inline-block;font:500 12px var(--mono);border:1px solid var(--line);border-radius:99px;padding:2px 8px;margin-right:6px}
[hidden]{display:none!important}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><div class="wrap">
<header class="top"><div class="eyebrow" id="eyebrow"></div><h1 id="h1"></h1><div class="sub" id="subline"></div></header>
<nav class="tabs" role="tablist">
<button role="tab" data-tab="results" aria-selected="true">Results</button>
<button role="tab" data-tab="explorer" aria-selected="false">Event explorer</button>
<button role="tab" data-tab="rules" aria-selected="false">How events are detected</button>
<button role="tab" data-tab="gaps" aria-selected="false">Gaps &amp; fixes</button>
</nav>

<main>
<div data-panel="results">
<section><h2>How well did it read this match?</h2><p class="lede" id="summaryline"></p><div class="cards" id="cards"></div></section>
<section><h2>The match, minute by minute</h2>
<div class="legend"><span><i class="s-hit"></i>found correctly</span><span><i class="s-type_differs"></i>found, wrong type</span><span><i class="s-missed"></i>missed</span><span><i class="s-false_alarm"></i>false alarm</span><span>Click a mark to open it.</span></div>
<div class="timeline" id="timeline"></div></section>
<div class="grid2">
<section class="box"><h3>When it finds an event, are the details right?</h3><div class="bars" id="details"></div></section>
<section class="box"><h3>Where the real events went</h3><div class="bars" id="where"></div>
<div class="legend" style="margin-top:10px"><span><i class="s-hit"></i>found</span><span><i class="s-type_differs"></i>other type</span><span><i style="background:var(--miss)"></i>missed, judged no event</span><span><i style="background:var(--miss);opacity:.5"></i>missed, never seen</span></div></section>
</div>
</div>

<div data-panel="explorer" hidden>
<div class="explorer">
<aside><div class="filters">
<select id="fType" aria-label="Event type"><option value="">All events</option><option value="G">Goals</option><option value="SH">Shots</option><option value="CR">Crosses</option><option value="KP">Key passes</option><option value="PA">Passes</option><option value="CL">Clearances</option></select>
<select id="fStatus" aria-label="Result"><option value="">All results</option><option value="hit">Found correctly</option><option value="type_differs">Wrong type</option><option value="missed">Missed</option><option value="false_alarm">False alarms</option></select>
<input type="search" id="fText" placeholder="Player name" aria-label="Filter by player name">
</div><div class="count" id="count"></div><div id="list"></div></aside>
<div class="detail" id="detail"></div>
</div></div>

<div data-panel="rules" hidden>__RULES__</div>
<div data-panel="gaps" hidden>__GAPS__</div>
</main></div>

<script>
const D=__DATA__;
const $=s=>document.querySelector(s), el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!==undefined)e.innerHTML=h;return e};
const NAME={PA:'Pass',CR:'Cross',SH:'Shot',CL:'Clearance'}, STATUS={hit:'Found correctly',type_differs:'Found, but as the wrong type',missed:'Missed',false_alarm:'False alarm'};
const OUT={C:'completed',D:'defended / intercepted',O:'out of play',B:'blocked',S:'stoppage',G:'goal',U:'unknown'};
const team=s=>s===0?D.match.home:s===1?D.match.away:'?';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=(a,b)=>b?Math.round(a/b*100):0;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function label(e){if(!e)return '';let s=NAME[e.type]||e.type;if(e.goal)s='Goal ('+s.toLowerCase()+')';else if(e.key)s='Key '+s.toLowerCase();return s}
function who(e,isDet){if(!e)return '';const n=isDet?(e.actorName||('#'+(e.actor||'').split(':')[1])):(e.playerName||'unknown player');return n}

// header
$('#eyebrow').textContent=`${D.match.stage||'FIFA World Cup 2022'} · ${D.match.date||''} · detector ${D.version}`;
$('#h1').textContent=`${D.match.home} v ${D.match.away}`;
$('#subline').textContent=(D.kind==='ml'?'Machine-learning detector':'Rule-based detector')+' compared with the PFF event labels. Positions only went in; the labels were used only to check the result.';

// tabs
const tabs=[...document.querySelectorAll('nav.tabs button')];
function show(name){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===name));document.querySelectorAll('[data-panel]').forEach(p=>p.hidden=p.dataset.panel!==name);
 if(name==='explorer'){renderList();if(!$('#detail').innerHTML){const it=D.items.find(i=>(i.truth&&i.truth.goal)||(i.det&&i.det.goal))||D.items[0];if(it)openItem(it.id,true)}}}
tabs.forEach(b=>b.onclick=()=>{show(b.dataset.tab);try{history.replaceState(null,'','#'+b.dataset.tab)}catch(e){}});

// results: cards
const C=D.counts;
const TYPES=[['PASS_ANY','Passes','PA'],['CR','Crosses','CR'],['SH','Shots','SH'],['GOAL','Goals','G'],['KEY_PASS','Key passes','KP']];
function f1(c){const p=c.tp/(c.tp+c.fp||1),r=c.tp/(c.tp+c.fn||1);return p+r?2*p*r/(p+r):0}
$('#summaryline').textContent=`Out of ${C.PASS_ANY.n_truth.toLocaleString()} real passes, the detector found ${C.PASS_ANY.tp.toLocaleString()} (${pct(C.PASS_ANY.tp,C.PASS_ANY.n_truth)}%), and ${pct(C.PASS_ANY.tp,C.PASS_ANY.n_pred)}% of the passes it reported really happened. It found ${C.SH.tp} of ${C.SH.n_truth} shots and ${C.GOAL.tp} of ${C.GOAL.n_truth} goals.`;
for(const [k,name,code] of TYPES){const c=C[k];const card=el('div','card');
 card.innerHTML=`<h3><span><span class="chip t-${code}">${code}</span> ${name}</span><span class="f1">F1 ${f1(c).toFixed(2)}</span></h3>
 <div class="big">${c.tp}<small>of ${c.n_truth} found</small></div>
 <div class="meter"><span>Real ${name.toLowerCase()} found: ${pct(c.tp,c.n_truth)}%</span><div><i style="width:${pct(c.tp,c.n_truth)}%"></i></div></div>
 <div class="meter"><span>Its ${name.toLowerCase()} that were real: ${pct(c.tp,c.n_pred)}% (${c.n_pred} reported)</span><div><i style="width:${pct(c.tp,c.n_pred)}%"></i></div></div>`;
 $('#cards').appendChild(card)}

// results: timeline
function rowOf(it){const e=it.truth||it.det;if((it.truth&&it.truth.goal)||(it.det&&it.det.goal))return 'Goals';if(e.type==='SH'||(it.det&&it.det.type==='SH'))return 'Shots';
 if(e.type==='CR'||(it.det&&it.det.type==='CR'))return 'Crosses';if((it.truth&&it.truth.key)||(it.det&&it.det.key))return 'Key passes';if(e.type==='CL')return null;return 'Passes'}
(function timeline(){const rows=['Goals','Shots','Crosses','Key passes','Passes'];const P=Object.entries(D.periods);
 const W=1200,L=96,gap=24,rowH=30,top=26;const span=P.map(([p,v])=>v.end-v.start),tot=span.reduce((a,b)=>a+b,0);const avail=W-L-gap*(P.length-1)-10;
 let x0=L;const geo={};P.forEach(([p,v],i)=>{geo[p]={x:x0,w:avail*span[i]/tot,s:v.start,e:v.end};x0+=geo[p].w+gap});
 const H=top+rowH*rows.length+26;let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Events along the match">`;
 rows.forEach((r,i)=>{const y=top+i*rowH;s+=`<text x="0" y="${y+rowH/2+5}" class="rowlab">${r}</text><line x1="${L}" y1="${y+rowH/2}" x2="${W-10}" y2="${y+rowH/2}" stroke="${css('--line')}"/>`});
 const startMin={1:0,2:45,3:90,4:105};
 P.forEach(([p])=>{const g=geo[p];s+=`<text x="${g.x}" y="14">${p<=2?(p==1?'1st half':'2nd half'):'Extra time '+(p-2)}</text>`;
  for(let m=0;m<=60;m+=15){const t=g.s+m*60;if(t>g.e)break;const x=g.x+g.w*(t-g.s)/(g.e-g.s);s+=`<text x="${x}" y="${H-6}" text-anchor="middle">${startMin[p]+m}'</text>`}});
 D.items.forEach(it=>{const r=rowOf(it);if(!r)return;const g=geo[it.period];if(!g)return;const i=rows.indexOf(r);const x=g.x+g.w*(it.t-g.s)/(g.e-g.s);const y=top+i*rowH+rowH/2;
  const col=css('--'+({hit:'hit',type_differs:'diff',missed:'miss',false_alarm:'fa'}[it.status]));const rr=r==='Passes'?2.6:(r==='Goals'?7:5);
  s+=`<circle class="mark" data-id="${it.id}" cx="${x.toFixed(1)}" cy="${y}" r="${rr}" fill="${col}" fill-opacity="${r==='Passes'?.55:.95}"><title>${it.clock} ${STATUS[it.status]}: ${esc(label(it.truth)||label(it.det))}</title></circle>`});
 s+='</svg>';$('#timeline').innerHTML=s;$('#timeline').onclick=e=>{const id=e.target.dataset&&e.target.dataset.id;if(id!==undefined)openItem(+id)}})();

// results: details + where
(function(){const a=(D.attributes.PASS_ANY||{}),sh=(D.attributes.SH||{});const dts=D.dts||[];const close=dts.filter(d=>Math.abs(d)<=0.2).length;
 const rows=[['Right team (passes)',a.team_agree,a.matched],['Right player (passes)',a.player_agree,a.player_comparable],['Completed or not right',a.outcome_agree,a.outcome_comparable],
 ['Timing within 0.2 s',close,dts.length],['Right team (shots)',sh.team_agree,sh.matched],['Right player (shots)',sh.player_agree,sh.player_comparable]];
 $('#details').innerHTML=rows.filter(r=>r[2]).map(([l,k,n])=>`<div class="barrow"><span>${l}</span><div class="stack"><div style="width:${pct(k,n)}%;background:var(--hit)"></div><div style="width:${100-pct(k,n)}%;background:var(--fa);opacity:.5"></div></div><span class="n">${pct(k,n)}% of ${n}</span></div>`).join('');
 const by={};D.items.forEach(it=>{if(!it.truth)return;const t=it.truth.type;by[t]=by[t]||{hit:0,diff:0,rej:0,no:0,n:0};const b=by[t];b.n++;
  if(it.status==='hit')b.hit++;else if(it.status==='type_differs')b.diff++;else if(it.why==='rejected')b.rej++;else b.no++});
 $('#where').innerHTML=['PA','CR','SH','CL'].filter(t=>by[t]).map(t=>{const b=by[t];const w=v=>(v/b.n*100).toFixed(1)+'%';
  return `<div class="barrow"><span>${{PA:'Passes',CR:'Crosses',SH:'Shots',CL:'Clearances'}[t]} (${b.n})</span><div class="stack" title="found ${b.hit}, other type ${b.diff}, judged no event ${b.rej}, never seen ${b.no}"><div style="width:${w(b.hit)};background:var(--hit)"></div><div style="width:${w(b.diff)};background:var(--diff)"></div><div style="width:${w(b.rej)};background:var(--miss)"></div><div style="width:${w(b.no)};background:var(--miss);opacity:.5"></div></div><span class="n">${pct(b.hit,b.n)}% found</span></div>`}).join('')
  +`<p class="sub" style="margin-top:8px">${D.items.filter(i=>i.status==='false_alarm').length} false alarms: detections with no labelled pass, cross, shot or clearance within 1 s.</p>`})();

// explorer
function matches(it){const f=$('#fType').value,s=$('#fStatus').value,q=$('#fText').value.trim().toLowerCase();if(s&&it.status!==s)return false;
 const es=[it.det,it.truth].filter(Boolean);
 if(f==='G'&&!es.some(e=>e.goal))return false;if(f==='KP'&&!es.some(e=>e.key))return false;if(f&&!['G','KP'].includes(f)&&!es.some(e=>e.type===f))return false;
 if(q&&!es.some(e=>[e.actorName,e.playerName,e.receiverName].some(n=>n&&n.toLowerCase().includes(q))))return false;return true}
let current=null;
function renderList(){const L=$('#list');L.innerHTML='';const its=D.items.filter(matches);$('#count').textContent=its.length.toLocaleString()+' events';
 const frag=document.createDocumentFragment();its.slice(0,1500).forEach(it=>{const r=el('div','row'+(current===it.id?' sel':''));r.dataset.id=it.id;
  const txt=it.truth?`${label(it.truth)} · ${esc(it.truth.playerName||team(it.truth.side))}`:`${label(it.det)} · ${esc(who(it.det,true))}`;
  r.innerHTML=`<span class="dot s-${it.status}"></span><span class="clk">${it.clock}</span><span class="txt">${txt}</span>`;r.onclick=()=>openItem(it.id,true);frag.appendChild(r)});
 L.appendChild(frag);if(its.length>1500)L.appendChild(el('div','count','Showing the first 1,500; narrow the filters.'))}
['#fType','#fStatus'].forEach(s=>$(s).onchange=renderList);$('#fText').oninput=renderList;
function openItem(id,fromList){current=id;if(!fromList)show('explorer');document.querySelectorAll('.row').forEach(r=>r.classList.toggle('sel',+r.dataset.id===id));
 const it=D.items[id];const d=it.det,t=it.truth,c=it.checks||{};const box=$('#detail');box.innerHTML='';
 const sub={hit:'Same event, found by the detector.',type_differs:`The detector found something here but called it a ${d&&NAME[d.type]?.toLowerCase()}.`,
  missed:it.why==='no_touch'?'No touch was detected near this moment, so the detector never considered it.':'A touch was detected here, but the detector decided it was not this event.',
  false_alarm:'The detector reported an event, but PFF has no pass, cross, shot or clearance within 1 s.'}[it.status];
 box.appendChild(el('div','verdict v-'+it.status,`<div><b>${STATUS[it.status]}</b><br><span>${sub}</span></div><span>${it.clock}${c.dt!==undefined?` · detected ${c.dt>=0?'+':''}${c.dt.toFixed(2)} s from the label`:''}</span>`));
 const mark=v=>v===undefined?'<div class="na">–</div>':v?'<div class="ok">✓</div>':'<div class="bad">✗</div>';
 const SHOT_OUT={G:'goal',S:'saved',O:'off target',B:'blocked',L:'woodwork',C:'cleared off the line',F:'other'};
 const outTxt=e=>e?(e.type==='SH'?(SHOT_OUT[e.outcome]||'no goal'):(OUT[e.outcome]||'–')):'–';
 const rows=[['Event',label(t)||'–',label(d)||'–',it.status==='hit'?true:it.status==='type_differs'?false:undefined],
  ['Team',t?team(t.side):'–',d?team(d.side):'–',c.team],['Player',t?esc(t.playerName||'–'):'–',d?esc(who(d,true)):'–',c.player],
  ['Result',outTxt(t),outTxt(d),c.outcome],['Receiver',t?esc(t.receiverName||'–'):'–',d?esc(d.receiverName||(d.receiver?'#'+d.receiver.split(':')[1]:'–')):'–',c.receiver],
  ['Time',t?it.clock:'–',d?(c.dt!==undefined?`${it.clock} (${c.dt>=0?'+':''}${c.dt.toFixed(2)} s)`:it.clock):'–',c.dt!==undefined?Math.abs(c.dt)<=0.5:undefined],
  ['Where',t&&t.x!=null?`x ${t.x}, y ${t.y} m`:'–',d?`x ${d.x}, y ${d.y} m`:'–',c.dist!==undefined?c.dist<=3:undefined]];
 let g='<div class="h"></div><div class="h">PFF label</div><div class="h">Detector</div><div class="h"></div>';
 rows.forEach(([k,a,b,ok])=>{g+=`<div class="k">${k}</div><div>${a}</div><div>${b}</div>${mark(ok)}`});
 if(c.dist!==undefined)g+=`<div class="k">Distance</div><div style="grid-column:span 3">The detected touch is ${c.dist} m from where PFF placed the ball.</div>`;
 box.appendChild(el('div','compare',g));
 if(d)box.appendChild(why(d));
 const rp=el('div','replay');box.appendChild(rp);
 if(it.clip)replay(rp,it);else rp.appendChild(el('div','noclip','No replay stored for this pass (the page keeps a sample of ordinary passes). Shots, crosses, goals and key passes always have one.'))}
function why(d){const r=d.reason||{};const w=el('div','box why');const L=[];
 if(r.class_probabilities){w.innerHTML='<h3>Why the detector said this</h3><p class="sub">Model confidence for this touch (the most confident player of the two candidates was used):</p>';
  const pr=el('div','probs');Object.entries(r.class_probabilities).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{pr.appendChild(el('div','prob',`<span>${k==='none'?'No event':NAME[k]}</span><div><i style="width:${v*100}%"></i></div><span class="v">${Math.round(v*100)}%</span>`))});w.appendChild(pr);
  if(r.p_completed!==undefined)L.push(`Completion model: ${Math.round(r.p_completed*100)}% chance the pass reached a teammate.`);
  if(r.actor_is_nearest===false)L.push('The event was given to the nearest player of the other team, not the very nearest player.')}
 else w.innerHTML='<h3>Why the detector said this</h3>';
 if(r.touch_dv_mps!==undefined)L.push(`The ball's velocity changed by ${r.touch_dv_mps} m/s at the touch and left at ${r.release_speed_mps} m/s.`);
 if(r.player_ball_distance_m!==undefined)L.push(`The player was ${r.player_ball_distance_m} m from the ball.`);
 if(r.goal_distance_m!==undefined)L.push(`Struck ${r.goal_distance_m} m from goal.`);
 if(r.projected_goal_line_y_m!==undefined)L.push(`Its direction meets the goal line ${Math.abs(r.projected_goal_line_y_m)} m from the centre of the goal (posts at 3.66 m).`);
 if(r.ball_reaches_goal_mouth)L.push('The ball was seen at the goal mouth.');
 if(r.start_abs_y_m!==undefined)L.push(`Played from ${r.start_abs_y_m} m wide, ${r.distance_from_goal_line_m} m from the goal line${r.ends_in_box_area?', into the box':''}.`);
 if(r.ball_goal_mouth_distance_m!==undefined&&r.ball_goal_mouth_distance_m!==null)L.push(`Closest the ball came to the goal frame afterwards: ${r.ball_goal_mouth_distance_m} m.`);
 if(r.stoppage_after_shot_s!==undefined&&r.stoppage_after_shot_s!==null)L.push(`Play then stopped for ${r.stoppage_after_shot_s} s${d.goal?' (goal: ball at the goal mouth and a kickoff-length pause)':''}.`);
 if(r.goal_withdrawn)L.push('Not counted as the goal: a later shot came before the same pause.');
 if(r.next_touch)L.push(`Next touch: ${r.next_touch.replace('_',' ')}.`);
 if(r.key_pass_for_shot_at_s!==undefined)L.push('Marked as a key pass: the shot followed without an opponent action.');
 const ul=el('ul');L.forEach(x=>ul.appendChild(el('li','',x)));w.appendChild(ul);return w}

// replay
let anim=null;
function replay(box,it){cancelAnimationFrame(anim);const c=it.clip;const cv=el('canvas');cv.width=1050;cv.height=700;box.appendChild(cv);
 const ctr=el('div','controls');ctr.innerHTML='<button type="button">Pause</button><input type="range" min="0" max="'+(c.t.length-1)+'" value="0" aria-label="Replay position"><span class="tm"></span>';box.appendChild(ctr);
 box.appendChild(el('div','legend','<span><i style="background:var(--home)"></i>'+esc(D.match.home)+'</span><span><i style="background:var(--away)"></i>'+esc(D.match.away)+'</span><span><i style="background:#f6d23a"></i>ball and its path</span><span>◯ white ring: detected player · ◇ yellow: PFF event position · ✕: detected touch position</span>'));
 const g=cv.getContext('2d'),W=cv.width,H=cv.height,S=W/112,X=x=>W/2+x*S,Y=y=>H/2-y*S;const btn=ctr.querySelector('button'),rng=ctr.querySelector('input'),tm=ctr.querySelector('.tm');
 const t0=it.truth?it.truth.t:it.det.t;const detOff=it.det?it.det.t-t0:null;
 let k=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,last=0;btn.textContent=playing?'Pause':'Play';
 function pitch(){g.fillStyle=css('--pitch');g.fillRect(0,0,W,H);g.strokeStyle='rgba(255,255,255,.7)';g.lineWidth=2;g.strokeRect(X(-52.5),Y(34),105*S,68*S);
  g.beginPath();g.moveTo(X(0),Y(34));g.lineTo(X(0),Y(-34));g.stroke();g.beginPath();g.arc(X(0),Y(0),9.15*S,0,7);g.stroke();
  for(const s of[-1,1]){g.strokeRect(X(s>0?36:-52.5),Y(20.16),16.5*S,40.32*S);g.strokeRect(X(s>0?47:-52.5),Y(9.16),5.5*S,18.32*S);g.strokeRect(X(s>0?52.5:-54.5),Y(3.66),2*S,7.32*S)}}
 function draw(){pitch();const dm=v=>v==null?null:v/10;
  g.strokeStyle='rgba(246,210,58,.75)';g.lineWidth=3;g.beginPath();let on=false;for(let i=0;i<=k;i++){const x=dm(c.bx[i]),y=dm(c.by[i]);if(x==null){on=false;continue}if(!on){g.moveTo(X(x),Y(y));on=true}else g.lineTo(X(x),Y(y))}g.stroke();
  const showMarks=c.t[k]>=-0.2;
  if(it.truth&&it.truth.x!=null&&showMarks){const x=X(it.truth.x),y=Y(it.truth.y);g.fillStyle='#f6d23a';g.beginPath();g.moveTo(x,y-9);g.lineTo(x+9,y);g.lineTo(x,y+9);g.lineTo(x-9,y);g.closePath();g.fill();g.strokeStyle='#000';g.lineWidth=1;g.stroke()}
  if(it.det&&detOff!==null&&c.t[k]>=detOff-0.2){const x=X(it.det.x),y=Y(it.det.y);g.strokeStyle='#fff';g.lineWidth=3;g.beginPath();g.moveTo(x-7,y-7);g.lineTo(x+7,y+7);g.moveTo(x+7,y-7);g.lineTo(x-7,y+7);g.stroke()}
  const hl=[it.det&&it.det.actor,it.det&&it.det.receiver];
  c.players.forEach(([j,xs,ys])=>{const x=dm(xs[k]),y=dm(ys[k]);if(x==null)return;g.fillStyle=D.sides[j]===0?css('--home'):css('--away');g.beginPath();g.arc(X(x),Y(y),9,0,7);g.fill();
   if(hl.includes(D.keys[j])){g.strokeStyle='#fff';g.lineWidth=3;g.beginPath();g.arc(X(x),Y(y),13,0,7);g.stroke()}
   g.fillStyle='#fff';g.font='600 11px sans-serif';g.textAlign='center';g.fillText(D.keys[j].split(':')[1],X(x),Y(y)+4)});
  const bx=dm(c.bx[k]),by=dm(c.by[k]);if(bx!=null){g.fillStyle='#f6d23a';g.beginPath();g.arc(X(bx),Y(by),6,0,7);g.fill();g.strokeStyle='#000';g.lineWidth=1;g.stroke()}
  g.textAlign='left';g.fillStyle='#fff';g.font='600 18px sans-serif';g.fillText((c.t[k]>=0?'+':'')+c.t[k].toFixed(1)+' s',14,26);
  if(Math.abs(c.t[k])<0.06){g.fillText(it.truth?'◆ PFF label':'detected',14,50)}
  rng.value=k;tm.textContent=(c.t[k]>=0?'+':'')+c.t[k].toFixed(1)+' s'}
 function loop(ts){if(playing&&ts-last>100){k=(k+1)%c.t.length;last=ts;draw()}anim=requestAnimationFrame(loop)}
 btn.onclick=()=>{playing=!playing;btn.textContent=playing?'Pause':'Play'};rng.oninput=()=>{k=+rng.value;playing=false;btn.textContent='Play';draw()};
 k=Math.max(0,c.t.findIndex(v=>v>=-1));draw();anim=requestAnimationFrame(loop)}

// open from hash
function fromHash(){const h=(location.hash||'').slice(1);if(['results','explorer','rules','gaps'].includes(h))show(h)}
fromHash();window.addEventListener('hashchange',fromHash);
</script></body></html>
"""
