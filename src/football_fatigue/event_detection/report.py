"""Mentor report: one self-contained HTML page built from the evaluation summaries (``summary.json``) of the runs compared.

Every number on the page is read from those files; nothing is typed in. The page holds aggregate scores only (no positions
or event rows), so it carries no source data.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .evaluate import PRIMARY_TOLERANCE_S, TOLERANCES_S, prf
from .features import FEATURES

TYPES = [("PASS_ANY", "Pass"), ("CR", "Cross"), ("SH", "Shot"), ("GOAL", "Goal"), ("KEY_PASS", "Key pass")]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _c(run: dict, kind: str, tol: float = PRIMARY_TOLERANCE_S) -> dict[str, int]:
    return run["merged"]["counts"][f"{kind}@{tol}"]


def _f(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


def _bars(baseline: dict, model: dict) -> str:
    """Horizontal paired bars of F1 per event type, drawn on one 0-1 scale."""
    W, left, right, row, top = 640, 92, 56, 46, 26
    width = W - left - right
    h = top + row * len(TYPES) + 24
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="F1 per event type, rules versus ML" class="chart">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + t * width
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{h - 22}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 6}" class="tick" text-anchor="middle">{t:.2f}</text>')
    for i, (kind, name) in enumerate(TYPES):
        y = top + i * row
        out.append(f'<text x="{left - 10}" y="{y + 19}" class="label" text-anchor="end">{name}</text>')
        for j, (run, cls) in enumerate(((baseline, "b-rules"), (model, "b-ml"))):
            f = prf(_c(run, kind))[2] or 0.0
            yy = y + j * 15
            out.append(f'<rect x="{left}" y="{yy}" width="{max(f * width, 1):.1f}" height="12" rx="2" class="{cls}"/>'
                       f'<text x="{left + f * width + 6:.1f}" y="{yy + 10}" class="val">{f:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def _score_table(runs: list[tuple[str, dict]]) -> str:
    head = "".join(f'<th colspan="3" class="grp">{html.escape(n)}</th>' for n, _ in runs)
    sub = "".join("<th>Precision</th><th>Recall</th><th>F1</th>" for _ in runs)
    rows = []
    for kind, name in TYPES:
        c0 = _c(runs[0][1], kind)
        cells = "".join(f"<td>{_f(p)}</td><td>{_f(r)}</td><td class='f1'>{_f(f)}</td>" for p, r, f in (prf(_c(run, kind)) for _, run in runs))
        rows.append(f"<tr><th scope='row'>{name}</th><td class='n'>{c0['n_truth']:,}</td>{cells}</tr>")
    return (f"<div class='scroll'><table><thead><tr><th rowspan='2'>Event</th><th rowspan='2' class='n'>Labelled</th>{head}</tr>"
            f"<tr>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _tolerance_table(run: dict) -> str:
    rows = "".join(f"<tr><th scope='row'>{name}</th>" + "".join(f"<td>{_f(prf(_c(run, kind, t))[2])}</td>" for t in TOLERANCES_S) + "</tr>"
                   for kind, name in TYPES)
    return ("<div class='scroll'><table><thead><tr><th>Event</th>" + "".join(f"<th>±{t:g} s</th>" for t in TOLERANCES_S)
            + f"</tr></thead><tbody>{rows}</tbody></table></div>")


def _confusion(run: dict) -> str:
    conf = run["merged"]["confusion"]
    names = {"PA": "Pass", "CR": "Cross", "SH": "Shot", "CL": "Clearance", "no_event": "No labelled event"}
    cols = ("PA", "CR", "SH", "missed")
    rows = []
    for t in ("PA", "CR", "SH", "CL", "no_event"):
        vals = [conf.get(f"{t}->{d}", 0) for d in cols]
        total = sum(vals) or 1
        cells = []
        for d, v in zip(cols, vals):
            good = (t == d)
            cells.append(f"<td class='{'hit' if good else ''}'>{v:,}<span class='pct'>{v / total * 100:.0f}%</span></td>"
                         if not (t == "no_event" and d == "missed") else "<td class='na'>–</td>")
        rows.append(f"<tr><th scope='row'>{names[t]}</th>{''.join(cells)}</tr>")
    return ("<div class='scroll'><table class='conf'><thead><tr><th>PFF label ↓ / detected →</th><th>Pass</th><th>Cross</th><th>Shot</th>"
            f"<th>Not detected</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _attributes(run: dict) -> str:
    a = run["merged"]["attributes"]
    p, s = a.get("PASS_ANY", {}), a.get("SH", {})
    items = [("Pass: right team", p.get("team_agree", 0), p.get("matched", 0)),
             ("Pass: right player", p.get("player_agree", 0), p.get("player_comparable", 0)),
             ("Pass: completed or not", p.get("outcome_agree", 0), p.get("outcome_comparable", 0)),
             ("Shot: right team", s.get("team_agree", 0), s.get("matched", 0)),
             ("Shot: right player", s.get("player_agree", 0), s.get("player_comparable", 0))]
    return "".join(f"<div class='stat'><div class='big'>{(k / n * 100 if n else 0):.0f}%</div><div class='lbl'>{lbl}</div>"
                   f"<div class='sm'>{k:,} of {n:,} matched events</div></div>" for lbl, k, n in items)


def render(rules: dict, ml: dict, rules_before: dict | None = None, model_manifest: dict | None = None) -> str:
    n_matches = ml["meta"].get("matches", "the")
    f_ml = {k: prf(_c(ml, k))[2] for k, _ in TYPES}
    f_ru = {k: prf(_c(rules, k))[2] for k, _ in TYPES}
    goal_note = ""
    if rules_before is not None:
        gb, ga = prf(_c(rules_before, "GOAL")), prf(_c(rules, "GOAL"))
        goal_note = (f"<p>Goal rule before and after it was made stricter (rules detector, same test matches): precision "
                     f"{_f(gb[0])} → {_f(ga[0])}, recall {_f(gb[1])} → {_f(ga[1])}, F1 {_f(gb[2])} → {_f(ga[2])}.</p>")
    lead = (f"On {n_matches} held-out World Cup 2022 matches, the ML detector finds passes with "
            f"F1 {_f(f_ml['PASS_ANY'])}, crosses {_f(f_ml['CR'])}, shots {_f(f_ml['SH'])} and goals {_f(f_ml['GOAL'])}, "
            f"against {_f(f_ru['PASS_ANY'])}, {_f(f_ru['CR'])}, {_f(f_ru['SH'])} and {_f(f_ru['GOAL'])} for the hand-written rules.")
    return _PAGE.format(
        lead=html.escape(lead), bars=_bars(rules, ml), table=_score_table([("Rules", rules), ("ML", ml)]), tol=_tolerance_table(ml),
        conf=_confusion(ml), attrs=_attributes(ml), goal_note=goal_note, not_scoreable=f"{ml['meta']['not_scoreable']:,}",
        rules_v=html.escape(rules["meta"]["detector_version"]), ml_v=html.escape(ml["meta"]["detector_version"]),
        n_features=len(FEATURES), n_rows=f"{model_manifest['rows']:,}" if model_manifest else "the",
    )


def write_report(path: Path, rules_summary: Path, ml_summary: Path, rules_before_summary: Path | None = None,
                 model_manifest: Path | None = None) -> Path:
    before = _load(rules_before_summary) if rules_before_summary and Path(rules_before_summary).exists() else None
    manifest = _load(model_manifest) if model_manifest and Path(model_manifest).exists() else None
    page = render(_load(rules_summary), _load(ml_summary), before, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    # plain ASCII with character references: the page reads the same opened from disk (no charset) or published
    path.write_text(page.encode("ascii", "xmlcharrefreplace").decode("ascii"), encoding="utf-8")
    return path


_PAGE = """<title>World Cup Event Detection</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column, like a match report, with full-width figures; pitch green as the one accent. */
:root{{--bg:#f7f8f6;--surface:#ffffff;--ink:#17201b;--muted:#5b675f;--line:#dbe1dc;--accent:#1f7a4c;--accent-2:#9db8a6;--hit:#e3f1e8;
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Source Serif 4",Georgia,serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111614;--surface:#18201c;--ink:#e6ece8;--muted:#9aa8a0;--line:#2a3530;--accent:#4cc38a;--accent-2:#3d5a4a;--hit:#1d3b2b;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#111614;--surface:#18201c;--ink:#e6ece8;--muted:#9aa8a0;--line:#2a3530;--accent:#4cc38a;--accent-2:#3d5a4a;--hit:#1d3b2b;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font:17px/1.6 var(--body);margin:0}}
main{{max-width:880px;margin:0 auto;padding-block:40px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:44px}}
header{{display:grid;gap:10px}}
.eyebrow{{font:500 12px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}}
h1{{font:700 clamp(40px,7vw,64px)/.95 var(--display);margin:0;text-wrap:balance;letter-spacing:.01em}}
h2{{font:700 28px/1.1 var(--display);margin:0 0 12px;letter-spacing:.02em;text-wrap:balance}}
p{{margin:0 0 12px;max-width:66ch}}
.lead{{font-size:20px;line-height:1.5;max-width:60ch}}
section{{display:grid;gap:8px;min-width:0}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font:14px/1.4 var(--mono);font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--muted);font-weight:500;font-size:12px;letter-spacing:.04em}}
th.grp{{text-align:center;color:var(--ink);border-bottom:2px solid var(--accent)}}
td.f1{{font-weight:500;color:var(--accent)}}
td.n{{color:var(--muted)}}
.conf td{{min-width:92px}}.conf td.hit{{background:var(--hit)}}.conf td.na{{color:var(--muted)}}
.pct{{display:block;font-size:11px;color:var(--muted)}}
.chart{{width:100%;height:auto;display:block}}
.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .tick,.chart .val{{font:11px var(--mono);fill:var(--muted)}}
.chart .label{{font:600 15px var(--display);fill:var(--ink);letter-spacing:.03em}}
.chart .b-rules{{fill:var(--accent-2)}}.chart .b-ml{{fill:var(--accent)}}
.key{{display:flex;gap:18px;font:12px var(--mono);color:var(--muted)}}.key i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;counter-reset:s}}
.step{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px;display:grid;gap:4px;min-width:0}}
.step b{{font:700 17px var(--display);letter-spacing:.03em}}
.step b::before{{counter-increment:s;content:counter(s) "  ";color:var(--accent)}}
.step span{{font-size:14px;color:var(--muted);line-height:1.45}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:18px}}
.stat .big{{font:700 38px/1 var(--display);color:var(--accent)}}.stat .lbl{{font-weight:600;font-size:15px}}.stat .sm{{font:12px var(--mono);color:var(--muted)}}
ul{{margin:0;padding-left:20px;max-width:66ch}}li{{margin-bottom:6px}}
code,pre{{font:13px/1.5 var(--mono)}}
pre{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px;overflow-x:auto;margin:0}}
.note{{font-size:14px;color:var(--muted)}}
</style>
<main>
<header>
<div class="eyebrow">Graduation project · Task 2 · Milestone 3A</div>
<h1>World Cup Event Detection</h1>
<p class="lead">{lead}</p>
<p class="note">Detectors <code>{rules_v}</code> and <code>{ml_v}</code>. Input is player and ball positions only; PFF event labels are used for scoring, never as input.</p>
</header>

<section>
<h2>Results on held-out matches</h2>
<div class="key"><span><i style="background:var(--accent-2)"></i>Rules</span><span><i style="background:var(--accent)"></i>ML</span><span>F1 at ±1 s</span></div>
{bars}
{table}
<p class="note">A detection is correct when a PFF label of the same type lies within ±1 s in the same period (one-to-one). {not_scoreable} labels fall outside the tracked time (excluded extra time, period edges) and are not scored. Pass includes crosses; key pass is derived the same way for labels and detections.</p>
</section>

<section>
<h2>How it works</h2>
<div class="flow">
<div class="step"><b>Positions</b><span>Raw player and ball positions per frame (29.97 fps). Missing ball stays missing.</span></div>
<div class="step"><b>Clean the ball</b><span>Drop held repeats (15 Hz source) and single-frame spikes over 40 m/s.</span></div>
<div class="step"><b>Find touches</b><span>Sudden change in ball velocity with a player within reach.</span></div>
<div class="step"><b>Classify</b><span>Rules: fixed thresholds. ML: gradient boosting on {n_features} features, scored for both candidate actors.</span></div>
<div class="step"><b>Post-process</b><span>Receiver, completed or not, goal check, key pass. Export as PFF-style JSON.</span></div>
</div>
<p>The rules version decides with a few hand-set thresholds, for example "within 25 m of goal, faster than 6 m/s, heading between the posts". The ML version keeps the touch finder and learns the decision from {n_rows} labelled (touch, candidate player) rows, using all features at once: speed, direction, distance and angle to goal, where the ball ends up, nearby defenders, ball height, and what happens next.</p>
</section>

<section>
<h2>Keeping the test honest</h2>
<ul>
<li><b>No label leakage.</b> PFF's smoothed ball is built from the event labels, and tracking frames carry embedded event tags. The detector reads neither: raw ball and raw players only.</li>
<li><b>Unseen matches.</b> Rule thresholds were tuned on 8 development matches and reported on the other 56. The ML models use 8 folds by match: every match is scored by a model that never saw it.</li>
<li><b>Nothing dropped silently.</b> Excluded periods (extra time with a known side-label problem) and labels outside tracking are counted and reported.</li>
<li><b>Same rules both sides.</b> Key passes are derived with one definition for labels and detections.</li>
</ul>
</section>

<section>
<h2>Who and what, on correctly timed events</h2>
<div class="stats">{attrs}</div>
</section>

<section>
<h2>Where the ML detector goes wrong</h2>
{conf}
<p class="note">Rows are PFF labels; the last row counts detections with no labelled pass, cross, shot or clearance within 1 s (false alarms). Clearances are not a detector class.</p>
</section>

<section>
<h2>Goals</h2>
<p>The ball is often not tracked into the net, so a goal needs two independent signs: after a detected shot the ball reaches the goal mouth (within 1.5 m), and play then stops for 60 to 300 s, the length of a celebration and kickoff. A pause alone (injury, VAR, time-wasting) or a ball near goal alone (a save) is not enough, and only the last shot before a pause can be the goal. Missed goals are mostly shots that were not detected in the first place.</p>
{goal_note}
</section>

<section>
<h2>Timing tolerance</h2>
{tol}
<p class="note">ML detector F1 when the allowed time difference to the label changes.</p>
</section>

<section>
<h2>Limits</h2>
<ul>
<li>PFF labels are human annotations, not physical ground truth.</li>
<li>The raw ball is missing in about a third of frames and updates at about 15 Hz; 6–9 % of labelled passes have no detectable touch at all.</li>
<li>Player identity comes from the tracking jersey join, and 68 % of raw jersey reads have low confidence.</li>
<li>The models learned PFF tracking. Positions from our video pipeline will be noisier; that needs its own robustness test before results transfer.</li>
</ul>
</section>

<section>
<h2>Reproduce</h2>
<pre>python -m football_fatigue.event_detection evaluate --split test --workers 4
python -m football_fatigue.event_detection train --workers 4
python -m football_fatigue.event_detection evaluate --detector ml --split test --workers 4
python -m football_fatigue.event_detection viewer 10517 --detector ml</pre>
</section>
</main>
"""
