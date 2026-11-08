"""One match end to end: positions, detection, optional export, scoring, and viewer (importable for worker processes)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from football_fatigue.ingestion import Dataset, EventVersion

from . import detect
from . import evaluate as E
from .config import COORDINATE_FLAGGED_MATCHES, DETECTOR_VERSION, DEV_MATCHES, EXCLUDED_PERIODS, ML_CANDIDATE_OVERRIDES, DetectorConfig
from .export import manifest, to_pff_like, write_run
from .features import actor_rows, touch_features
from .inputs import load_positions
from .kinematics import ball_track
from .ml import ML_VERSION, TouchClassifier, events_from_rows, row_targets
from .touches import detect_touches
from .viewer import write_viewer

DEFAULT_MODEL_DIR = Path("models/event_detection") / ML_VERSION

NOTES = [
    "Input: raw tracking positions only (raw ball, raw players). `ballsSmoothed` is not used because PFF builds it from the "
    "event labels; the event overlays embedded in tracking frames are not read. PFF events are used only for scoring.",
    "Thresholds were tuned on the dev matches only " + ", ".join(map(str, DEV_MATCHES)) + "; the test split excludes them.",
    "Excluded periods: extra time of " + ", ".join(map(str, sorted(EXCLUDED_PERIODS))) + " (suspected side/identity anomaly or no tracking).",
    "Matches " + ", ".join(map(str, sorted(COORDINATE_FLAGGED_MATCHES))) + " are flagged: their event coordinates disagree with the "
    "supplied tracking, so labels and positions may come from different tracking releases.",
    "A detection counts as correct when a labelled event of the same type lies within the tolerance in the same period (one-to-one). "
    "Key passes are derived the same way for labels and detections: the last completed pass/cross of the shooting team before a shot, "
    "with no opponent action in between, within 10 s.",
    "PFF labels are human annotations, not physical ground truth. Raw ball positions are missing in about a third of frames and update "
    "at about 15 Hz; goals in particular are often not visible in the ball track and are detected mainly from the stoppage that follows.",
]

ML_NOTES = [
    "ML detector: the touch finder runs with looser thresholds (more candidates); a gradient-boosted classifier (scikit-learn "
    "HistGradientBoosting) on 36 position-only features labels each touch as PA, CR, SH, CL or none. Pass outcome, receiver, goal "
    "and key pass use the same rules as the rules detector.",
    "Out-of-fold scoring: the 64 matches are dealt into 8 folds; each match is scored by the model trained on the other 7 folds, "
    "so no match is scored by a model that saw it. Model hyperparameters are fixed; the one tuned value, the event-probability "
    "threshold, is chosen on the dev matches' out-of-fold predictions only.",
]


def run_match(match_id: int, root: str, cache: str | None, write_out: str | None, overwrite: bool, score: bool,
              viewer_dir: str | None = None, detector: str = "rules", model_dir: str | None = None) -> dict[str, Any]:
    pos = load_positions(root, match_id, cache_dir=Path(cache) if cache else None, exclude_periods=EXCLUDED_PERIODS.get(match_id, ()))
    model_info = None
    if detector == "ml":
        cfg = ml_config()
        ball = ball_track(pos, cfg)
        touches = detect_touches(pos, ball, cfg)
        clf = TouchClassifier(Path(model_dir or DEFAULT_MODEL_DIR))
        rows = actor_rows(touches)
        X = touch_features(pos, ball, touches, cfg, rows)
        name, proba = clf.predict_proba(match_id, X)
        events = events_from_rows(pos, ball, touches, rows, proba, clf.predict_completion(match_id, X), cfg, clf.manifest.get("event_threshold"))
        version = ML_VERSION
        ball_stats, model_info = ball.stats, {"dir": str(clf.dir), "model": name, "trained_without_this_match": name != "full"}
    else:
        cfg = DetectorConfig()
        res = detect(pos, cfg)
        events, touches, ball_stats, version = res.events, res.touches, res.ball_stats, DETECTOR_VERSION
    out: dict[str, Any] = {"match_id": match_id, "events": len(events), "touches": len(touches)}
    ds = Dataset(root)
    if write_out:
        trk, _ = ds.tracking_source(match_id)
        rows = ds.load_events(match_id, EventVersion.ROOT)
        ev_path = Path(root) / rows[0].source.path if rows and rows[0].source is not None else None
        sources = {"tracking": trk, **({"events_root_for_scoring_only": ev_path} if ev_path else {})}
        out["written"] = str(write_run(Path(write_out), Path(root), match_id, to_pff_like(pos, events, version),
                                       manifest(pos, cfg, sources, ball_stats, len(touches), len(events), version, model_info),
                                       overwrite=overwrite, version=version))
    if score:
        truth, outside = E.scoreable(E.truth_events(ds.load_events(match_id, EventVersion.ROOT), cfg), pos)
        s = E.score(E.detected_as_labelled(pos, events), truth)
        s["not_scoreable"] = len(outside)
        s["flags"] = (["coordinate_flagged"] if match_id in COORDINATE_FLAGGED_MATCHES else []) +                      ([f"excluded_periods={list(pos.excluded_periods)}"] if pos.excluded_periods else []) +                      ([f"model={model_info['model']}"] if model_info else [])
        out["score"] = s
        if viewer_dir:
            dest = Path(viewer_dir) / version / f"viewer_{match_id}.html"
            out["written"] = str(write_viewer(dest, pos, events, truth, outside, touches, s, viewer_names(ds, match_id, pos),
                                              viewer_match(ds.metadata(match_id)), cfg, version,
                                              dict(model_info, event_threshold=clf.manifest.get("event_threshold"), rows=clf.manifest.get("rows")) if model_info else None))
    return out


_STAGES = {4: "Round of 16", 5: "Quarter-final", 6: "Semi-final", 7: "Third place", 8: "Final"}   # inferred from metadata weeks (audit §2)


def viewer_match(meta) -> dict[str, Any]:
    stage = _STAGES.get(meta.matchweek) if meta.matchweek and meta.matchweek > 3 else (f"Group stage, matchday {meta.matchweek}" if meta.matchweek else None)
    return {"home": meta.home_team_name, "away": meta.away_team_name, "date": (meta.kickoff_date or "")[:10],
            "stage": "FIFA World Cup 2022" + (f" · {stage}" if stage else "")}


def viewer_names(ds: Dataset, match_id: int, pos) -> dict[str, Any]:
    """Display names: player key ("home:10") and player id -> "Name (#10)" from the match roster."""
    by_key, by_id = {}, {}
    for e in ds.roster(match_id).entries:
        label = f"{e.nickname or e.player_id} (#{e.shirt_number})"
        by_key[f"{e.roster_side}:{e.shirt_number}"] = label
        by_id[e.player_id] = label
    return {"by_key": {k: by_key[k] for k in pos.player_key if k in by_key}, "by_id": by_id}


def ml_config() -> DetectorConfig:
    return replace(DetectorConfig(), **ML_CANDIDATE_OVERRIDES)


def touch_table_rows(match_id: int, root: str, cache: str | None) -> tuple[int, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Features and targets of every (candidate touch, candidate actor) row of one match (ML training data)."""
    cfg = ml_config()
    pos = load_positions(root, match_id, cache_dir=Path(cache) if cache else None, exclude_periods=EXCLUDED_PERIODS.get(match_id, ()))
    ball = ball_track(pos, cfg)
    touches = detect_touches(pos, ball, cfg)
    truth, _ = E.scoreable(E.truth_events(Dataset(root).load_events(match_id, EventVersion.ROOT), cfg), pos)
    rows = actor_rows(touches)
    y, done = row_targets(pos, touches, rows, truth)
    return match_id, touch_features(pos, ball, touches, cfg, rows), y, done


CALIBRATION_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def calibrate_threshold(root: str, cache: str | None, model_dir: Path, matches=DEV_MATCHES) -> dict[str, Any]:
    """Pick the event threshold on the dev matches only (their out-of-fold predictions); the test matches are not used.

    Objective: mean F1 at the primary tolerance of passes (incl. crosses), crosses, and shots."""
    clf = TouchClassifier(model_dir)
    cfg = ml_config()
    prepared = []
    for m in matches:
        pos = load_positions(root, m, cache_dir=Path(cache) if cache else None, exclude_periods=EXCLUDED_PERIODS.get(m, ()))
        ball = ball_track(pos, cfg)
        touches = detect_touches(pos, ball, cfg)
        rows = actor_rows(touches)
        X = touch_features(pos, ball, touches, cfg, rows)
        _, proba = clf.predict_proba(m, X)
        truth, _ = E.scoreable(E.truth_events(Dataset(root).load_events(m, EventVersion.ROOT), cfg), pos)
        prepared.append((pos, ball, touches, rows, proba, clf.predict_completion(m, X), truth))
    table = {}
    for thr in CALIBRATION_GRID:
        merged = E.merge_scores(E.score(E.detected_as_labelled(pos, events_from_rows(pos, ball, touches, rows, proba, pc, cfg, thr)), truth)
                                for pos, ball, touches, rows, proba, pc, truth in prepared)
        f = {k: E.prf(merged["counts"][f"{k}@{E.PRIMARY_TOLERANCE_S}"])[2] or 0.0 for k in ("PASS_ANY", "CR", "SH", "GOAL", "KEY_PASS")}
        table[str(thr)] = {k: round(v, 4) for k, v in f.items()} | {"objective": round((f["PASS_ANY"] + f["CR"] + f["SH"]) / 3, 4)}
    best = max(CALIBRATION_GRID, key=lambda t: table[str(t)]["objective"])
    return {"event_threshold": best, "threshold_calibration": {"matches": list(matches), "grid": table}}
