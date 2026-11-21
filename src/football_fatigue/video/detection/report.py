"""Markdown tables over every diagnosed development source (diag-v1 + replay-v1 outputs)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from football_fatigue.video.detection import DIAG_VERSION
from football_fatigue.video.detection.stages import LOST_STAGES, SIZE_BINS, STAGES

ORDER = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
VIS = ("visible", "partial", "heavy", "hidden")
ANY_BOX = ("tracked", "detected_untracked", "below_threshold", "turf_filtered")


def _summaries(out_root: Path) -> dict[str, dict]:
    out = {}
    for s in ORDER:
        p = Path(out_root) / s / "detection" / DIAG_VERSION / "summary.json"
        if p.exists():
            out[s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _rate(sv: dict, stages, vis) -> tuple[float, int]:
    n = sum(sv.get(s, {}).get(v, 0) for s in STAGES for v in vis)
    k = sum(sv.get(s, {}).get(v, 0) for s in stages for v in vis)
    return (k / n if n else float("nan")), n


def recall_rows(summ: dict) -> list[str]:
    rows = ["| match | role | visibility | GT boxes | tracker output | tracker input (conf>0.4, turf ok) | any cached box (conf>0.1) |",
            "|---|---|---|---|---|---|---|"]
    for s, d in summ.items():
        sv = d["stage_visibility"]
        for v in VIS:
            out, n = _rate(sv, ("tracked",), (v,))
            inp, _ = _rate(sv, ("tracked", "detected_untracked"), (v,))
            anyb, _ = _rate(sv, ANY_BOX, (v,))
            rows.append(f"| {s[-6:]} | {d['role']} | {v} | {n} | {out:.3f} | {inp:.3f} | {anyb:.3f} |")
    return rows


def size_rows(summ: dict) -> list[str]:
    names = [n for _lo, _hi, n in SIZE_BINS]
    rows = ["| match | " + " | ".join(f"{n} (n)" for n in names) + " |", "|---|" + "---|" * len(names)]
    for s, d in summ.items():
        c = Counter()
        for key, n in d["stage_visibility_size"].items():
            stage, vis, size = key.split("|")
            if vis != "visible":
                continue
            c[(size, "n")] += n
            if stage in ("tracked", "detected_untracked"):
                c[(size, "k")] += n
        rows.append(f"| {s[-6:]} | " + " | ".join(
            f"{c[(n, 'k')] / c[(n, 'n')]:.3f} ({c[(n, 'n')]})" if c[(n, "n")] else "–" for n in names) + " |")
    return rows


def loss_rows(summ: dict) -> list[str]:
    rows = ["| match | visible GT boxes | lost before tracker | " + " | ".join(LOST_STAGES) + " |",
            "|---|---|---|" + "---|" * len(LOST_STAGES)]
    for s, d in summ.items():
        sv = d["stage_visibility"]
        n = sum(sv.get(st, {}).get("visible", 0) for st in STAGES)
        lost = {st: sv.get(st, {}).get("visible", 0) for st in LOST_STAGES}
        tot = sum(lost.values())
        rows.append(f"| {s[-6:]} | {n} | {tot} ({tot / n:.1%}) | " + " | ".join(
            f"{lost[st]} ({lost[st] / tot:.0%})" for st in LOST_STAGES) + " |")
    return rows


def precision_rows(summ: dict) -> list[str]:
    rows = ["| match | AP50 (all boxes) | tracker-input boxes | TP | duplicate | poor overlap | off pitch | referee label | other non-GT on pitch | precision vs 22-player GT |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in summ.items():
        c = d["tracker_input_detections"]
        n = sum(c.values())
        rows.append(f"| {s[-6:]} | {d['ap50']['all_person_boxes']:.3f} | {n} | {c.get('tp', 0)} | {c.get('duplicate', 0)} | "
                    f"{c.get('poor_overlap', 0)} | {c.get('off_pitch', 0)} | {c.get('referee_label', 0)} | "
                    f"{c.get('fp_on_pitch', 0)} | {c.get('tp', 0) / n:.3f} |")
    return rows


def sweep_rows(summ: dict) -> list[str]:
    rows = ["| match | threshold | recall | TP | non-GT boxes (dup / off-pitch / referee / other) |", "|---|---|---|---|---|"]
    for s, d in summ.items():
        for t in ("0.1", "0.2", "0.3", "0.4", "0.5"):
            c = d["threshold_sweep"][t]
            rows.append(f"| {s[-6:]} | {t} | {c['tp'] / c['gt']:.3f} | {c['tp']} | {c['duplicate']} / {c['off_pitch']} / "
                        f"{c['referee_label']} / {c['fp_on_pitch']} |")
    return rows


def xref_rows(summ: dict) -> list[str]:
    rows = ["| match | engine 'detector miss' events | turf filter | no detection (model) | below threshold | merge suspect | other GT stage | no GT player (referee/staff/FP) |",
            "|---|---|---|---|---|---|---|---|"]
    for s, d in summ.items():
        ev = d["engine_xref"]["events_by_dominant_stage"].get("detection_loss/detector_miss", {})
        tot = sum(ev.values())
        grp = Counter()
        for k, n in ev.items():
            st = k.split("|")[0]
            grp[st if st in ("turf_filtered", "no_detection", "below_threshold", "merge_suspect", "no_gt_player")
                else "other"] += n
        rows.append(f"| {s[-6:]} | {tot} | " + " | ".join(
            f"{grp[k]} ({grp[k] / tot:.0%})" for k in ("turf_filtered", "no_detection", "below_threshold",
                                                      "merge_suspect", "other", "no_gt_player")) + " |")
    return rows


def replay_rows(out_root: Path) -> list[str]:
    rows = ["| match | role | variant | input filter | HOTA | DetRe | DetPr | AssA | IDF1 | IDSW | tracks | HOTA on-pitch | DetPr on-pitch |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ORDER:
        base = Path(out_root) / s / "detection" / "replay-v1"
        for p in sorted(base.glob("*/result.json")) if base.exists() else []:
            r = json.loads(p.read_text(encoding="utf-8"))
            m, mp = r["metrics_all_tracks"], r["metrics_on_pitch_tracks"]
            rows.append(f"| {s[-6:]} | {r['role']} | {r['variant']['name']} | {r['variant']['note']} | {m['hota']['HOTA']:.3f} | "
                        f"{m['hota']['DetRe']:.3f} | {m['hota']['DetPr']:.3f} | {m['hota']['AssA']:.3f} | "
                        f"{m['identity']['IDF1']:.3f} | {m['clear']['IDSW']} | {m['counts']['tracker_ids']} | "
                        f"{mp['hota']['HOTA']:.3f} | {mp['hota']['DetPr']:.3f} |")
    return rows


def markdown(out_root: Path) -> str:
    summ = _summaries(out_root)
    parts = [("Where visible GT players are lost before the tracker", loss_rows(summ)),
             ("Recall by visibility (stage reached)", recall_rows(summ)),
             ("Tracker-input recall of visible players by GT box height", size_rows(summ)),
             ("Tracker-input precision split (conf > 0.4, turf filter as production)", precision_rows(summ)),
             ("Confidence-threshold sweep on cached boxes (turf filter as production)", sweep_rows(summ)),
             ("What the engine's 'detector miss' events really were", xref_rows(summ)),
             ("Tracker-input A/B (ByteTrack replay, identical cached detections)", replay_rows(out_root))]
    return "\n\n".join(f"### {t}\n\n" + "\n".join(r) for t, r in parts)
