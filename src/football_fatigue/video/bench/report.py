"""Markdown tables from saved benchmark outputs (``_bench/report-v1.md``); regenerated from results, never edited."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from football_fatigue.video.bench import detectors as BD
from football_fatigue.video.bench import evaluate as BE
from football_fatigue.video.bench import trackers as BT
from football_fatigue.video.bench.registry import as_rows


def _f(v, n=3):
    return "–" if v is None else f"{v:.{n}f}" if isinstance(v, float) else str(v)


def detector_tables(out_root: Path) -> list[str]:
    ev = BE.load_eval(out_root)
    runs = BD.load_runs(out_root)
    L = ["| run | device | frames | group | AP50 | AP50:95 | AP50 (hidden ignored) | P@0.4 | R@0.4 | FP/frame@0.4 | dup/100 TP | "
         "tune conf | P@tune | R@tune | F1@tune | s/frame |", "|" + "---|" * 16]
    for k, r in ev.items():
        v = r["views"]["on_pitch"]
        for grp in ("tune", "check", "all"):
            if grp not in v:
                continue
            g, p, t = v[grp], v[grp]["at_production_conf"], v[grp]["at_tune_selected_conf"] or {}
            sec = np.mean([x["mean_s_per_frame"] for x in r["latency"].values()])
            L.append(f"| {k}{' (subset)' if runs[k].get('subset') else ''} | {r['device']} | {g['frames']} | {grp} | {_f(g['AP50'])} | "
                     f"{_f(g['AP50_95'])} | {_f(g['AP50_ignore_hidden'])} | {_f(p['precision'])} | {_f(p['recall'])} | "
                     f"{_f(p['fp_per_frame'], 2)} | {_f(p['duplicates_per_100_tp'], 1)} | {_f(v.get('tune_selected_conf'), 2)} | "
                     f"{_f(t.get('precision'))} | {_f(t.get('recall'))} | {_f(t.get('f1'))} | {_f(float(sec), 2)} |")
    L += ["", "Recall at conf > 0.4 by GT height and by annotated visibility (all 4 matches, on_pitch view):", "",
          "| run | <35 px | 35-50 px | 50-80 px | >=80 px | visible | partial | heavy | hidden | uncertain (border) |", "|" + "---|" * 10]
    for k, r in ev.items():
        p = r["views"]["on_pitch"]["all"]["at_production_conf"]
        sz, vi = p["recall_by_size"], p["recall_by_visibility"]
        cell = lambda d, s: f"{d[s]['recall']:.3f} (n={d[s]['n']})" if s in d else "–"  # noqa: E731
        L.append(f"| {k} | " + " | ".join(cell(sz, s) for s in ("<35 px", "35-50 px", "50-80 px", ">=80 px")) + " | "
                 + " | ".join(cell(vi, s) for s in ("visible", "partial", "heavy", "hidden", "uncertain")) + " |")
    L += ["", "AP50 / recall@0.4 by stratum (on_pitch):", ""]
    strata = sorted({s for r in ev.values() for s in r["views"]["on_pitch"]["by_stratum"]})
    L += ["| run | " + " | ".join(strata) + " |", "|" + "---|" * (len(strata) + 1)]
    for k, r in ev.items():
        b = r["views"]["on_pitch"]["by_stratum"]
        L.append(f"| {k} | " + " | ".join(f"{b[s]['AP50']:.3f} / {b[s]['recall']:.3f} (n={b[s]['frames']})" if s in b else "–" for s in strata) + " |")
    L += ["", "Views compared (AP50, all frames): all person boxes / on_pitch / no_referee:", "", "| run | all | on_pitch | no_referee |", "|---|---|---|---|"]
    for k, r in ev.items():
        vv = r["views"]
        L.append(f"| {k} | {_f(vv['all']['all']['AP50'])} | {_f(vv['on_pitch']['all']['AP50'])} | "
                 f"{_f(vv['no_referee']['all']['AP50']) if 'no_referee' in vv else '–'} |")
    return L


def tracker_tables(out_root: Path) -> list[str]:
    res = BT.load_results(out_root)
    L = ["| match | role | input | tracker | HOTA | DetA | AssA | IDF1 | IDSW | Frag | track ids | ms/frame | RSS +MB | dropped rows |",
         "|" + "---|" * 14]
    agg = defaultdict(list)
    for r in sorted(res, key=lambda r: (r["source_id"], r["input"], -r["metrics_all_tracks"]["hota"]["HOTA"])):
        m = r["metrics_all_tracks"]
        L.append(f"| {r['source_id'][12:]} | {r['role']} | {r['input']} | {r['tracker']} | {m['hota']['HOTA']:.4f} | "
                 f"{m['hota']['DetA']:.4f} | {m['hota']['AssA']:.4f} | {m['identity']['IDF1']:.4f} | {m['clear']['IDSW']} | "
                 f"{m['clear']['Frag']} | {r['track_ids']} | {r['ms_per_frame']:.2f} | {r['rss_increase_mb']} | {r['unmeasured_rows_dropped']} |")
        agg[(r["input"], r["tracker"], r["role"])].append(r)
    L += ["", "Mean over matches by role (tune = 117093, 118575; check = 117092, 118576):", "",
          "| input | tracker | role | n | HOTA | AssA | IDF1 | IDSW (sum) | Frag (sum) |", "|" + "---|" * 9]
    for (inp, t, role), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][2], -np.mean([x["metrics_all_tracks"]["hota"]["HOTA"] for x in kv[1]]))):
        mm = [x["metrics_all_tracks"] for x in rs]
        L.append(f"| {inp} | {t} | {role} | {len(rs)} | {np.mean([m['hota']['HOTA'] for m in mm]):.4f} | "
                 f"{np.mean([m['hota']['AssA'] for m in mm]):.4f} | {np.mean([m['identity']['IDF1'] for m in mm]):.4f} | "
                 f"{sum(m['clear']['IDSW'] for m in mm)} | {sum(m['clear']['Frag'] for m in mm)} |")
    return L


def write_report(data_root: Path, out_root: Path, project_root: Path) -> Path:
    runs = BD.load_runs(out_root)
    L = ["# Session B benchmark tables (generated)", "", "## Detector candidates", "",
         "| status | key | family | track | params (M) | GFLOPs@640 | licence | training data |", "|" + "---|" * 8]
    for r in as_rows(project_root, runs):
        L.append(f"| {r['status']} | {r['key']} | {r['family']} | {r['track']} | {_f(r['params_m'], 2)} | {_f(r['gflops_640'], 1)} | "
                 f"{r['licence']} | {r['training_data'][:90]} |")
    L += ["", "## Detector accuracy (benchmark frames, on_pitch view)", ""] + detector_tables(out_root)
    L += ["", "## Trackers on identical detections (full matches)", ""] + tracker_tables(out_root)
    dest = Path(out_root) / "_bench" / "report-v1.md"
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")
    return dest
