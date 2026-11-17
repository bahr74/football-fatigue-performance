"""CLI: ``python -m football_fatigue.intersections run --match <development id> [--dry-run]``.

Detects trajectory intersections on one development match, builds the four-segment samples and writes them, with a summary,
figures and a manifest, to ``data/processed/m11-intersections-v1/<match>/`` (never overwritten). Validation and locked-test
matches are refused. ``--dry-run`` prints the summary and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import EventVersion, IngestionError, approved_output_root, safe_output_path
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import SOURCE_DIR, jsonl, load_source

from . import figures as FG
from .detect import DetectConfig, Timeline, detect, duplicate_positions
from .samples import SCHEMA, SampleConfig, build_samples

OUTPUT_DIR = SCHEMA
DT_BINS = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
SUFFIXES = (".events.jsonl", ".instances.jsonl", ".references.jsonl", ".exclusions.jsonl", ".summary.json", ".config.json")


def code_digest() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def peak_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)


def write_dir(out: Path, blobs: dict[str, bytes]) -> None:
    """Write the whole match directory or nothing: a temporary sibling directory, renamed at the end; never replaces one."""
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    tmp = out.parent / f".{out.name}.partial-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        for name, b in blobs.items():
            with open(tmp / name, "xb") as f:
                f.write(b)
        if out.exists():
            raise IngestionError(f"{out} appeared while writing; not overwritten")
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def _bin(v: float) -> str:
    for lo, hi in zip(DT_BINS, DT_BINS[1:]):
        if v < hi:
            return f"{lo:g}-{hi:g}s"
    return f">={DT_BINS[-1]:g}s"


def _quant(v, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) -> dict[str, float] | None:
    v = np.asarray([x for x in v if x is not None], float)
    return {f"q{int(q * 100)}": round(float(np.quantile(v, q)), 4) for q in qs} if len(v) else None


def diagnostics(insts: list[dict], refs: list[dict]) -> dict[str, Any]:
    """Leakage checks and two rule-based difficulty indicators (no fitting): does anything but motion reveal the target?"""
    y = np.array([r["target"] for r in refs])
    n = lambda tr: len(tr["t"])  # noqa: E731
    len_diff = np.array([n(i["outgoing"][0]) - n(i["outgoing"][1]) for i in insts])
    start_diff = np.array([i["outgoing"][0]["t"][0] - i["outgoing"][1]["t"][0] for i in insts])
    in_len_diff = np.array([n(i["incoming"][0]) - n(i["incoming"][1]) for i in insts])

    def corr(v):
        return dict(differs_share=round(float(np.mean(np.abs(v) > 1e-9)), 4),
                    corr_with_target=None if v.std() == 0 else round(float(np.corrcoef(v, y)[0, 1]), 4))

    def nearest(i):
        a = i["incoming"][0]
        d = [np.hypot(o["x"][0] - a["x"][-1], o["y"][0] - a["y"][-1]) for o in i["outgoing"]]
        return int(np.argmin(d))

    def const_vel(i):
        a = i["incoming"][0]
        t = np.asarray(a["t"])
        k = t >= t[-1] - 0.5
        if k.sum() < 2:
            return nearest(i)
        vx, vy = np.polyfit(t[k], np.asarray(a["x"])[k], 1)[0], np.polyfit(t[k], np.asarray(a["y"])[k], 1)[0]
        d = [np.hypot(o["x"][0] - (a["x"][-1] + vx * (o["t"][0] - t[-1])), o["y"][0] - (a["y"][-1] + vy * (o["t"][0] - t[-1]))) for o in i["outgoing"]]
        return int(np.argmin(d))

    rules = {}
    bins = np.array([_bin(r["abs_time_diff_s"]) for r in refs])
    for name, f in (("nearest_position", nearest), ("constant_velocity", const_vel)):
        pred = np.array([f(i) for i in insts])
        rules[name] = dict(all=round(float((pred == y).mean()), 4),
                           by_abs_time_diff={b: dict(n=int((bins == b).sum()), accuracy=round(float((pred[bins == b] == y[bins == b]).mean()), 4))
                                             for b in sorted(set(bins), key=lambda s: float(s.split("-")[0].rstrip("s>=")))})
    return dict(target_balance=dict(Counter(int(v) for v in y)), target_share_out1=round(float(y.mean()), 4),
                outgoing_length_difference=corr(len_diff), outgoing_start_difference=corr(start_diff), incoming_length_difference=corr(in_len_diff),
                instance_keys=sorted({k for i in insts for k in i}), tracklet_keys=sorted({k for i in insts for tr in i["incoming"] + i["outgoing"] for k in tr}),
                rule_based_agreement_with_provider=rules,
                note="rule-based indicators of task difficulty on development data; nothing is fitted or tuned")


def summarize(events, stats, insts, refs, excl, seconds) -> dict[str, Any]:
    sampled = {r["event_id"] for r in refs}
    by_type = Counter(e["type"] for e in events)
    return dict(
        schema=SCHEMA, detection=stats,
        events=dict(total=len(events), by_type=dict(by_type), by_period=dict(Counter(str(e["period"]) for e in events)),
                    by_team_relation=dict(Counter(e["team_relation"] for e in events)),
                    abs_time_diff_s=_quant(e["abs_time_diff_s"] for e in events),
                    abs_time_diff_bins=dict(Counter(_bin(e["abs_time_diff_s"]) for e in events)),
                    simultaneous=int(sum(e["simultaneous"] for e in events)),
                    min_synchronous_distance_m=_quant(e["min_synchronous_distance_m"] for e in events),
                    windows_per_event=dict(Counter(str(e["n_windows"]) for e in events)),
                    quality_flags={k: int(sum(e["quality"][k] for e in events)) for k in (events[0]["quality"] if events else {})}),
        samples=dict(valid=len(insts), excluded=len(excl), exclusion_reasons=dict(Counter(x["reason"] for x in excl)),
                     by_event_type=dict(Counter(r["event_type"] for r in refs)), by_abs_time_diff=dict(Counter(_bin(r["abs_time_diff_s"]) for r in refs)),
                     clusters_with_samples=len({e["cluster_id"] for e in events if e["event_id"] in sampled}),
                     diagnostics=diagnostics(insts, refs) if insts else None),
        seconds=seconds,
        note=("Geometric trajectory intersection is the advisor's operational occlusion definition; it is not verified visual "
              "occlusion. Labels are provider trajectory continuity (slot labels), not verified identity."))


def pick_examples(events: list[dict], refs: list[dict]) -> list[tuple[str, dict]]:
    """Deterministic examples by fixed rules, in time order (never chosen by any model output)."""
    sampled = {r["event_id"] for r in refs}
    rules = [("simultaneous_crossing", lambda e: e["type"] == "proper_crossing" and e["simultaneous"] and e["event_id"] in sampled),
             ("crossing_1_to_2s_apart", lambda e: e["type"] == "proper_crossing" and 1 <= e["abs_time_diff_s"] < 2 and e["event_id"] in sampled),
             ("crossing_over_3s_apart_excluded", lambda e: e["abs_time_diff_s"] > 3),
             ("vertex_crossing", lambda e: e["type"] == "vertex_crossing"), ("touch", lambda e: e["type"] == "touch"),
             ("collinear_overlap", lambda e: e["type"] == "collinear_overlap")]
    out = [(name, next((e for e in events if f(e)), None)) for name, f in rules]
    big = max(events, key=lambda e: (e["cluster_size"], -e["time_a_s"]), default=None)
    out.append(("largest_pair_cluster", big))
    return [(n, e) for n, e in out if e is not None]


def run(raw_root: str, match_id: str, dry_run: bool, dcfg: DetectConfig, scfg: SampleConfig) -> dict[str, Any]:
    t_start = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    base = approved_output_root(raw_root) / OUTPUT_DIR
    paths = {s: safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=s) for s in SUFFIXES + (".manifest.json",)}
    if not dry_run and paths[".manifest.json"].parent.exists():
        raise IngestionError(f"{paths['.manifest.json'].parent} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    qpath = safe_output_path(raw_root, match_id, EventVersion.ROOT, approved_output_root(raw_root) / SOURCE_DIR, suffix=".quality.npz")
    with np.load(qpath) as z:
        steps = {k[5:]: z[k] for k in z.files if k.startswith("step_")}
    t_load = time.perf_counter()
    events, stats = detect(c, mask, steps, dcfg, quarantine=tuple(c.quarantine_periods))
    t_det = time.perf_counter()
    tl, dup = Timeline(c), duplicate_positions(c)
    insts, refs, excl = build_samples(c, mask, events, scfg, tl=tl, dup=dup)
    t_smp = time.perf_counter()
    seconds = dict(load=round(t_load - t_start, 1), detect=round(t_det - t_load, 1), samples=round(t_smp - t_det, 1))
    summary = summarize(events, stats, insts, refs, excl, seconds)
    summary["peak_memory_mb"] = peak_memory_mb()
    if dry_run:
        return summary
    ref_by_event = {r["event_id"]: r for r in refs}
    figs: dict[str, bytes] = {}
    examples = []
    for name, ev in pick_examples(events, refs):
        figs[f".fig-{name}.png"] = FG.event_figure(c, tl, ev, ref_by_event.get(ev["event_id"]), name.replace("_", " "))
        examples.append(dict(name=name, event_id=ev["event_id"], file=f"root.fig-{name}.png"))
    figs[".fig-overview.png"] = FG.overview_figure(events, refs)
    d = FG.duplicate_figure(c, tl, dup)
    if d:
        figs[".fig-duplicate_position_artifact.png"] = d[0]
        summary["duplicate_position_example"] = d[1]
    summary["examples"] = examples
    config = dict(detect=dcfg.as_dict(), samples=scfg.as_dict(), dt_bins_s=DT_BINS)
    blobs = {".events.jsonl": jsonl(events), ".instances.jsonl": jsonl(insts), ".references.jsonl": jsonl(refs), ".exclusions.jsonl": jsonl(excl),
             ".summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(), ".config.json": json.dumps(config, indent=1, sort_keys=True).encode(),
             **figs}
    for sfx in figs:
        paths[sfx] = safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=sfx)
    out = paths[".manifest.json"].parent
    snap1 = validate.raw_snapshot(raw_root)
    manifest = dict(schema=SCHEMA, match_id=int(c.match_id), split=split["matches"][str(match_id)], config=config, code_sha256=code_digest(),
                    source=prov, source_schema=src_manifest["schema"], coordinate_policy="raw", provider_smoothed_coordinates_used=False,
                    raw_snapshot_before=snap0, raw_snapshot_after=snap1, raw_unchanged=snap0 == snap1,
                    outputs_sha256={paths[sfx].name: hashlib.sha256(b).hexdigest() for sfx, b in blobs.items()},
                    python=platform.python_version(), numpy=np.__version__, seconds=seconds,
                    model_facing_files=["root.instances.jsonl"], evaluation_only_files=["root.references.jsonl", "root.events.jsonl", "root.exclusions.jsonl"])
    blobs[".manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_dir(out, {paths[sfx].name: b for sfx, b in blobs.items()})
    summary["written_to"] = str(paths[".manifest.json"].parent)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections")
    ap.add_argument("--root", default="data/raw")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="detect intersections and build four-segment samples for one development match")
    r.add_argument("--match", required=True)
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = run(str(Path(a.root).resolve()), a.match, a.dry_run, DetectConfig(), SampleConfig())
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
