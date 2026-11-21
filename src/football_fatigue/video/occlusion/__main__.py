"""CLI: ``python -m football_fatigue.video.occlusion run …`` (see docs/video_occlusion_step1.md)."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import football_fatigue
from football_fatigue.video import ENGINE_VERSION
from football_fatigue.video.occlusion.engine import PROFILES, OcclusionEngine
from football_fatigue.video.schema import SCHEMA_NAME, SCHEMA_VERSION
from football_fatigue.video.sources import SOURCES, SourceError, guard_not_reserved, load_inputs, load_source, play_area_polygon
from football_fatigue.video.tracking import DEFAULT_TRACKER_INPUT, TRACKER_INPUTS, check_name, tracks_dir

PACKAGE_ROOT = Path(football_fatigue.__file__).resolve().parent


def import_roots() -> dict[str, str]:
    """Where the code actually came from; every module must resolve inside this checkout."""
    import football_fatigue.multicam_lab.media as media
    import football_fatigue.video as video
    import football_fatigue.vision.detection_cache as cache

    roots = {name: str(Path(m.__file__).resolve()) for name, m in
             (("football_fatigue", football_fatigue), ("video", video), ("vision.detection_cache", cache),
              ("multicam_lab.media", media))}
    for name, path in roots.items():
        if not Path(path).is_relative_to(PACKAGE_ROOT):
            raise SystemExit(f"import isolation failed: {name} resolves to {path}, outside {PACKAGE_ROOT}")
    return roots


def git_state() -> dict[str, object]:
    repo = PACKAGE_ROOT.parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def cmd_run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.is_relative_to(data_root):
        raise SystemExit(f"--out-root {out_root} is inside the read-only data root {data_root}; choose a separate location")
    spec = SOURCES.get(args.source, {})
    tracker_input = check_name(args.tracker_input)
    if tracker_input != "B0" and args.replay_run:
        raise SystemExit("--replay-run names a production (B0) replay; it cannot be combined with --tracker-input "
                         f"{tracker_input}")
    args.cache_run = args.cache_run or spec.get("cache_run")
    args.replay_run = args.replay_run or spec.get("replay_run")
    if not args.cache_run or not args.replay_run:
        raise SystemExit("--cache-run and --replay-run are required for this source")
    guard_not_reserved(args.source, args.cache_run, args.replay_run)
    vision_dir = data_root / "processed" / "vision"
    replay_dir = vision_dir / args.replay_run if tracker_input == "B0" else tracks_dir(data_root, out_root, args.source,
                                                                                       tracker_input)
    if not (replay_dir / "frames.jsonl").is_file():
        raise SystemExit(f"tracks of tracker input {tracker_input} not found: {replay_dir} "
                         f"(build them with python -m football_fatigue.video.tracking --profile {tracker_input} …)")
    roots = import_roots()
    config = PROFILES[args.profile]
    source, video_path = load_source(data_root, args.source)
    if args.end is None:
        args.end = _replay_last_frame(replay_dir)
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    play_area = None
    if spec.get("keypoints") and (data_root / spec["keypoints"]).is_file():
        play_area = play_area_polygon(data_root / spec["keypoints"])
    first = max(0, args.start - args.lookback)
    last = min(args.end + config.gap_limit + config.close_after_quiet, _replay_last_frame(replay_dir))
    inputs = load_inputs(vision_dir / args.cache_run, replay_dir, source, first, last, tracker_input)
    last = inputs.frames[-1].frame

    # B0 keeps the historical run ids; other tracker inputs are named in the id so no B0 run can be overwritten
    ti_tag = "" if tracker_input == "B0" else f"{tracker_input}-"
    run_id = f"occl-{source.source_id}-{args.start}-{args.end}-{args.profile}-{ti_tag}{config.sha256()[:8]}"
    out_dir = out_root / source.source_id / "occlusion" / ENGINE_VERSION / run_id
    if out_dir.exists():
        raise SystemExit(f"output already exists (write-once): {out_dir}")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    git = git_state()
    provenance = {"engine": "football_fatigue.video.occlusion", "engine_version": ENGINE_VERSION,
                  "profile": args.profile, "config_sha256": config.sha256(), **inputs.provenance,
                  "git_commit": git["commit"], "git_dirty": git["dirty"]}
    t0 = time.perf_counter()
    result = OcclusionEngine(config).run(inputs.frames, source, run_id, (args.start, args.end), provenance, play_area=play_area)
    engine_s = time.perf_counter() - t0

    with (tmp / "events.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for e in result.events:
            fh.write(e.to_json() + "\n")
    with (tmp / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in result.frame_rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    with (tmp / "suppressed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for e in result.suppressed:
            fh.write(e.to_json() + "\n")
    with (tmp / "suppressed_frames.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in result.suppressed_rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    outputs = {"events": "events.jsonl", "frames": "frames.jsonl", "suppressed": "suppressed.jsonl",
               "suppressed_frames": "suppressed_frames.jsonl"}
    render_s = None
    if args.render:
        from football_fatigue.video import render
        from football_fatigue.video.sources import ExactFrameReader, verify_reader

        t1 = time.perf_counter()
        # decoded-index check on window frames (sequential decode reference) before anything is drawn
        sync = verify_reader(video_path, [args.start, (args.start + args.end) // 2, args.end])
        if not sync["exact"]:
            raise SystemExit(f"frame indexing check failed: {sync}")
        outputs["frame_index_check"] = sync
        reader = ExactFrameReader(video_path, cache_frames=64)
        try:
            from football_fatigue.video import render_focus

            if args.render_style == "focus":
                outputs["video"] = render_focus.render_focus_video(tmp / "annotated.mp4", reader.frame, inputs.frames,
                                                                   result.events, result.frame_rows, (args.start, args.end),
                                                                   source.fps, source.pts)
            else:
                outputs["video"] = render.render_video(tmp / "annotated.mp4", reader.frame, inputs.frames, result.events,
                                                       result.frame_rows, (args.start, args.end), source.fps, source.pts)
            labels = render.event_labels(result.events)
            by_id = {e.event_id: e for e in result.events}
            (tmp / "strips").mkdir()
            strips = []
            for e in render.select_strip_events(result.events, (args.start, args.end), args.strips):
                name = f"{labels[e.event_id]}_{e.event_class}_{e.subtype}_f{e.start_frame}.png"
                if args.render_style == "focus":
                    strips.append(render_focus.render_event_strip(tmp / "strips" / name, e, labels[e.event_id], reader.frame,
                                                                  inputs.frames, result.frame_rows, result.events,
                                                                  (args.start, args.end)))
                else:
                    strips.append(render.render_strip(tmp / "strips" / name, e, labels[e.event_id], reader.frame,
                                                      inputs.frames, result.frame_rows, by_id, labels))
            outputs["strips"] = strips
            outputs["labels"] = labels
        finally:
            reader.close()
        render_s = time.perf_counter() - t1

    n_frames = len(inputs.frames)
    manifest = {
        "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "status": "complete", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "source": source.__dict__, "video_path_relative": str(video_path.relative_to(data_root)),
        "window": {"start": args.start, "end": args.end, "processing": [inputs.frames[0].frame, last],
                   "lookback": args.lookback, "lookahead": config.gap_limit + config.close_after_quiet},
        "profile": args.profile, "config": config.__dict__, "play_area": play_area,
        "cache_run": args.cache_run, "replay_run": args.replay_run if tracker_input == "B0" else None,
        "tracker_input": tracker_input,
        "tracks_dir": None if tracker_input == "B0" else str(replay_dir.relative_to(out_root)).replace("\\", "/"),
        "provenance": provenance, "stats": result.stats,
        "timing": {"engine_s": round(engine_s, 3), "engine_ms_per_frame": round(1000 * engine_s / n_frames, 3),
                   "render_s": round(render_s, 1) if render_s else None},
        "import_roots": roots,
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform()},
        "outputs": outputs,
        "notes": ["pitch coordinates are not computed (pitch.status = not_computed)",
                  "play_area = image polygon from SoccerTrack's annotated pitch landmarks (not from GT player boxes)",
                  "GT boxes are not used by the engine; no accuracy is claimed by this run"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.rename(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "stats": result.stats, "timing": manifest["timing"]}, indent=1))
    return 0


def _replay_last_frame(replay_dir: Path) -> int:
    last = None
    with (replay_dir / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = int(json.loads(line)["frame_num"])
    if last is None:
        raise SystemExit(f"empty replay: {replay_dir}")
    return last


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Write <run-dir>/../../evaluation/<run_id>.<eval version>.json (the run folder itself stays write-once)."""
    from football_fatigue.video.occlusion import evaluate as E
    from football_fatigue.vision.mot_eval import load_reference

    data_root = Path(args.data_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source_id = manifest["source"]["source_id"]
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["gt"])
    import_roots()
    source, _video = load_source(data_root, source_id)
    lo, hi = manifest["window"]["start"], manifest["window"]["end"]
    p0, p1 = manifest["window"]["processing"]
    vision = data_root / "processed" / "vision"
    cache_run = manifest.get("cache_run") or spec["cache_run"]
    tracker_input = check_name(manifest.get("tracker_input", "B0"))
    if tracker_input == "B0":
        replay_dir = vision / (manifest.get("replay_run") or spec["replay_run"])
    else:
        replay_dir = run_dir.parents[3] / manifest["tracks_dir"]       # <out_root>/<tracks_dir>
    inputs = load_inputs(vision / cache_run, replay_dir, source, p0, p1, tracker_input)
    tracks = {f.frame: {tid: o.box for tid, o in f.tracks.items()} for f in inputs.frames}
    gt, gt_stats = load_reference(data_root / spec["gt"], 0)   # SoccerTrack MOT frames are 0-based decoded indices
    result = E.evaluate_run(E.load_jsonl(run_dir / "events.jsonl"), E.load_jsonl(run_dir / "suppressed.jsonl"),
                            E.load_jsonl(run_dir / "frames.jsonl") + E.load_jsonl(run_dir / "suppressed_frames.jsonl"),
                            tracks, gt, (lo, hi))
    result.update(run_id=manifest["run_id"], source_id=source_id, split_role=spec["role"],
                  profile=manifest.get("profile", "step1"), tracker_input=tracker_input, gt_file=spec["gt"],
                  gt_stats=gt_stats,
                  note="development data; check matches are development-internal checks, not independent validation")
    out = run_dir.parent.parent / "evaluation" / f"{manifest['run_id']}.{E.EVAL_VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": result["summary"]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m football_fatigue.video.occlusion")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="detect occlusion events in a frame window and write versioned outputs")
    run.add_argument("--data-root", required=True, help="read-only data root holding external/ and processed/vision/")
    run.add_argument("--out-root", default="data/processed/video", help="Session B output root (must be outside --data-root)")
    run.add_argument("--source", required=True)
    run.add_argument("--cache-run", help="run folder under <data-root>/processed/vision holding detection_cache/ (default: the source's)")
    run.add_argument("--replay-run", help="tracker replay run folder under <data-root>/processed/vision (default: the source's)")
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--end", type=int, default=None, help="last frame (default: last replay frame)")
    run.add_argument("--profile", choices=sorted(PROFILES), default="v2", help="engine settings (step1 = Step 1 baseline)")
    run.add_argument("--tracker-input", choices=sorted(TRACKER_INPUTS), default=DEFAULT_TRACKER_INPUT,
                     help="tracks to analyse: B0 = production replay (default, historical); T0 = turf filter off "
                          "(recommended; build first with python -m football_fatigue.video.tracking)")
    run.add_argument("--render-style", choices=["focus", "full"], default="focus")
    run.add_argument("--lookback", type=int, default=50, help="frames processed before --start for track history")
    run.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--strips", type=int, default=10)
    ev = sub.add_parser("evaluate", help="score a saved run against SoccerTrack MOT ground truth (development matches)")
    ev.add_argument("--data-root", required=True)
    ev.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return cmd_evaluate(args)
        return cmd_run(args)
    except SourceError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
