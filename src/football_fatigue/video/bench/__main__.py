"""``python -m football_fatigue.video.bench <command>`` - detector and tracker benchmark (development data only).

  manifest   build the stratified frame manifest (write-once)
  detect     run one detector/protocol on the manifest frames (CPU by default; resumable, write-once when complete)
  evaluate   score every completed detector run on the same frames
  track      tracker comparison on identical cached detections (CPU)
  registry   print the detector candidates with their current status
  gallery    comparison PNGs and short clips from executed runs
  report     write the markdown summary tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("data/processed/video")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, project=False):
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
        if project:
            p.add_argument("--project-root", type=Path, default=None,
                           help="Session A project root holding models/ (default: parent of --data-root)")

    common(sub.add_parser("manifest"))
    d = sub.add_parser("detect")
    common(d, True)
    d.add_argument("--detector", required=True)
    d.add_argument("--protocol", required=True)
    d.add_argument("--device", default="cpu")
    d.add_argument("--max-frames", type=int)
    d.add_argument("--source", action="append")
    e = sub.add_parser("evaluate")
    common(e)
    t = sub.add_parser("track")
    common(t)
    t.add_argument("--source", action="append", required=True)
    t.add_argument("--tracker", action="append", required=True)
    t.add_argument("--input", action="append", default=None, help="tracker input profile(s): T0, T0L (default T0)")
    t.add_argument("--buffer", type=int, default=75)
    r = sub.add_parser("registry")
    common(r, True)
    g = sub.add_parser("gallery")
    common(g)
    g.add_argument("--alt", default=None, help="alternative detector run (default: best other model by check AP50)")
    rp = sub.add_parser("report")
    common(rp, True)
    a = ap.parse_args(argv)
    project = getattr(a, "project_root", None) or Path(a.data_root).resolve().parent

    if a.cmd == "manifest":
        from football_fatigue.video.bench.frames import build

        print(build(a.data_root, a.out_root))
    elif a.cmd == "detect":
        from football_fatigue.video.bench.detectors import run

        print(run(a.data_root, a.out_root, project, a.detector, a.protocol, a.device, a.max_frames, a.source))
    elif a.cmd == "evaluate":
        from football_fatigue.video.bench.evaluate import evaluate_all

        print(evaluate_all(a.data_root, a.out_root))
    elif a.cmd == "track":
        from football_fatigue.video.bench.trackers import run_matrix

        for line in run_matrix(a.data_root, a.out_root, a.source, a.tracker, a.input or ["T0"], a.buffer):
            print(json.dumps(line), flush=True)
    elif a.cmd == "registry":
        from football_fatigue.video.bench.detectors import load_runs
        from football_fatigue.video.bench.registry import as_rows

        for row in as_rows(project, load_runs(a.out_root)):
            print(f"{row['status']:<36} {row['key']:<28} {row['family']:<8} {row['track']}  {row['licence'][:60]}")
    elif a.cmd == "gallery":
        from football_fatigue.video.bench.gallery import build_gallery

        print(build_gallery(a.data_root, a.out_root, a.alt))
    elif a.cmd == "report":
        from football_fatigue.video.bench.report import write_report

        print(write_report(a.data_root, a.out_root, project))
    return 0


if __name__ == "__main__":
    sys.exit(main())
