"""CLI: ``python -m football_fatigue.video.detection {diagnose,report,examples,ab,capture}``.

``diagnose``, ``report``, ``examples`` and ``ab`` are CPU-only and read existing artefacts. ``capture`` runs the detector
on the GPU and refuses to start without ``--approved`` (explicit user approval per run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_fatigue.video.sources import SOURCES

DEFAULT_OUT = Path(__file__).resolve().parents[4] / "data" / "processed" / "video"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="stage attribution from the cache + replay + GT (CPU)")
    d.add_argument("--data-root", type=Path, required=True)
    d.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    d.add_argument("--source", choices=sorted(SOURCES), action="append", required=True)
    d.add_argument("--max-frames", type=int)
    d.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("replay", help="tracker-input filter A/B: ByteTrack replay on identical cached detections (CPU)")
    t.add_argument("--data-root", type=Path, required=True)
    t.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    t.add_argument("--source", choices=sorted(SOURCES), action="append", required=True)
    t.add_argument("--variant", action="append", help="default: all variants")
    t.add_argument("--max-frames", type=int)
    t.add_argument("--overwrite", action="store_true")

    r = sub.add_parser("report", help="markdown tables over all diagnosed sources")
    r.add_argument("--out-root", type=Path, default=DEFAULT_OUT)

    e = sub.add_parser("examples", help="render stage-comparison panels for selected misses (CPU)")
    e.add_argument("--data-root", type=Path, required=True)
    e.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    e.add_argument("--source", choices=sorted(SOURCES), action="append", required=True)
    e.add_argument("--per-stage", type=int, default=3)

    a = sub.add_parser("ab", help="merge-policy A/B on a raw per-tile capture (CPU)")
    a.add_argument("--data-root", type=Path, required=True)
    a.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    a.add_argument("--capture", type=Path, required=True)

    c = sub.add_parser("capture", help="raw per-tile detector capture (GPU; needs --approved)")
    c.add_argument("--data-root", type=Path, required=True)
    c.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    c.add_argument("--source", choices=sorted(SOURCES), required=True)
    c.add_argument("--start", type=int, required=True)
    c.add_argument("--frames", type=int, required=True)
    c.add_argument("--approved", action="store_true", help="the user approved this GPU run")

    pl = sub.add_parser("plan2", help="A2-cap v2 window plan from the saved T0 decomposition (CPU)")
    pl.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    c2 = sub.add_parser("capture2", help="A2-cap v2 stage-resolved raw-tile capture (GPU needs --approved)")
    c2.add_argument("--data-root", type=Path, required=True)
    c2.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    c2.add_argument("--plan", type=Path, help="plan.json from plan2 (the approved capture)")
    c2.add_argument("--device", default="cuda:0")
    c2.add_argument("--smoke", type=int, default=0, help="readiness test: <= 2 frames to a separate smoke folder")
    c2.add_argument("--source", choices=sorted(SOURCES))
    c2.add_argument("--start", type=int)
    c2.add_argument("--approved", action="store_true", help="the user approved this GPU run")
    e2 = sub.add_parser("evaluate2", help="attribute missing GT boxes to capture-v2 stages (CPU)")
    e2.add_argument("--data-root", type=Path, required=True)
    e2.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    e2.add_argument("--capture-root", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "diagnose":
        from football_fatigue.video.detection.diagnose import diagnose

        for s in args.source:
            out = diagnose(args.data_root, args.out_root, s, max_frames=args.max_frames, overwrite=args.overwrite)
            print(json.dumps({"source": s, "out": str(out),
                              "runtime_s": json.loads((out / "summary.json").read_text())["runtime_s"]}))
    elif args.cmd == "replay":
        from football_fatigue.video.detection.replay import VARIANTS, run_variants

        names = args.variant or list(VARIANTS)
        for s in args.source:
            base = run_variants(args.data_root, args.out_root, s, names, args.max_frames, args.overwrite)
            for n in names:
                res = json.loads((base / n / "result.json").read_text())
                m = res["metrics_all_tracks"]
                print(json.dumps({"source": s, "variant": n, "HOTA": m["hota"]["HOTA"], "DetRe": m["hota"]["DetRe"],
                                  "DetPr": m["hota"]["DetPr"], "IDF1": m["identity"]["IDF1"],
                                  "IDSW": m["clear"]["IDSW"], "baseline": res["baseline_check"]}))
    elif args.cmd == "report":
        from football_fatigue.video.detection.report import markdown

        print(markdown(args.out_root))
    elif args.cmd == "examples":
        from football_fatigue.video.detection.visual import render_examples

        for s in args.source:
            print(json.dumps({"source": s, "written": [str(p) for p in render_examples(
                args.data_root, args.out_root, s, per_stage=args.per_stage)]}))
    elif args.cmd == "ab":
        from football_fatigue.video.detection.merge import run_ab

        print(json.dumps(run_ab(args.data_root, args.out_root, args.capture), indent=1))
    elif args.cmd == "capture":
        if not args.approved:
            ap.error("capture runs the detector on the GPU; it needs explicit user approval (--approved)")
        from football_fatigue.video.detection.capture import capture

        print(capture(args.data_root, args.out_root, args.source, args.start, args.frames))
    elif args.cmd == "plan2":
        from football_fatigue.video.detection.capture_v2 import make_plan

        print(make_plan(args.out_root))
    elif args.cmd == "capture2":
        if args.device != "cpu" and not args.approved:
            ap.error("capture2 on the GPU needs explicit user approval (--approved); only --device cpu --smoke runs without it")
        if not args.smoke and not args.approved:
            ap.error("the capture itself needs explicit user approval (--approved)")
        from football_fatigue.video.detection.capture_v2 import capture as capture2

        print(capture2(args.data_root, args.out_root, args.plan, args.device, args.smoke, args.source, args.start))
    elif args.cmd == "evaluate2":
        from football_fatigue.video.detection.capture_v2 import evaluate

        print(evaluate(args.data_root, args.out_root, args.capture_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
