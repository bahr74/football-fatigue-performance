"""M1 step 0: score our ball smoother against PFF's ballsSmoothed on the dev matches (evaluation only).

The smoother (``m1_possession.smoother``) sees only the raw ball. This script additionally reads, for scoring only:
PFF ``ballsSmoothed`` (reference), the tracking ``game_event_id`` overlay mapped to the event file's game event type
(frames inside OTB spells vs outside), and the eventTime of PA/CR/SH/CL possession events aligned to frames with
``synchronization.align_events`` (kick moments).

    python scripts/m1/evaluate_smoother.py --workers 4      # builds the position cache once (about 2 min per match)

Writes data/processed/m1_possession/smoother_eval/results.json and plots in reports/m1_possession/smoother/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from football_fatigue.event_detection.config import DEV_MATCHES
from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.synchronization import MatchStatus, align_events
from m1_possession import FLAG_MISSING, SmootherConfig, smooth_ball

ROOT = "data/raw/FIFA World Cup 2022"
CACHE = Path("data/processed/m1_possession/cache")
OUT = Path("data/processed/m1_possession/smoother_eval")
PLOTS = Path("reports/m1_possession/smoother")
KICK_TYPES = ("PA", "CR", "SH", "CL")
VIS = {None: 0, "VISIBLE": 1, "ESTIMATED": 2}
KICK_WIN = 3          # frames each side for kick preservation
DV_K = 3              # frames for the velocity-change measure
DV_THR = 3.0          # m/s: minimum change of velocity counted as a peak

SETTINGS = {
    "linear (no smoothing)": SmootherConfig(method="linear"),
    "savgol w7 p2": SmootherConfig(method="savgol", savgol_window=7, savgol_poly=2),
    "savgol w11 p2": SmootherConfig(method="savgol", savgol_window=11, savgol_poly=2),
    "savgol w15 p3": SmootherConfig(method="savgol", savgol_window=15, savgol_poly=3),
    "rts q30": SmootherConfig(method="rts", rts_q=30.0),
    "rts q300": SmootherConfig(method="rts", rts_q=300.0),
    "rts_adaptive q30 kick4": SmootherConfig(method="rts_adaptive", kick_dv_mps=4.0),
    "rts_adaptive q10 kick3": SmootherConfig(method="rts_adaptive", rts_q=10.0),
    "rts_adaptive q30 kick3 (default)": SmootherConfig(),
    "rts_adaptive q30 kick3 gap0.2": SmootherConfig(max_gap_s=0.2),
    "rts_adaptive q30 kick3 gap1.0": SmootherConfig(max_gap_s=1.0),
}


# ---------------------------------------------------------------- cache (one tracking pass per match)

def build_cache(mid: str) -> Path:
    path = CACHE / f"{mid}.npz"
    if path.exists():
        return path
    ds = Dataset(ROOT)
    meta = ds.metadata(mid)
    events = ds.load_events(mid, EventVersion.ROOT)
    otb_ids = {r.game_event_id for r in events if r.game_event_type == "OTB" and r.game_event_id is not None}
    kicks = [r for r in events if r.possession_event_type in KICK_TYPES]
    cols = {k: [] for k in ("t", "per", "fn", "rx", "ry", "rz", "rvis", "rcount", "sx", "sy", "sz", "otb")}

    def wrap(frames):
        for fr in frames:
            b = fr.ball
            cols["t"].append(fr.video_time_s); cols["per"].append(fr.canonical_period or 0); cols["fn"].append(fr.frame_num)
            raw_ok = b is not None and b.x_raw_m is not None and b.y_raw_m is not None
            sm_ok = b is not None and b.x_smooth_m is not None and b.y_smooth_m is not None
            nan = np.nan
            cols["rx"].append(b.x_raw_m if raw_ok else nan); cols["ry"].append(b.y_raw_m if raw_ok else nan)
            cols["rz"].append(b.z_raw_m if raw_ok and b.z_raw_m is not None else nan)
            cols["rvis"].append(VIS.get(b.visibility_raw, 3) if raw_ok else 0)
            cols["rcount"].append(b.raw_count if b is not None else 0)
            cols["sx"].append(b.x_smooth_m if sm_ok else nan); cols["sy"].append(b.y_smooth_m if sm_ok else nan)
            cols["sz"].append(b.z_smooth_m if sm_ok and b.z_smooth_m is not None else nan)
            cols["otb"].append(fr.game_event_id in otb_ids)
            yield fr

    sync = align_events(kicks, wrap(ds.iter_tracking_frames(mid, metadata=meta)), match_id=int(mid),
                        event_version=EventVersion.ROOT, fps=meta.fps)
    if not sync.scan_complete:
        raise RuntimeError(f"{mid}: tracking scan incomplete: {sync.fatal_error}")
    fidx = {f: i for i, f in enumerate(cols["fn"])}
    by_id = {r.event_row_id: r for r in kicks}
    kf, kt = [], []
    for a in sync.alignments:
        if a.status is MatchStatus.MATCHED:
            kf.append(fidx[a.frame_num]); kt.append(KICK_TYPES.index(by_id[a.event_row_id].possession_event_type))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in cols.items()}, kick_idx=np.asarray(kf, int),
                        kick_type=np.asarray(kt, int), fps=meta.fps, kicks_total=len(kicks))
    return path


# ---------------------------------------------------------------- metrics

def speed_from_positions(t, x, y) -> np.ndarray:
    """Horizontal speed by central differences over one frame each side (NaN where a neighbour is missing)."""
    s = np.full(len(t), np.nan)
    s[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) / (t[2:] - t[:-2])
    return s


def dv_series(t, x, y, per) -> np.ndarray:
    """|mean velocity over the next K frames - mean velocity over the previous K frames| (m/s), same period only."""
    n, k = len(t), DV_K
    out = np.full(n, np.nan)
    i = np.arange(k, n - k)
    ok = (per[i - k] == per[i + k])
    va = np.stack([(x[i + k] - x[i]), (y[i + k] - y[i])]) / (t[i + k] - t[i])
    vb = np.stack([(x[i] - x[i - k]), (y[i] - y[i - k])]) / (t[i] - t[i - k])
    d = np.hypot(*(va - vb))
    out[i] = np.where(ok, d, np.nan)
    return out


def peaks(dv: np.ndarray) -> np.ndarray:
    """Frames where dv is a local maximum over +-KICK_WIN frames and at least DV_THR."""
    v = np.where(np.isfinite(dv), dv, -1.0)
    w = np.lib.stride_tricks.sliding_window_view(np.pad(v, KICK_WIN, constant_values=-1.0), 2 * KICK_WIN + 1).max(axis=1)
    return (v >= DV_THR) & (v == w)


def jitter_frames(c, ps) -> np.ndarray:
    """Ball rolling smoothly (selected with the reference only): outside OTB spells, PFF speed 3-20 m/s, PFF z < 0.5 m,
    no labelled kick within 15 frames."""
    n = len(c["t"])
    near = np.zeros(n, bool)
    for k in c["kick_idx"]:
        near[max(0, k - 15):k + 16] = True
    return (~c["otb"]) & (ps >= 3) & (ps <= 20) & (np.nan_to_num(c["sz"], nan=9) < 0.5) & ~near


def central_v(t, x, y) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = np.full(len(t), np.nan), np.full(len(t), np.nan)
    dt = t[2:] - t[:-2]
    vx[1:-1], vy[1:-1] = (x[2:] - x[:-2]) / dt, (y[2:] - y[:-2]) / dt
    return vx, vy


def accel(t, vx, vy) -> np.ndarray:
    """Frame-to-frame acceleration |v(t+1) - v(t)| / dt (m/s^2) from a velocity series."""
    a = np.full(len(t), np.nan)
    a[:-1] = np.hypot(np.diff(vx), np.diff(vy)) / np.diff(t)
    return a


def kick_hits(dvp: np.ndarray, kick_idx: np.ndarray) -> np.ndarray:
    n = len(dvp)
    return np.array([dvp[max(0, k - KICK_WIN):min(n, k + KICK_WIN + 1)].any() for k in kick_idx], bool)


def eval_match(mid: str) -> dict:
    t0 = time.time()
    c = dict(np.load(build_cache(mid)))
    t, per = c["t"], c["per"]
    pff_ok = np.isfinite(c["sx"])
    ps = speed_from_positions(t, c["sx"], c["sy"])
    jit = jitter_frames(c, ps)
    out = dict(match=mid, frames=len(t), seconds_cache=round(time.time() - t0, 1), kicks_total=int(c["kicks_total"]),
               kicks_aligned=len(c["kick_idx"]), multi_ball_frames=int((c["rcount"] > 1).sum()),
               raw_visibility={k: int((c["rvis"] == v).sum()) for k, v in VIS.items() if k} | {"other": int((c["rvis"] == 3).sum())},
               minutes=float(np.sum(np.diff(t)[np.diff(per) == 0]) / 60))
    arrays = {}
    # reference rows: raw and PFF themselves
    for name, (x, y) in (("raw", (c["rx"], c["ry"])), ("pff_smoothed", (c["sx"], c["sy"]))):
        dvp = peaks(dv_series(t, x, y, per))
        arrays[name] = dict(cov=float(np.isfinite(x).mean()), kick=kick_hits(dvp, c["kick_idx"]), peaks=int(dvp.sum()),
                            # raw is held for two frames, so its frame-to-frame acceleration is not comparable
                            jit=(accel(t, *central_v(t, x, y))[jit] if name != "raw" else np.array([])).astype(np.float32))
        if name == "raw":
            both = np.isfinite(x) & pff_ok
            d = np.hypot(x - c["sx"], y - c["sy"])
            sd = np.abs(speed_from_positions(t, x, y) - ps)
            arrays[name].update(d_out=d[both & ~c["otb"]].astype(np.float32), d_in=d[both & c["otb"]].astype(np.float32),
                                s_out=sd[both & ~c["otb"]].astype(np.float32), s_in=sd[both & c["otb"]].astype(np.float32))
    for name, cfg in SETTINGS.items():
        t1 = time.time()
        r = smooth_ball(t, c["rx"], c["ry"], c["rz"], period=per, cfg=cfg)
        ok = r.flag != FLAG_MISSING
        both = ok & pff_ok
        d = np.hypot(r.x - c["sx"], r.y - c["sy"])
        sd = np.abs(np.hypot(r.vx, r.vy) - ps)
        dvp = peaks(dv_series(t, r.x, r.y, per))
        arrays[name] = dict(cov=float(ok.mean()), cov_of_pff=float(ok[pff_ok].mean()), counts=r.counts,
                            d_out=d[both & ~c["otb"]].astype(np.float32), d_in=d[both & c["otb"]].astype(np.float32),
                            s_out=sd[both & ~c["otb"]].astype(np.float32), s_in=sd[both & c["otb"]].astype(np.float32),
                            kick=kick_hits(dvp, c["kick_idx"]), peaks=int(dvp.sum()), jit=accel(t, r.vx, r.vy)[jit].astype(np.float32),
                            seconds=round(time.time() - t1, 1))
    out["arrays"] = arrays
    return out


def _q(a, p):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else None


def summarise(res: list[dict]) -> dict:
    names = list(res[0]["arrays"])
    minutes = sum(r["minutes"] for r in res)
    frames = sum(r["frames"] for r in res)
    table = {}
    for nm in names:
        rows = [r["arrays"][nm] for r in res]
        cat = lambda k: np.concatenate([x[k] for x in rows]) if k in rows[0] else np.array([])  # noqa: E731
        kick = cat("kick")
        table[nm] = dict(
            coverage_pct=100 * sum(x["cov"] * r["frames"] for x, r in zip(rows, res)) / frames,
            coverage_of_pff_pct=(100 * float(np.mean([x["cov_of_pff"] for x in rows])) if "cov_of_pff" in rows[0] else None),
            dist_out_med=_q(cat("d_out"), 50), dist_out_p95=_q(cat("d_out"), 95),
            dist_in_med=_q(cat("d_in"), 50), dist_in_p95=_q(cat("d_in"), 95),
            speed_out_med=_q(cat("s_out"), 50), speed_out_p95=_q(cat("s_out"), 95),
            speed_in_med=_q(cat("s_in"), 50), speed_in_p95=_q(cat("s_in"), 95),
            kick_pct=100 * float(kick.mean()) if len(kick) else None, peaks_per_min=sum(x["peaks"] for x in rows) / minutes,
            jitter_med=_q(cat("jit"), 50), jitter_frames=int(len(cat("jit"))),
            seconds=sum(x.get("seconds", 0) for x in rows))
    counts = {nm: {k: int(sum(r["arrays"][nm]["counts"].get(k, 0) for r in res)) for k in res[0]["arrays"][nm]["counts"]}
              for nm in names if "counts" in res[0]["arrays"][nm]}
    return dict(matches=[r["match"] for r in res], frames=frames, minutes=minutes, kicks_total=sum(r["kicks_total"] for r in res),
                kicks_aligned=sum(r["kicks_aligned"] for r in res), multi_ball_frames=sum(r["multi_ball_frames"] for r in res),
                raw_visibility={k: sum(r["raw_visibility"][k] for r in res) for k in res[0]["raw_visibility"]},
                table=table, counts=counts)


def print_table(s: dict) -> None:
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    print(f"matches {s['matches']}, frames {s['frames']}, minutes {s['minutes']:.0f}, kicks aligned {s['kicks_aligned']}/{s['kicks_total']}, "
          f"multi-ball frames {s['multi_ball_frames']}, raw visibility {s['raw_visibility']}")
    print("| method | coverage % | dist out med/p95 m | dist in med/p95 m | speed out med/p95 m/s | speed in med/p95 m/s | kick % | peaks/min | jitter m/s² |")
    print("|---|---|---|---|---|---|---|---|---|")
    for nm, r in s["table"].items():
        print(f"| {nm} | {f(r['coverage_pct'], 1)} | {f(r['dist_out_med'])} / {f(r['dist_out_p95'])} | {f(r['dist_in_med'])} / {f(r['dist_in_p95'])} | "
              f"{f(r['speed_out_med'])} / {f(r['speed_out_p95'])} | {f(r['speed_in_med'])} / {f(r['speed_in_p95'])} | {f(r['kick_pct'], 1)} | "
              f"{f(r['peaks_per_min'], 1)} | {f(r['jitter_med'], 1)} |")
    print("counts per setting:", json.dumps(s["counts"], indent=None))


# ---------------------------------------------------------------- plots

def plots(mid: str, recommended: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = dict(np.load(CACHE / f"{mid}.npz"))
    t, per = c["t"], c["per"]
    fps = float(c["fps"])
    win = int(12 * fps)
    kicks = np.sort(c["kick_idx"])
    start = None
    for i0 in range(0, len(t) - win, int(fps)):           # first 12 s window in play with >= 5 kicks and good coverage
        sl = slice(i0, i0 + win)
        if per[i0] == per[i0 + win - 1] and ((kicks >= i0) & (kicks < i0 + win)).sum() >= 5 \
                and np.isfinite(c["rx"][sl]).mean() > 0.9 and np.isfinite(c["sx"][sl]).mean() > 0.9:
            start = i0
            break
    sl = slice(start, start + win)
    rec = smooth_ball(t, c["rx"], c["ry"], c["rz"], period=per, cfg=SETTINGS[recommended])
    sg = smooth_ball(t, c["rx"], c["ry"], c["rz"], period=per, cfg=SETTINGS["savgol w11 p2"])
    rts = smooth_ball(t, c["rx"], c["ry"], c["rz"], period=per, cfg=SETTINGS["rts q30"])
    kk = kicks[(kicks >= start) & (kicks < start + win)]
    PLOTS.mkdir(parents=True, exist_ok=True)
    tt = t[sl] - t[start]
    files = []

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(c["rx"][sl], c["ry"][sl], ".", ms=4, color="0.6", label="raw (balls)")
    ax.plot(c["sx"][sl], c["sy"][sl], "-", lw=2, color="tab:orange", alpha=.8, label="PFF ballsSmoothed (reference)")
    ax.plot(rec.x[sl], rec.y[sl], "-", lw=1.5, color="tab:blue", label=f"ours: {recommended}")
    ax.plot(rec.x[kk], rec.y[kk], "kx", ms=9, label="PFF kick events (PA/CR/SH/CL)")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=8)
    ax.set_title(f"Match {mid}: ball path, 12 s from video time {t[start]:.1f} s")
    files.append(str(PLOTS / f"path_{mid}.png")); fig.savefig(files[-1], dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tt, speed_from_positions(t, c["rx"], c["ry"])[sl], color="0.7", lw=1, label="raw (finite differences)")
    ax.plot(tt, speed_from_positions(t, c["sx"], c["sy"])[sl], color="tab:orange", lw=2, label="PFF ballsSmoothed")
    ax.plot(tt, np.hypot(rec.vx, rec.vy)[sl], color="tab:blue", lw=1.5, label=f"ours: {recommended}")
    for k in kk:
        ax.axvline(t[k] - t[start], color="k", ls=":", lw=1)
    ax.set_ylim(0, 40); ax.set_xlabel("seconds"); ax.set_ylabel("horizontal speed (m/s)"); ax.legend(fontsize=8)
    ax.set_title(f"Match {mid}: ball speed over the same passage (dotted = PFF kick events)")
    files.append(str(PLOTS / f"speed_{mid}.png")); fig.savefig(files[-1], dpi=130, bbox_inches="tight"); plt.close(fig)

    k = int(kk[len(kk) // 2])
    s2 = slice(k - int(fps), k + int(fps))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[s2] - t[k], speed_from_positions(t, c["sx"], c["sy"])[s2], color="tab:orange", lw=2, label="PFF ballsSmoothed")
    ax.plot(t[s2] - t[k], np.hypot(sg.vx, sg.vy)[s2], color="tab:green", label="savgol w11 p2")
    ax.plot(t[s2] - t[k], np.hypot(rts.vx, rts.vy)[s2], color="tab:purple", label="rts q30")
    ax.plot(t[s2] - t[k], np.hypot(rec.vx, rec.vy)[s2], color="tab:blue", label=recommended)
    ax.axvline(0, color="k", ls=":"); ax.set_xlabel("seconds from PFF kick eventTime"); ax.set_ylabel("speed (m/s)")
    ax.legend(fontsize=8); ax.set_title(f"Match {mid}: one kick, methods compared")
    files.append(str(PLOTS / f"kick_{mid}.png")); fig.savefig(files[-1], dpi=130, bbox_inches="tight"); plt.close(fig)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--recommended", default="rts_adaptive q30 kick3 (default)")
    a = ap.parse_args()
    mids = [str(m) for m in DEV_MATCHES]
    if a.workers > 1:
        with multiprocessing.Pool(a.workers) as pool:
            res = pool.map(eval_match, mids)
    else:
        res = [eval_match(m) for m in mids]
    s = summarise(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(dict(s, settings={k: v.as_dict() for k, v in SETTINGS.items()},
                                                      kick_window_frames=KICK_WIN, dv_frames=DV_K, dv_threshold=DV_THR), indent=1), encoding="utf-8")
    print_table(s)
    for f in plots(mids[0], a.recommended):
        print("plot:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
