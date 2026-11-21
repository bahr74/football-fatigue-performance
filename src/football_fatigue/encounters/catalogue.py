"""Two-player proximity encounter catalogue over RAW pitch coordinates (M11 step 1).

An encounter is a *candidate* for an occlusion or identity problem, never a confirmed occlusion. Detection follows the
FOOTBALLTrace occlusion rule (windows of ``window_s`` with ``window_overlap``; a pair qualifies in a window when its distance is
at most ``tau`` for at least ``min_close_fraction`` of the window), measured in time from real timestamps, not in a fixed
number of samples. Windows live inside *segments*: one canonical period without an unobserved stretch longer than
``max_gap_s``, so no window, and no encounter, crosses a period boundary or a frame gap. Overlapping qualifying windows of one
pair are merged; the encounter runs from the first to the last close frame of the merged windows.

Every encounter gets one exclusive category, in this precedence:
``suspicious_identity`` (a ``DISCONTINUITY_CANDIDATE`` jump of either player within the context: see ``quality``),
``unresolved`` (ambiguous raw values, unresolved or quarantined attribution, a substitution window, an over-full side list or
too little context), ``trusted_reference_candidate`` (none of the above, no implausible jump or over-speed step, no third
player within ``tau``, both players observed and VISIBLE in at least the trusted shares of the context) and ``ordinary``
(everything else, including encounters whose only anomaly is kinematic noise or a frame-wide shift).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import EventVersion, IngestionError, QualityFlag, approved_output_root, safe_output_path
from football_fatigue.match_data import CoordinatePolicy, MatchData

from . import quality as Qm
from .quality import M11Config, QualityResult, assess, sample_categories
from .trajectories import CONFIDENCE, STATE_ABSENT, STATE_AVAILABLE, VISIBILITY, RawCache, build_cache, load_cache

SCHEMA_VERSION = "m11-encounters-v2"     # v2: OVERSPEED_STEP, FRAME_WIDE_SHIFT, DISCONTINUITY_CANDIDATE; v1 outputs are kept unchanged
OUTPUT_DIR = "m11-encounters-v2"
EVENT_VERSION = EventVersion.ROOT          # substitution events; tracking supplies every position
CATEGORIES = ("suspicious_identity", "unresolved", "trusted_reference_candidate", "ordinary")
OUTPUT_SUFFIXES = (".cache.npz", ".quality.npz", ".encounters.jsonl", ".jumps.jsonl", ".summary.json", ".manifest.json")
_SUSPICIOUS = Qm.DISCONTINUITY_CANDIDATE
_UNRESOLVED = Qm.AMBIGUOUS_RAW | Qm.ATTRIBUTION_UNRESOLVED | Qm.SUBSTITUTION_WINDOW | Qm.SIDE_OVERCOUNT
_NOT_TRUSTED = Qm.IMPLAUSIBLE_JUMP | Qm.OVERSPEED_STEP
_ENCOUNTER_FLAGS = ("DISCONTINUITY_CANDIDATE", "IMPLAUSIBLE_JUMP", "LABEL_TRANSFER_CANDIDATE", "RECIPROCAL_EXCHANGE", "FRAME_WIDE_SHIFT",
                    "OVERSPEED_STEP", "AMBIGUOUS_RAW", "ATTRIBUTION_UNRESOLVED", "SUBSTITUTION_WINDOW", "SIDE_OVERCOUNT")


def segments(c: RawCache, cfg: M11Config) -> list[tuple[int, int]]:
    """Half-open row ranges of one canonical period without a time step longer than ``max_gap_s``."""
    F = c.n_frames
    if F == 0:
        return []
    brk = np.zeros(F, bool)
    brk[0] = True
    brk[1:] = (c.canonical_period[1:] != c.canonical_period[:-1]) | (np.diff(c.video_time_s) > cfg.max_gap_s)
    edges = np.r_[np.flatnonzero(brk), F]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if c.canonical_period[a] > 0]


def frame_weights(c: RawCache, segs: list[tuple[int, int]]) -> np.ndarray:
    """Seconds each frame stands for: the time to the next frame of its segment, capped at one nominal frame interval."""
    nominal = 1.0 / c.fps
    w = np.zeros(c.n_frames)
    for a, b in segs:
        dt = np.diff(c.video_time_s[a:b])
        w[a:b - 1] = np.minimum(dt, nominal)
        w[b - 1] = nominal
    return w


def _r(v: float | None, n: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _slot(c: RawCache, s: int) -> dict[str, Any]:
    return dict(slot=s, side=str(c.slot_side[s]), jersey=int(c.slot_jersey[s]),
                player_id=None if c.slot_player_id[s] < 0 else int(c.slot_player_id[s]),
                team_id=None if c.slot_team_id[s] < 0 else int(c.slot_team_id[s]))


def _windows(t: np.ndarray, C: np.ndarray, r0: int, r1: int, cfg: M11Config) -> list[tuple[int, int, int]]:
    """Merged qualifying windows of one segment: (row_lo, row_hi, n_windows), rows half-open."""
    W = cfg.window_s
    step = W * (1.0 - cfg.window_overlap)
    ts = t[r0] + step * np.arange(int((t[r1 - 1] - t[r0]) // step) + 1)
    lo = r0 + np.searchsorted(t[r0:r1], ts, "left")
    hi = r0 + np.searchsorted(t[r0:r1], ts + W, "left")
    q = (C[hi] - C[lo]) >= cfg.min_close_fraction * W - 1e-9
    out, k = [], 0
    idx = np.flatnonzero(q)
    while k < len(idx):
        j = k
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:      # consecutive windows overlap: merge
            j += 1
        out.append((int(lo[idx[k]]), int(hi[idx[j]]), j - k + 1))
        k = j + 1
    return out


def find_encounters(c: RawCache, qr: QualityResult, cfg: M11Config = M11Config()) -> list[dict[str, Any]]:
    segs = segments(c, cfg)
    w = frame_weights(c, segs)
    t = c.video_time_s
    avail = c.state == STATE_AVAILABLE
    jumps_by_slot: dict[int, list[dict]] = {}
    for j in qr.jumps:
        jumps_by_slot.setdefault(j["slot"], []).append(j)
    out = []
    S = c.n_slots
    for a in range(S):
        for b in range(a + 1, S):
            both = avail[:, a] & avail[:, b]
            if not both.any():
                continue
            d = np.where(both, np.hypot(c.x[:, a] - c.x[:, b], c.y[:, a] - c.y[:, b]), np.inf)
            for tau in cfg.taus_m:
                close = d <= tau
                if not close.any():
                    continue
                C = np.r_[0.0, np.cumsum(w * close)]
                for r0, r1 in segs:
                    if not close[r0:r1].any():
                        continue
                    for lo, hi, nw in _windows(t, C, r0, r1, cfg):
                        rows = lo + np.flatnonzero(close[lo:hi])
                        out.append(_record(c, qr, cfg, tau, a, b, int(rows[0]), int(rows[-1]), nw, d, close, r0, r1, w, jumps_by_slot))
    out.sort(key=lambda e: (e["tau_m"], e["rows"]["during"][0], e["pair"][0]["slot"], e["pair"][1]["slot"]))
    counters: dict[float, int] = {}
    for e in out:
        n = counters[e["tau_m"]] = counters.get(e["tau_m"], -1) + 1
        e["encounter_id"] = f"{c.match_id}-t{e['tau_m']:g}-{n:05d}"
    return out


def _record(c, qr, cfg, tau, a, b, start, end, nw, d, close, r0, r1, w, jumps_by_slot) -> dict[str, Any]:
    t = c.video_time_s
    t0, t1 = t[start], t[end]
    cx0 = max(r0, int(np.searchsorted(t, t0 - cfg.context_s, "left")))
    cx1 = min(r1, int(np.searchsorted(t, t1 + cfg.context_s, "right")))
    expected = (t1 - t0 + 2 * cfg.context_s) * c.fps + 1
    edge = t0 - cfg.context_s < t[r0] - 1e-9 or t1 + cfg.context_s > t[r1 - 1] + 1e-9
    avail = c.state[cx0:cx1] == STATE_AVAILABLE
    mask = qr.mask[cx0:cx1]
    players, bits = [], 0
    for s in (a, b):
        av = avail[:, s]
        vis = c.visibility[cx0:cx1, s]
        dur = slice(start - cx0, end - cx0 + 1)
        bef = slice(0, start - cx0)
        conf = c.confidence[start:end + 1, s][c.state[start:end + 1, s] != STATE_ABSENT]
        sp = qr.speed[start:end + 1, s]
        js = [j for j in jumps_by_slot.get(s, []) if cx0 <= j["row_to"] < cx1]
        sbits = int(np.bitwise_or.reduce(mask[:, s])) if len(mask) else 0
        bits |= sbits
        players.append(dict(
            **_slot(c, s),
            coverage_context=_r(min(1.0, av.sum() / expected)),
            visible_share_context=_r((av & (vis == VISIBILITY["VISIBLE"])).sum() / av.sum()) if av.any() else None,
            estimated_share_before=_r((av[bef] & (vis[bef] == VISIBILITY["ESTIMATED"])).sum() / av[bef].sum()) if av[bef].any() else None,
            estimated_share_during=_r((av[dur] & (vis[dur] == VISIBILITY["ESTIMATED"])).sum() / av[dur].sum()) if av[dur].any() else None,
            confidence_during={k: int((conf == v).sum()) for k, v in CONFIDENCE.items()},
            median_speed_during_m_s=_r(np.nanmedian(sp)) if np.isfinite(sp).any() else None,
            mask_flags=[k for k, v in Qm.MASK_BITS.items() if sbits & v],
            jump_ids=[j["jump_id"] for j in js]))
    # third players within tau of either pair member during the encounter
    third = []
    xa, ya, xb, yb = c.x[start:end + 1, a], c.y[start:end + 1, a], c.x[start:end + 1, b], c.y[start:end + 1, b]
    ww = w[start:end + 1]
    for o in range(c.n_slots):
        if o in (a, b):
            continue
        xo, yo = c.x[start:end + 1, o], c.y[start:end + 1, o]
        do = np.fmin(np.hypot(xo - xa, yo - ya), np.hypot(xo - xb, yo - yb))
        if np.isfinite(do).any() and np.nanmin(do) <= tau:
            third.append(dict(**_slot(c, o), min_distance_m=_r(np.nanmin(do)), close_time_s=_r(float((ww * (do <= tau)).sum()), 3)))
    coverage = [p["coverage_context"] or 0.0 for p in players]
    flags = [k for k in _ENCOUNTER_FLAGS if bits & Qm.MASK_BITS[k]]
    if min(coverage) < cfg.min_context_coverage:
        flags.append("INSUFFICIENT_CONTEXT")
    if edge:
        flags.append("SEGMENT_EDGE")
    if third:
        flags.append("THIRD_PLAYER_NEARBY")
    discontinuity = bool(bits & _SUSPICIOUS)
    if discontinuity:
        category = "suspicious_identity"
    elif bits & _UNRESOLVED or "INSUFFICIENT_CONTEXT" in flags:
        category = "unresolved"
    elif (not third and not bits & _NOT_TRUSTED and min(coverage) >= cfg.trusted_min_coverage
          and all((p["visible_share_context"] or 0.0) >= cfg.trusted_min_visible for p in players)):
        category = "trusted_reference_candidate"
    else:
        category = "ordinary"
    dd = d[start:end + 1]
    k = int(np.argmin(dd))
    seg_dur = slice(start, end + 1)
    return dict(
        encounter_id=None, match_id=c.match_id, period=int(c.canonical_period[start]), tau_m=tau, category=category,
        pair=[_slot(c, a), _slot(c, b)], same_team=bool(c.slot_side[a] == c.slot_side[b]),
        start_frame=int(c.frame_num[start]), end_frame=int(c.frame_num[end]), start_time_s=_r(t0, 4), end_time_s=_r(t1, 4),
        duration_s=_r(t1 - t0, 4), close_time_s=_r(float((w[seg_dur] * close[seg_dur]).sum()), 4), n_windows=nw,
        min_distance_m=_r(dd[k]), min_distance_frame=int(c.frame_num[start + k]),
        rows=dict(context=[cx0, cx1], before=[cx0, start], during=[start, end + 1], after=[end + 1, cx1]),
        context_frames=[int(c.frame_num[cx0]), int(c.frame_num[cx1 - 1])],
        players=players, third_players=third, flags=flags, possible_identity_discontinuity=discontinuity,
        discontinuity_evidence=sorted({j["jump_id"] for s in (a, b) for j in jumps_by_slot.get(s, [])
                                       if j["discontinuity_candidate"] and cx0 <= j["row_to"] < cx1}),
        provenance=dict(tracking_file=c.tracking_path, source_records=[int(c.record_first[cx0]), int(c.record_first[cx1 - 1] + c.record_count[cx1 - 1] - 1)],
                        cache=f"{EVENT_VERSION.value}.cache.npz"))


# ---------------------------------------------------------------- summary

def _q(v: np.ndarray, ps=(50, 95, 99)) -> dict[str, float | None]:
    v = v[np.isfinite(v)]
    return {f"p{p}": _r(np.percentile(v, p)) if len(v) else None for p in ps} | {"max": _r(v.max()) if len(v) else None, "n": int(len(v))}


def summarize(c: RawCache, qr: QualityResult, encounters: list[dict], cfg: M11Config) -> dict[str, Any]:
    segs = segments(c, cfg)
    tracked_s = float(sum(c.video_time_s[b - 1] - c.video_time_s[a] for a, b in segs))
    present = c.state != STATE_ABSENT
    avail = c.state == STATE_AVAILABLE
    st = qr.steps
    ok = st["valid"] & ~st["jump"] & ~st["overspeed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(ok, st["distance_m"] / st["dt_s"], np.nan)
        acc = np.abs(v[1:] - v[:-1]) / ((st["dt_s"][1:] + st["dt_s"][:-1]) / 2)
    nxt = (st["slot"][1:] == st["slot"][:-1]) & (st["row_from"][1:] == st["row_to"][:-1]) & ok[1:] & ok[:-1]
    cat = sample_categories(c, qr.mask)
    enc: dict[str, Any] = {}
    for tau in cfg.taus_m:
        e = [x for x in encounters if x["tau_m"] == tau]
        enc[f"{tau:g}m"] = dict(
            count=len(e), per_hour=_r(len(e) / tracked_s * 3600, 1) if tracked_s else None,
            by_category={k: sum(x["category"] == k for x in e) for k in CATEGORIES},
            same_team=sum(x["same_team"] for x in e), with_third_player=sum(bool(x["third_players"]) for x in e),
            possible_identity_discontinuity=sum(x["possible_identity_discontinuity"] for x in e),
            duration_s=_q(np.array([x["duration_s"] for x in e], float)), min_distance_m=_q(np.array([x["min_distance_m"] for x in e], float)))
    per_slot = []
    for s in range(c.n_slots):
        sm = st["slot"] == s
        per_slot.append(dict(**_slot(c, s), samples=int(present[:, s].sum()), available=int(avail[:, s].sum()),
                             estimated=int(((qr.mask[:, s] & Qm.ESTIMATED) != 0).sum()), updates=int(((qr.mask[:, s] & Qm.UPDATE_INSTANT) != 0).sum()),
                             jumps=int(st["jump"][sm].sum()), max_valid_speed_m_s=_r(np.nanmax(v[sm])) if np.isfinite(v[sm]).any() else None))
    js = qr.jumps
    return dict(
        schema=SCHEMA_VERSION, match_id=c.match_id, fps=c.fps, frames=c.n_frames, slots=c.n_slots, segments=len(segs),
        tracked_time_s=_r(tracked_s, 2), samples_without_jersey=c.samples_without_jersey, duplicate_slot_samples=c.duplicate_slot_samples,
        source_records=dict(frames_with_repeated_records=int((c.record_count > 1).sum()), frames_with_gap=int((c.frame_gap_before >= 0).sum())),
        sample_states=dict(present=int(present.sum()), available=int(avail.sum()), missing=int((c.state == 2).sum()), ambiguous=int((c.state == 3).sum())),
        mask_bits={k: int(((qr.mask & v) != 0).sum()) for k, v in Qm.MASK_BITS.items()},
        sample_categories={k: int((cat == i).sum()) for i, k in enumerate(Qm.SAMPLE_CATEGORIES)},
        cadence=dict(repeat_share_of_available=_r(((qr.mask & Qm.REPEAT_OF_PREVIOUS) != 0).sum() / avail.sum()) if avail.any() else None,
                     held_share_of_available=_r(((qr.mask & Qm.HELD_POSITION) != 0).sum() / avail.sum()) if avail.any() else None,
                     update_interval_s=_q(st["dt_s"][st["valid"]])),
        motion=dict(steps=int(len(ok)), valid_steps=int(st["valid"].sum()), jumps=int(st["jump"].sum()), overspeed_steps=int(st["overspeed"].sum()),
                    kinematic_steps=int(ok.sum()), valid_speed_m_s=_q(v[ok]), abs_accel_m_s2=_q(acc[nxt]),
                    note="raw finite differences between update instants; acceleration is dominated by provider noise and is a diagnostic only"),
        jumps=dict(count=len(js), landing_estimated=sum(j["visibility_to"] == VISIBILITY["ESTIMATED"] for j in js),
                   over_transfer_min=sum(j["distance_m"] > cfg.transfer_min_m for j in js),
                   frame_wide_shift_jumps=sum(j["frame_wide_shift"] for j in js),
                   frame_wide_shift_frames=len({j["row_to"] for j in js if j["frame_wide_shift"]}),
                   lands_on_other_slot=sum(j["lands_on_other_slot"] for j in js),
                   discontinuity_candidates=sum(j["discontinuity_candidate"] for j in js),
                   label_transfer_candidates=sum(j["label_transfer_candidate"] for j in js),
                   reciprocal_exchange_jumps=sum(j["reciprocal_exchange"] for j in js), distance_m=_q(np.array([j["distance_m"] for j in js], float))),
        substitutions=qr.substitutions, encounters=enc, per_slot=per_slot)


# ---------------------------------------------------------------- outputs

def output_paths(raw_root: str | Path, match_id: object, out_root: str | Path | None = None) -> dict[str, Path]:
    base = Path(out_root) if out_root is not None else approved_output_root(raw_root) / OUTPUT_DIR
    return {sfx: safe_output_path(raw_root, match_id, EVENT_VERSION, base, suffix=sfx) for sfx in OUTPUT_SUFFIXES}


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def code_digest() -> dict[str, str]:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}


def write_outputs(raw_root: str | Path, c: RawCache, qr: QualityResult, encounters: list[dict], summary: dict, manifest: dict,
                  out_root: str | Path | None = None) -> dict[str, Path]:
    """Write all outputs or none. Existing outputs are never replaced."""
    paths = output_paths(raw_root, c.match_id, out_root)
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise IngestionError(f"outputs already exist and are never overwritten: {existing}")
    next(iter(paths.values())).parent.mkdir(parents=True, exist_ok=True)
    paths = output_paths(raw_root, c.match_id, out_root)                      # re-validate once the directory exists
    blobs: dict[str, Any] = {
        ".encounters.jsonl": _jsonl(encounters), ".jumps.jsonl": _jsonl(qr.jumps),
        ".summary.json": json.dumps(summary, indent=1, sort_keys=True).encode()}
    tmp = {sfx: safe_output_path(raw_root, c.match_id, EVENT_VERSION, paths[sfx].parent.parent, suffix=sfx + ".tmp") for sfx in OUTPUT_SUFFIXES}
    digests = {}
    try:
        for sfx in OUTPUT_SUFFIXES[:-1]:
            with open(tmp[sfx], "xb") as f:
                if sfx == ".cache.npz":
                    np.savez_compressed(f, **c.arrays())
                elif sfx == ".quality.npz":
                    np.savez_compressed(f, mask=qr.mask, speed=qr.speed, **{f"step_{k}": v for k, v in qr.steps.items()})
                else:
                    f.write(blobs[sfx])
            digests[paths[sfx].name] = hashlib.sha256(tmp[sfx].read_bytes()).hexdigest()
        manifest = dict(manifest, outputs_sha256=digests)
        with open(tmp[".manifest.json"], "xb") as f:
            f.write(json.dumps(manifest, indent=1, sort_keys=True).encode())
        for sfx in OUTPUT_SUFFIXES:
            if paths[sfx].exists():
                raise IngestionError(f"output appeared while writing; not overwritten: {paths[sfx]}")
            os.rename(tmp[sfx], paths[sfx])
    finally:
        for p in tmp.values():
            if p.exists():
                p.unlink()
    return paths


def build_manifest(md: MatchData, c: RawCache, cfg: M11Config, split: dict[str, Any] | None) -> dict[str, Any]:
    src, _ = md.dataset.tracking_source(md.match_id)
    return dict(
        schema=SCHEMA_VERSION, match_id=c.match_id, event_version=EVENT_VERSION.value, coordinate_policy=CoordinatePolicy.RAW.value,
        provider_smoothed_coordinates_used=False, config=cfg.as_dict(), quarantine_periods=list(c.quarantine_periods), split=split,
        tracking_source=dict(path=c.tracking_path, bytes=src.stat().st_size), quality_flag_bits=[f.value for f in QualityFlag],
        mask_bits=Qm.MASK_BITS, sample_categories=list(Qm.SAMPLE_CATEGORIES), encounter_categories=list(CATEGORIES),
        code_sha256=code_digest(), python=platform.python_version(), numpy=np.__version__)


def run_match(raw_root: str | Path, match_id: object, *, cfg: M11Config = M11Config(), out_root: str | Path | None = None,
              quarantine_periods=(), split: dict[str, Any] | None = None, max_frames: int | None = None,
              extra_manifest: dict[str, Any] | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    t0 = time.perf_counter()
    output_paths(raw_root, match_id, out_root)                                 # fail on an unsafe destination before reading anything
    md = MatchData.open(raw_root, match_id, EVENT_VERSION, policy=CoordinatePolicy.RAW, quarantine_periods=quarantine_periods)
    c = build_cache(md, max_frames=max_frames)
    qr = assess(c, md.events(), cfg)
    enc = find_encounters(c, qr, cfg)
    summary = summarize(c, qr, enc, cfg)
    manifest = build_manifest(md, c, cfg, split) | dict(max_frames=max_frames, processing=_processing(t0, "raw_tracking")) | (extra_manifest or {})
    return write_outputs(raw_root, c, qr, enc, summary, manifest, out_root), summary


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MiB (measurement only; None when unavailable)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                   "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:                                    # measurement only; never fails a run
        return None


def _processing(t0: float, source: str) -> dict[str, Any]:
    return dict(source=source, seconds_before_write=round(time.perf_counter() - t0, 1), peak_memory_mb_before_write=peak_memory_mb())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_from_cache(raw_root: str | Path, cache_path: str | Path, *, cfg: M11Config = M11Config(), out_root: str | Path | None = None,
                   split: dict[str, Any] | None = None, extra_manifest: dict[str, Any] | None = None,
                   expected_match_id: object | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """Recompute quality flags and the catalogue from a previously written cache, without a raw tracking pass.

    The cache must match the SHA-256 recorded in its own manifest, and the tracking file it was built from must still be the
    dataset's tracking source with the same size. Metadata, roster and events are re-read (they are small)."""
    t0 = time.perf_counter()
    cache_path = Path(cache_path)
    sibling = lambda sfx: cache_path.with_name(cache_path.name.replace(".cache.npz", sfx))   # noqa: E731
    src_manifest = json.loads(sibling(".manifest.json").read_text(encoding="utf-8"))
    digest = _sha256(cache_path)
    if src_manifest.get("outputs_sha256", {}).get(cache_path.name) != digest:
        raise IngestionError(f"{cache_path} does not match the checksum in its manifest; not used")
    mid = str(src_manifest["match_id"])
    if expected_match_id is not None and str(expected_match_id) != mid:
        raise IngestionError(f"the cache belongs to match {mid}, not {expected_match_id}")
    output_paths(raw_root, mid, out_root)
    md = MatchData.open(raw_root, mid, EVENT_VERSION, policy=CoordinatePolicy.RAW, quarantine_periods=src_manifest["quarantine_periods"])
    src, ref = md.dataset.tracking_source(mid)
    if ref.path != src_manifest["tracking_source"]["path"] or src.stat().st_size != src_manifest["tracking_source"]["bytes"]:
        raise IngestionError(f"the cache was built from {src_manifest['tracking_source']}, which is no longer the dataset's tracking source")
    src_summary = json.loads(sibling(".summary.json").read_text(encoding="utf-8"))
    c = load_cache(cache_path, defaults=dict(match_id=int(mid), fps=md.metadata.fps, tracking_path=ref.path,
                                             quarantine_periods=tuple(src_manifest["quarantine_periods"]),
                                             samples_without_jersey=src_summary["samples_without_jersey"],
                                             duplicate_slot_samples=src_summary["duplicate_slot_samples"]))
    qr = assess(c, md.events(), cfg)
    enc = find_encounters(c, qr, cfg)
    summary = summarize(c, qr, enc, cfg)
    manifest = (build_manifest(md, c, cfg, split) | dict(max_frames=src_manifest.get("max_frames"))
                | dict(derived_from_cache=dict(path=str(cache_path), sha256=digest, source_schema=src_manifest.get("schema"),
                                               source_code_sha256=src_manifest.get("code_sha256")))
                | dict(processing=_processing(t0, "cache")) | (extra_manifest or {}))
    return write_outputs(raw_root, c, qr, enc, summary, manifest, out_root), summary
