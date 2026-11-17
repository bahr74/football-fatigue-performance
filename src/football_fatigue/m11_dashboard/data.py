"""Read-only data access for the Step 8 dashboard: development matches, task B instances, public metadata, provider reference.

Three views of one accepted encounter are kept apart:

- the **instance** (``root.instances_b.jsonl``): the anonymous ``in-k`` / ``out-k`` tracklets the model sees, with the
  ambiguity interval withheld;
- the **public context**: period, time, frames, player count, gap, tier and quality flags. It is copied key by key from the
  reference record (:data:`PUBLIC_KEYS`) and holds no slot, jersey, player, team or expected mapping;
- the **provider reference**: identities and the expected mapping, returned only by :meth:`MatchData.provider_reference`
  (the app calls it after "Reveal Provider Reference").

Only development matches of the frozen split are served; validation and locked-test matches are refused. Nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.splits import MANIFEST_NAME
from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import OUTPUT_DIR as MATCHING_DIR, SCHEMA_VERSION, SOURCE_DIR as ENCOUNTERS_DIR, _sha256, output_paths
from football_fatigue.matching.evaluation import gap_bin

PUBLIC_KEYS = ("instance_id", "match_id", "period", "tier", "below_high_reasons", "start_frame", "end_frame", "start_time_s", "end_time_s",
               "rows", "encounter_ids", "step1_categories", "discontinuity_evidence", "n_participants")
# reference keys that reveal identity or are computed from the expected mapping; never part of the public context
HIDDEN_KEYS = ("incoming", "outgoing", "expected", "without_incoming", "without_outgoing", "team_relation", "difficulty")
LABEL_TIER, EXPLORATORY_TIER, UNVERIFIED_TIER = ml.LABEL_TIER, ml.EXPLORATORY_TIER, ml.UNVERIFIED_TIER
CACHE_KEYS = ("x", "y", "visibility", "video_time_s", "slot_side", "slot_jersey")


class DashboardError(RuntimeError):
    """A missing or inconsistent artifact; the message says which file and why."""


def processed_root(raw_root) -> Path:
    return approved_output_root(raw_root)


def split_manifest(raw_root) -> dict:
    path = processed_root(raw_root) / ml.SPLITS_DIR / MANIFEST_NAME
    if not path.exists():
        raise DashboardError(f"the frozen split manifest {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_spec(raw_root, frozen_dir=None) -> dict:
    d = Path(frozen_dir) if frozen_dir is not None else processed_root(raw_root) / validate.FROZEN_DIR
    path = d / "spec.json"
    if not path.exists():
        raise DashboardError(f"the frozen specification {path} is missing (run Step 6 freeze first)")
    return json.loads(path.read_text(encoding="utf-8"))


def require_development(split: dict, match_id) -> str:
    """The match ID as a string, or :class:`DashboardError` unless the frozen split lists it as development."""
    mid = str(match_id)
    try:
        validate.check_split(split, [mid], "development")
    except IngestionError as e:
        raise DashboardError(str(e)) from None
    return mid


def development_matches(raw_root, *, frozen_dir=None, matching_root=None) -> list[str]:
    """Development matches the frozen model was fitted on that also have a Step 2-3 benchmark output, in spec order.

    Discovered from the frozen specification and the split manifest; no ID is hard-coded."""
    split = split_manifest(raw_root)
    spec = frozen_spec(raw_root, frozen_dir)
    split_path = processed_root(raw_root) / ml.SPLITS_DIR / MANIFEST_NAME
    if _sha256(split_path) != spec["splits"]["manifest_sha256"]:
        raise DashboardError("the split manifest changed after the freeze")
    out = []
    for m in spec["splits"]["development"]:
        if split["matches"].get(str(m), {}).get("split") != "development":
            continue
        if output_paths(raw_root, m, matching_root)[".manifest.json"].exists():
            out.append(str(m))
    return out


# ---------------------------------------------------------------- one match

def public_context(ref: dict, inst: dict) -> dict[str, Any]:
    """Identity-free metadata of one encounter (see the module docstring)."""
    ctx = {k: ref[k] for k in PUBLIC_KEYS}
    amb = float(inst["ambiguity_interval_s"][1])
    ctx.update(
        ambiguity_s=amb, gap_bin=gap_bin(amb), n_incoming=len(inst["incoming"]), n_outgoing=len(inst["outgoing"]),
        min_distance_m=ref["difficulty"]["min_distance_m"],       # minimum over the Step 1 pair encounters: symmetric, no mapping
        suspicious=ref["tier"] == UNVERIFIED_TIER or bool(ref["discontinuity_evidence"]),
        # reference-derived (uses the expected mapping): offered as a selection filter only, never shown or given to the model
        filter_sharp_turn="sharp_turn" in ref["difficulty"]["reasons"])
    return ctx


def training_pairs(inst: dict, tier: str, cfg: ml.MLConfig = ml.MLConfig()) -> set[tuple[str, str]]:
    """Pairs of this instance that the frozen model was fitted on (high tier, no ESTIMATED sample in either endpoint window),
    from the instance alone: the same rule as ``ml.pair_labels`` status ``train``."""
    if tier != LABEL_TIER:
        return set()
    est_in = {a["tracklet_id"]: 2 in np.asarray(a["visibility"])[ml._window(a, True, cfg.estimated_window_s)] for a in inst["incoming"]}
    est_out = {b["tracklet_id"]: 2 in np.asarray(b["visibility"])[ml._window(b, False, cfg.estimated_window_s)] for b in inst["outgoing"]}
    return {(i, o) for i in est_in for o in est_out if not est_in[i] and not est_out[o]}


@dataclass
class MatchData:
    match_id: str
    instances: dict[str, dict]          # matcher-visible task B instances, file order
    contexts: dict[str, dict]           # public metadata, same keys
    sources: dict[str, str]             # verified SHA-256 of the files read
    _references: dict[str, dict] = field(repr=False, default_factory=dict)

    def provider_reference(self, instance_id: str) -> dict:
        """Evaluation-only provider identities and expected mapping. Call only after the user asked to reveal it."""
        return self._references[instance_id]

    def participant_slots(self, instance_id: str) -> set[int]:
        """Tracking slots of every participant, used only to keep them out of the context layer (never displayed)."""
        r = self._references[instance_id]
        return ({v["slot"] for v in r["incoming"].values()} | {v["slot"] for v in r["outgoing"].values()}
                | {v["slot"] for v in r["without_incoming"]} | {v["slot"] for v in r["without_outgoing"]})


def _read_verified(path: Path, manifest: dict) -> list[dict]:
    if not path.exists():
        raise DashboardError(f"{path} is missing")
    if manifest["outputs_sha256"].get(path.name) != _sha256(path):
        raise DashboardError(f"{path} does not match its manifest checksum")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def load_match(raw_root, match_id, *, matching_root=None) -> MatchData:
    """Task B instances, public contexts and (hidden) references of one development match, checksum-verified."""
    mid = require_development(split_manifest(raw_root), match_id)
    paths = output_paths(raw_root, mid, matching_root)
    mpath = paths[".manifest.json"]
    if not mpath.exists():
        raise DashboardError(f"no Step 2-3 benchmark output for match {mid}: {mpath} is missing")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA_VERSION or str(m.get("match_id")) != mid or (m.get("split") or {}).get("split") != "development":
        raise DashboardError(f"{mpath} is not a {SCHEMA_VERSION} development output of match {mid}")
    if m.get("synthetic_swaps_injected") is not False or m.get("provider_smoothed_coordinates_used") is not False:
        raise DashboardError(f"{mpath}: not a RAW, swap-free benchmark")
    insts = _read_verified(paths[".instances_b.jsonl"], m)
    refs = {r["instance_id"]: r for r in _read_verified(paths[".references.jsonl"], m)}
    instances, contexts = {}, {}
    for inst in insts:
        iid = inst["instance_id"]
        if iid not in refs:
            raise DashboardError(f"instance {iid} has no reference record")
        instances[iid] = inst
        contexts[iid] = public_context(refs[iid], inst)
    sources = {p.name: _sha256(p) for p in (paths[".instances_b.jsonl"], paths[".references.jsonl"], mpath)}
    return MatchData(mid, instances, contexts, sources, refs)


# ---------------------------------------------------------------- RAW tracking around one encounter

def load_cache(raw_root, match_id, *, encounters_root=None) -> dict[str, np.ndarray]:
    """The Step 1 RAW cache arrays needed for the pitch (checksum-verified against the Step 1 manifest)."""
    mid = require_development(split_manifest(raw_root), match_id)
    base = Path(encounters_root) if encounters_root is not None else processed_root(raw_root) / ENCOUNTERS_DIR
    d = base / mid
    path, mpath = d / "root.cache.npz", d / "root.manifest.json"
    if not path.exists() or not mpath.exists():
        raise DashboardError(f"the Step 1 RAW cache {path} or its manifest is missing")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m["outputs_sha256"].get(path.name) != _sha256(path):
        raise DashboardError(f"{path} does not match its manifest checksum")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in CACHE_KEYS}


def window_rows(ctx: dict) -> dict[str, tuple[int, int]]:
    return {k: (int(a), int(b)) for k, (a, b) in ctx["rows"].items()}


def raw_window(cache: dict[str, np.ndarray], ctx: dict, exclude_slots: set[int]) -> dict[str, Any]:
    """RAW positions of every non-participant slot over the before, during and after rows (context layer).

    Times are relative to the start of the ambiguity interval, as in the instance. Participants are excluded in every row, so the
    withheld interval of the players being matched never appears here."""
    w = window_rows(ctx)
    r0, r1 = w["before"][0], w["after"][1]
    rows = np.arange(r0, r1)
    t = cache["video_time_s"][rows] - cache["video_time_s"][w["during"][0]]
    keep = [s for s in range(cache["x"].shape[1]) if s not in exclude_slots]
    phase = np.where(rows < w["during"][0], "before", np.where(rows < w["during"][1], "during", "after"))
    return dict(rows=rows, t=t, phase=phase, slots=keep, x=cache["x"][r0:r1][:, keep], y=cache["y"][r0:r1][:, keep],
                visibility=cache["visibility"][r0:r1][:, keep], side=cache["slot_side"][keep])


def hidden_interval(cache: dict[str, np.ndarray], ctx: dict, reference: dict) -> list[dict[str, Any]]:
    """Reference-only: the provider-labelled RAW positions of each participant during the withheld interval.

    Needs the provider reference, so the app calls it only after the user revealed it. The model never receives these rows."""
    w = window_rows(ctx)
    s, e = w["during"]
    t = cache["video_time_s"][s:e] - cache["video_time_s"][s]
    out = []
    seen = set()
    for side_key in ("incoming", "outgoing"):
        for tid, info in reference[side_key].items():
            if info["slot"] in seen:
                continue
            seen.add(info["slot"])
            x, y = cache["x"][s:e, info["slot"]], cache["y"][s:e, info["slot"]]
            ok = np.isfinite(x) & np.isfinite(y)
            out.append(dict(label=identity_label(info), slot=info["slot"], t=t[ok], x=x[ok], y=y[ok],
                            visibility=cache["visibility"][s:e, info["slot"]][ok]))
    return out


def identity_label(info: dict) -> str:
    """Side and shirt number only; names and player IDs are never displayed."""
    return f"{info['side']} #{info['jersey']}"


def period_clock(meta: Any, period: int, video_time_s: float) -> float | None:
    """Seconds since the start of the period from the match metadata (same video clock), or None when unknown."""
    start = getattr(meta, f"start_period_{period}_s", None) if meta is not None else None
    return None if start is None else float(video_time_s) - float(start)


def match_metadata(raw_root, match_id) -> Any | None:
    """RAW match metadata (pitch size, team names, period starts), or None when it cannot be read."""
    from football_fatigue.ingestion import Dataset

    try:
        return Dataset(raw_root).metadata(str(match_id))
    except Exception:          # the app falls back to 105 x 68 and says so
        return None
