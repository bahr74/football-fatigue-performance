"""Deterministic, frozen, match-disjoint split manifest for the 64 FIFA World Cup 2022 matches (M11).

3812, 3816 and 10506 were inspected during the M11 audits and are fixed in development. The remaining matches are ordered by
SHA-256 of ``"<SEED>:<match_id>"`` and assigned in that order to development, validation and locked test. Release
disagreement and extra-time anomalies are *attributes* of a match, never extra splits. Locked test matches must not be used
for tuning; the CLI refuses to process them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.ingestion.validation import _assert_no_links, _under

SEED = "m11-encounters-v1"
SIZES = {"development": 40, "validation": 12, "test": 12}
FIXED_DEVELOPMENT = ("3812", "3816", "10506")
# May/root event coordinates disagree with the tracking files (docs/event_version_comparison.md, docs/dataset_schema_audit.md)
RELEASE_DISAGREEMENT = ("10502", "10512", "3816", "3817", "3829", "3835", "3841", "3845", "3852", "3859")
# periods 3-4 flagged suspected_side_identity_anomaly (docs/dataset_schema_audit.md section 7); attribution is quarantined
EXTRA_TIME_ANOMALY = ("10506", "10510", "10511", "10517")
EXTRA_TIME_QUARANTINE = (3, 4)
MANIFEST_NAME = "splits.json"


def assign(match_ids: Iterable[str]) -> dict[str, Any]:
    ids = sorted({str(m) for m in match_ids}, key=lambda m: (len(m), m))
    if len(ids) != sum(SIZES.values()):
        raise IngestionError(f"expected {sum(SIZES.values())} matches, got {len(ids)}")
    missing = [m for m in FIXED_DEVELOPMENT + RELEASE_DISAGREEMENT + EXTRA_TIME_ANOMALY if m not in ids]
    if missing:
        raise IngestionError(f"matches named by the split policy are not in the dataset: {missing}")
    rest = sorted((m for m in ids if m not in FIXED_DEVELOPMENT), key=lambda m: hashlib.sha256(f"{SEED}:{m}".encode()).hexdigest())
    n_dev = SIZES["development"] - len(FIXED_DEVELOPMENT)
    groups = {"development": list(FIXED_DEVELOPMENT) + rest[:n_dev], "validation": rest[n_dev:n_dev + SIZES["validation"]],
              "test": rest[n_dev + SIZES["validation"]:]}
    matches = {m: dict(split=s, release_disagreement=m in RELEASE_DISAGREEMENT, extra_time_anomaly=m in EXTRA_TIME_ANOMALY)
               for s, ms in groups.items() for m in ms}
    return dict(schema=f"{SEED}-splits", seed=SEED, sizes=SIZES, fixed_development=list(FIXED_DEVELOPMENT),
                splits={s: sorted(ms, key=lambda m: (len(m), m)) for s, ms in groups.items()},
                matches={m: matches[m] for m in ids},
                note="match-disjoint; attributes do not form extra splits; never tune on 'test'")


def default_quarantine(match_id: str) -> tuple[int, ...]:
    return EXTRA_TIME_QUARANTINE if str(match_id) in EXTRA_TIME_ANOMALY else ()


def manifest_path(raw_root: str | Path, out_root: str | Path) -> Path:
    raw = Path(raw_root).resolve()
    approved = approved_output_root(raw_root)
    base = Path(os.path.abspath(out_root))
    if _under(base, raw) or not _under(base, approved):
        raise IngestionError(f"output location {base} is not inside the approved output root {approved} (or lies inside the raw data)")
    dest = base / MANIFEST_NAME
    _assert_no_links(raw.parent, dest)
    return dest


def write_manifest(raw_root: str | Path, out_root: str | Path, manifest: dict[str, Any]) -> Path:
    """Freeze the manifest. An existing manifest is never replaced; an identical one is accepted."""
    dest = manifest_path(raw_root, out_root)
    text = json.dumps(manifest, indent=1, sort_keys=True)
    if dest.exists():
        if dest.read_text(encoding="utf-8") != text:
            raise IngestionError(f"a different split manifest is already frozen at {dest}; it is never replaced")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = manifest_path(raw_root, out_root)
    with open(dest, "x", encoding="utf-8") as f:
        f.write(text)
    return dest
