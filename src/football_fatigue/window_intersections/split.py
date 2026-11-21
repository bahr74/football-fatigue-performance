"""Frozen internal held-out split of the 40 FIFA development matches (M11 Stage 2): ``m11-stage2-split-v1``.

Rule (deterministic, documented before any training):

1. Population: the development matches of the frozen ``m11-encounters-v1/splits.json``; official validation and test are never read.
2. Excluded from the held-out pool: the Stage 1 pilot matches and every match already used to fit or select an M11 model. The
   exposed set is derived from the earlier modelling outputs on disk (``EXPOSURE_SOURCES``) and must equal ``EXPECTED_EXPOSED``.
3. The remaining matches are ranked by ``sha256(SALT + "|" + match_id)``; the first ``N_HELDOUT`` form the held-out set, the other
   development matches (including the exposed ones) are training / grouped cross-validation.

All 40 development matches were previously used for descriptive extractions only (no model fitted); this is recorded.
The split file is written once (never overwritten) together with a manifest of its SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SALT = "m11-stage2-heldout-v1"
N_HELDOUT = 8
PILOTS = ("3812", "3816")
EXPOSURE_SOURCES = ("m11-matching-v3", "m11-recovery-v1/views", "m11-contacts-v1", "m11-intersections-v3")
EXPECTED_EXPOSED = ("10506", "3812", "3814", "3815", "3816", "3829", "3833", "3836", "3847", "3851")
APPROVED_HELDOUT = ("3822", "3824", "3834", "3839", "3844", "10510", "10514", "10517")
SPLIT_FILE = "split.json"


def rank(matches: list[str], salt: str = SALT) -> list[str]:
    return sorted(matches, key=lambda m: hashlib.sha256(f"{salt}|{m}".encode()).hexdigest())


def make_split(dev: list[str], exposed: set[str], n: int = N_HELDOUT, salt: str = SALT) -> dict[str, Any]:
    if not exposed >= set(PILOTS):
        raise ValueError("pilot matches must be excluded from the held-out pool")
    pool = [m for m in dev if m not in exposed]
    if len(pool) < n:
        raise ValueError("held-out pool too small")
    ranked = rank(pool, salt)
    held = sorted(ranked[:n], key=int)
    train = sorted((m for m in dev if m not in held), key=int)
    return dict(salt=salt, n_heldout=n, development=sorted(dev, key=int), excluded_from_pool=sorted(exposed, key=int), pool_ranked=ranked,
                heldout=held, train_cv=train)


def exposed_from_disk(proc: Path) -> tuple[set[str], dict[str, list[str]]]:
    found: dict[str, list[str]] = {}
    for src in EXPOSURE_SOURCES:
        d = proc / src
        found[src] = sorted({p.stem for p in d.iterdir() if p.stem.isdigit()}, key=int) if d.exists() else []
    return set().union(*map(set, found.values())) | set(PILOTS), found


def load_split(proc: Path, name: str = "m11-stage2-split-v1") -> dict[str, Any]:
    """Read and hash-verify a frozen split."""
    d = proc / name
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((d / SPLIT_FILE).read_bytes()).hexdigest() != man["split_sha256"]:
        raise RuntimeError(f"{d / SPLIT_FILE} does not match its manifest")
    return json.loads((d / SPLIT_FILE).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    from .__main__ import PROC, SPLITS, dev_matches, dumps

    ap = argparse.ArgumentParser(description="Freeze the M11 Stage 2 internal held-out split (written once)")
    ap.add_argument("--out", default="m11-stage2-split-v1")
    a = ap.parse_args(argv)
    out = PROC / a.out
    if out.exists():
        raise SystemExit(f"{out} exists; the split is frozen and never overwritten")
    dev = dev_matches()
    exposed, found = exposed_from_disk(PROC)
    if exposed != set(EXPECTED_EXPOSED):
        raise SystemExit(f"exposed set on disk {sorted(exposed)} differs from the documented {sorted(EXPECTED_EXPOSED)}")
    sp = make_split(dev, exposed)
    if tuple(sp["heldout"]) != tuple(sorted(APPROVED_HELDOUT, key=int)):
        raise SystemExit(f"rule gives {sp['heldout']}, not the approved list {APPROVED_HELDOUT}")
    sp.update(schema="m11-stage2-split-v1", rule=__doc__.strip(), exposure_sources=found, approved_heldout=list(APPROVED_HELDOUT),
              source_splits_file=str(SPLITS.relative_to(PROC)).replace("\\", "/"),
              source_splits_sha256=hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
              note="all 40 development matches were used earlier for descriptive extraction only (no model fitted on the held-out 8)")
    out.mkdir(parents=True)
    text = dumps(sp)
    (out / SPLIT_FILE).write_bytes(text.encode("utf-8"))
    (out / "manifest.json").write_text(dumps(dict(split_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), heldout=sp["heldout"],
                                                  n_train_cv=len(sp["train_cv"]))), encoding="utf-8")
    print(dumps(dict(heldout=sp["heldout"], train_cv=len(sp["train_cv"]), out=str(out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
