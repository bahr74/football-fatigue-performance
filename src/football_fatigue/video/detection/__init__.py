"""Detection-quality diagnostics for Session B (development SoccerTrack sequences only).

* ``stages``: per-frame attribution of every ground-truth player to the pipeline stage that kept or lost it.
* ``diagnose``: batch diagnosis of a source from the existing detection cache and tracker replay (CPU only).
* ``merge``: tile-merge policies for the post-processing A/B on a raw per-tile capture.
* ``capture``: the raw per-tile capture (GPU; runs only with explicit approval).
"""

DIAG_VERSION = "diag-v1"
