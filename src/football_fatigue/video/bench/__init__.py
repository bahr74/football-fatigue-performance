"""Detector and tracker benchmark (Session B, research track; read-only on every input).

* ``registry``  - detector candidates with provenance, licence and execution status (nothing is downloaded here).
* ``frames``    - deterministic, ground-truth-stratified frame manifest over the four development matches.
* ``detectors`` - one adapter interface (cache, Ultralytics tiled, unavailable) and a resumable, write-once runner.
* ``evaluate``  - AP50, AP50:95, precision/recall, recall by size and annotated visibility, duplicates, FP/frame.
* ``trackers``  - tracker comparison on identical cached detections (supervision ByteTrack and the Ultralytics
  ByteTrack / BoT-SORT / OC-SORT / Deep OC-SORT / FastTracker / TrackTrack implementations), scored with the same
  HOTA/CLEAR/Identity evaluator as the production replay.
* ``gallery``   - comparison PNGs and short annotated clips from executed runs only.

Outputs live under ``data/processed/video/_bench/`` (versioned, write-once). Reserved matches are refused.
"""
