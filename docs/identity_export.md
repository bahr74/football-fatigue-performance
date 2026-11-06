# Identity-provenance export (M7-0)

Module: `football_fatigue.vision.identity_export`. Opt-in, read-only over `frames.jsonl` records; it changes no identity code,
threshold, default output or analytics behaviour (`analytics/cv.py` does not import it).

## What it is
A pixel-space export of *safe observation segments*. A segment is a maximal run of consecutive **observed** foot-point samples that
share one tracklet, one local tracker id, one global-id hypothesis, one camera/view, one association provenance and one trust class, with
no foot-point jump. Nothing is bridged across a boundary; each boundary is listed in `transitions` with `bridged: false`.

It is **not** a metric product: coordinates are image pixels (`coordinate_system.metric_valid = false`), there is no speed, distance or
sprint total, and `claims` states `metric_distance`, `global_identity_validated` and `fatigue_or_load_metric` are all `false`.

## API
```python
from football_fatigue.vision.identity_export import (ExportConfig, build_export, validate_export, write_export, read_export,
                                                      segment_path_length_px)
export = build_export(records, source={"match_id": "117093", "camera_id": "panorama"}, config=ExportConfig())
write_export("out.json", export)       # validates; opens with 'x' (never overwrites)
export = read_export("out.json")       # validates schema and segment invariants
```
CLI: `python -m football_fatigue.vision.identity_export frames.jsonl --output NEW.json --match-id ID [--camera-id C] [--max-missing-frames N] [--max-jump-box-heights F]`.

`records` are `frames.jsonl` records, ideally from `identity.annotate_records` (fields `local_track_id`, `tracklet_id`, `global_id`,
`identity_status`, `identity_confidence`, `identity_confidence_local`, `association_status`, `association_link_confidence`,
`position_status`). Plain tracker output is accepted: it exports as `local_only` with reason `not_identity_annotated`. An optional
per-record `camera_id` (or `source.camera_id`) is compared between consecutive samples.

`ExportConfig` (defaults conservative): `max_missing_frames=0`, `max_jump_box_heights=1.5` (foot-point jump per frame in own box
heights), `min_attributable_samples=25`, `trusted_statuses=("confirmed",)`, `require_provenance=True`.

## Segment guards (a segment ends when any holds between consecutive samples)
`missing_observations` (frame gap > `max_missing_frames`), `global_id_change`, `camera_change`, `association_change`, `trust_change`,
`coordinate_discontinuity`, `duplicate_or_unordered_frame`; samples with `position_status != observed` and boxes without a valid bbox are
excluded (interpolated points are kept apart in `interpolated_samples`). Samples are grouped by (`tracklet_id`, `local_track_id`) first, so a
tracklet boundary is always a segment boundary.

`global_id_use` is `attributable` only if every sample is `confirmed` **and** `association_status == "none"` **and** the segment has at
least `min_attributable_samples` samples; otherwise `local_only` with `global_id_use_reasons` in {`status_not_confirmed`,
`linked_unvalidated`, `provenance_not_recorded`, `not_identity_annotated`, `short_segment`}. "Attributable" means "no known reason to
withhold the global id", not "verified correct".

`transitions` lists consecutive segments of the same global id: `tracklet_transition`, `within_tracklet_break`, or `overlap_conflict`
(one global id on two segments at once, i.e. global-id reuse that cannot be one trajectory). A consumer must treat each segment as the unit of
attribution and may not accumulate across a transition.

## Data contract
Schema: `data/processed/vision_eval/m70-identity-export-v1/identity_export.schema.json` (generated from `EXPORT_SCHEMA`, which
`validate_export` checks without a third-party validator, plus the invariants: array lengths, strictly increasing frames within the allowed
gap, no `bridged` transition, `attributable` segments contain only trusted samples). Segment arrays are columnar and parallel:
`frames`, `timestamp_s`, `x_px`, `y_px`, `box_height_px`, `identity_status`, `identity_confidence` (as recorded), `identity_confidence_local`
(link-independent), `association_link_confidence` (weakest incident link probability, null when none). Example (real excerpt):
`.../m70-identity-export-v1/example_export.json`.

## Known limitations
Identity swaps inside a tracklet are not detectable (M6.3-D) and are not cut unless they produce a coordinate jump or a global-id change.
Metres per pixel vary across the panorama. No pitch calibration exists for most matches (see `NEXT_SYSTEM_READINESS.md`). The export has not
been connected to the CV board or to movement analytics.
