# M11 — Advisor-directed trajectory-intersection pipeline: Stages B–C prototype (match 3812)

Status: **one-match prototype, awaiting advisor review (Stage D).** No model was trained, nothing was tuned, no other match was
processed. Separate experiment: new package `src/football_fatigue/intersections/`, new output root
`data/processed/m11-intersections-v1/`, new tests `tests/test_intersections.py`. Earlier M11 datasets, the frozen model
(`m11-frozen-v1`, model SHA-256 `629cca77…`, spec `6fa1df5c…`), the locked-test outputs and the Step 8 dashboard are untouched
(all 639 earlier `m11-*` processed files were verified byte-for-byte by size and mtime before and after; RAW snapshot unchanged).

> **Scope of claims.** A geometric trajectory intersection is the advisor's *operational* definition of a possible occlusion;
> it is **not** verified visual occlusion, and it is **not** a confirmed identity swap. Sample labels are provider trajectory
> continuity (the provider's slot labels), not independently verified identity. Match 3812 is a *development* match; nothing
> here is evidence of generalisation, and the consumed locked test was not used for any choice.

Run (development matches only; validation and test matches are refused; outputs are never overwritten):

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections run --match 3812 --dry-run   # summary only, writes nothing
.\.venv\Scripts\python.exe -m football_fatigue.intersections run --match 3812             # writes data/processed/m11-intersections-v1/3812/
```

## 1. Input

Step 1 RAW cache and quality flags (`m11-encounters-v2/<match>/root.cache.npz`, `root.quality.npz`), loaded through
`matching.benchmark.load_source`, which verifies every file against its manifest and the manifest against the dataset's tracking
source. RAW coordinates only; provider-smoothed coordinates are never read.

## 2. Detection (`detect.py`, `geometry.py`)

**Polyline segments.** One segment per Step 1 *step* (between consecutive position-update instants of a slot). A step is joined
only when it is valid in Step 1 (same period, gap ≤ 0.5 s), so a path is **never joined across a missing-data gap**. Excluded, in
this order and counted exclusively: invalid gap / period change, implausible jump (Step 1 flag), unresolved / ambiguous
attribution or side overcount, duplicate-position artifact (§4), zero length.

**Windows.** Centres every 1 s, window ±2.5 s (5 s), full windows only, per period. A segment belongs to a window when it lies
fully inside it. Every unordered pair of slots with segments in the window is tested (231 pairs with 22 players); a pair whose
bounding boxes do not overlap is rejected before segment tests (exact, not an approximation).

**Segment intersection** (vectorised, absolute tolerance `tol_m = 1e-6` m): `proper` (crossing strictly inside both),
`endpoint` (contact at a vertex of at least one segment, incl. T-junctions), `collinear` (overlap of positive length; the point
is the overlap midpoint). Parallel non-collinear and degenerate segments give no hit.

**Event types.** `proper_crossing`; for vertex contacts the positions 0.2 s before/after decide `vertex_crossing` (A changes side
of B's local line), `touch` (same side) or `vertex_undetermined`; `collinear_overlap`.

**Traversal times** are interpolated linearly along each segment's timestamps. The time difference `t_B − t_A` is stored as
metadata (`time_diff_s`, `abs_time_diff_s`, `simultaneous` = |Δt| ≤ 0.2 s) and is **not** a filter. No proximity (e.g. 2 m)
selection is applied; `min_synchronous_distance_m` (closest same-frame distance within ±0.5 s of the crossing) is metadata only.

**Deduplication**, two stages, deterministic:
1. the same (pair, segment-of-A, segment-of-B) hit found in several overlapping windows → one record that keeps the list of
   windows (`windows`, `n_windows`);
2. hits of one pair on *adjacent contiguous* segments of both players are one physical contact only when they are the same
   point (≤ 1 mm: a contact at a shared vertex is found on both neighbouring segments) or a collinear chain. Distinct proper
   crossings are never merged, even on the same segment (tested: out-and-back across one segment → 2 events).

`event_id` = SHA-256 of `match|period|slotA|slotB|source row of A's first segment|source row of B's first segment` (stable,
independent of internal indices). **Pair clusters** (`cluster_id`, `cluster_size`) chain events of the same pair within 5 s of
each other: audit metadata for grouping, not deduplication.

## 3. Four-segment samples (`samples.py`)

For an event with traversal times t_P, t_Q: `t_lo = min`, `t_hi = max`, `t_mid` = midpoint.

| span | time range (same for both players) | role |
|---|---|---|
| incoming | `[t_mid − 2.5 s, t_lo)` | model input: in-0 (query A), in-1 (B) |
| withheld | `[t_lo, t_hi]` (pad configurable, 0 here) | never model input |
| outgoing | `(t_hi, t_mid + 2.5 s]` | model input: out-0, out-1 (anonymous) |

**Target:** does A continue into out-0 or out-1? Label = the outgoing tracklet cut from A's provider slot.
**Leakage controls:** which player is A and the outgoing order come from two independent bits of a seeded SHA-256 of the event
key (balanced, unrelated to slot number, time or file order); both incoming tracklets share identical time stamps, as do both
outgoing ones; files are ordered by an opaque `instance_id`; instances carry only whitelisted keys (`instance_id, schema, task,
units, ambiguity_interval_s, query, incoming, outgoing, during`; tracklets `tracklet_id, t, x, y, visibility, confidence`).
Identities, event type, Δt, team relation and quality flags are only in the separate `root.references.jsonl`. The instance
layout follows the earlier Task B schema so the historical feature code (`validate.features_only`) can be applied unchanged.

**Exclusion reasons** (recorded per event in `root.exclusions.jsonl`): `period_boundary`, `segment_span_too_short` (< 1 s on either
side, i.e. |Δt| > 3 s with the ±2.5 s span), `attribution_unresolved_or_ambiguous`, `duplicate_position_artifact`,
`substitution_window`, `low_coverage` (< 80 % of frames), `internal_gap` (> 0.5 s), `too_few_samples` (< 10).

Example instance (abridged) and its reference:

```json
{"instance_id": "00005fe49446cd68", "schema": "m11-intersections-v1", "task": "intersection_four_segment", "query": "in-0",
 "units": {"t": "s relative to the start of the withheld interval", "x": "m", "y": "m"}, "ambiguity_interval_s": [0.0, 0.5522],
 "incoming": [{"tracklet_id": "in-0", "t": [-2.2123, …, -0.0101] (67 frames), "x": […], "y": […], "visibility": […], "confidence": […]},
              {"tracklet_id": "in-1", "t": [-2.2123, …, -0.0101], …}],
 "outgoing": [{"tracklet_id": "out-0", "t": [0.5571, …, 2.7593], …}, {"tracklet_id": "out-1", "t": [0.5571, …, 2.7593], …}], "during": {"t": [], "points": []}}
```
```json
{"instance_id": "00005fe49446cd68", "event_id": "59bef255784f44ed", "target": 0, "expected": {"in-0": "out-0", "in-1": "out-1"},
 "incoming": {"in-0": {"side": "away", "jersey": 8, "slot": 11, …}, "in-1": {"side": "home", "jersey": 4, "slot": 1, …}},
 "event_type": "proper_crossing", "abs_time_diff_s": 0.552, "team_relation": "opposite_teams", "quality": {…}, …}
```

## 4. Data finding: duplicate-position artifact

In 3812, **7 fixed slot pairs share bit-identical x/y** (68 frames each in period 1, 3,269–3,840 frames each in period 2; e.g.
away #5/#3, home #4/#3, home #6/#8; 38,391 samples). Coincident paths produced ~44k spurious intersections in an earlier dry run.
Exact floating-point equality of two players' positions is physically impossible, so it is treated as a provider artifact by an
exact-equality validity rule (not a proximity criterion): such samples are not joined into segments and samples touching them
are excluded. Figure `root.fig-duplicate_position_artifact.png`. Whether this occurs in other matches is unknown.

## 5. Results, match 3812 (development)

| | |
|---|---|
| windows scanned | 6,159 (11–23 available players, median 22) |
| pair-window comparisons | 1,404,228 (1,323,785 rejected by bounding box) |
| segment-pair tests | 419.7 M |
| pair-windows with ≥ 1 intersection | 26,662 |
| segment-pair hits across windows → unique hits → **unique events** | 52,155 → 18,643 → **18,469** (174 vertex/overlap hits merged) |
| event types | proper crossing 18,455 · vertex crossing 8 · touch 6 · collinear 0 |
| by period / team relation | P1 8,788 · P2 9,681 / opposite 11,377 · same team 7,092 |
| windows per event | 1: 3,913 · 2: 3,958 · 3: 4,361 · 4: 4,411 · 5: 1,826 |
| polyline steps | 2,035,107; kept 1,999,746; not joined: invalid 22, jump 14,658, attribution 1,504, duplicate 19,177 |
| **valid four-segment samples** | **12,754** (target out-0 6,406 / out-1 6,348) |
| excluded events | 5,715: span too short (|Δt| > 3 s) 5,657 · duplicate artifact 24 · attribution 20 · period boundary 7 · substitution 7 |
| runtime / memory | load 1.7 s · detection 32 s · samples 14 s (≈ 65 s with figures and writing) · peak ≈ 1.0 GB |
| outputs | 111 MB (instances 76 MB, events 21 MB, references 15 MB) |

**|Δt| distribution** (events): min 0 · p10 0.44 · p25 1.06 · median 2.10 · p75 3.25 · p90 3.96 · max 4.91 s; simultaneous
(≤ 0.2 s) 797. Bins: 0–0.2 s 797 · 0.2–0.5 1,306 · 0.5–1 2,240 · 1–2 4,454 · 2–3 4,015 · 3–4 3,973 · 4–5 1,684. Samples (|Δt| ≤ 3 s by
construction): 797 / 1,295 / 2,225 / 4,429 / 4,008. Most "crossings" are paths crossing at different times, not simultaneous
encounters.

**Pair clusters:** 6,655 clusters, 3,381 with a single event, p90 size 5, max 179. 47 % of events are in clusters of ≥ 5 and 16 %
in clusters of ≥ 20: slow or standing players whose jittery RAW tracks (largely ESTIMATED; example
`root.fig-largest_pair_cluster.png`) cross each other repeatedly. In samples, the mean ESTIMATED share has median 0.50; 32 % of
samples are all-ESTIMATED and 15 % all-VISIBLE; 4,144 samples have a Step 1 jump flag within ±2.5 s.

**Leakage checks** (summary `samples.diagnostics`): the two outgoing tracklets differ in length or start time in 0 % of
samples, the two incoming ones in 0 %; instance keys equal the whitelist; target share 0.498.

**Rule-based difficulty indicators** (no fitting, not models, development data): agreement with the provider label of
*nearest position* (closest outgoing start to A's last point) 0.471 — below chance, because at `t_hi` the *later* crosser sits at
the intersection where the earlier crosser was at `t_lo`; *constant velocity* (last 0.5 s extrapolated) 0.753 overall, 0.690 for
simultaneous crossings, 0.73–0.79 in the other |Δt| bins.

## 6. Figures (`data/processed/m11-intersections-v1/3812/`)

Chosen by fixed rules (first event in time satisfying each), never by any model output: `simultaneous_crossing`,
`crossing_1_to_2s_apart`, `crossing_over_3s_apart_excluded`, `vertex_crossing`, `touch`, `largest_pair_cluster`, plus
`overview` (Δt histogram, intersection map, cluster sizes) and `duplicate_position_artifact`. Each event figure states match,
period, event ID, video times and frame ranges of both segments, Δt, windows, cluster size and the sample ID (identities shown:
audit view).

## 7. Tests (`tests/test_intersections.py`, 25 tests)

Geometry: proper crossing, non-crossing (segment / parallel / collinear disjoint), endpoint touch and T-junction, collinear
overlap and single shared point, degenerate (repeated coordinate) and tolerance, vectorised = pairwise on random segments,
vertex side classification. Detection on synthetic RAW caches through the real Step 1 quality pass: one crossing found in 4
windows → 1 event with exact point, times and Δt; non-crossing; missing-data gap of 1.1 s not joined (0.5 s gap joined);
two separate crossings of the same segment not merged (same cluster); vertex crossing (two endpoint hits → one event); touch;
duplicate positions excluded; deterministic IDs. Samples: schema whitelist, label = query's own continuation (checked against
the cache) over 24 seeds with both targets and both query choices occurring, equal time stamps, no identity / event tokens in
instances, balanced label bits (4,000 keys), exclusions (|Δt| 3.5 s, period boundary, outgoing coverage, unresolved
attribution), opaque-ID ordering. Outputs: no overwrite and no partial directory; CLI refuses validation and test matches;
written 3812 outputs match their manifest checksums.

## 8. Limitations

- Geometric intersection ≠ visual occlusion ≠ identity swap; the label is provider continuity, which can itself be wrong.
- RAW jitter of slow players produces many repeated crossings; a large share of events are tracking-noise crossings rather than
  football-relevant passes. Nothing filters them (by design, per the advisor's specification); cluster ID and ESTIMATED share
  are recorded so this can be stratified or grouped.
- The withheld interval is `[t_lo, t_hi]`; for near-simultaneous crossings it is almost empty, so A's last incoming and first
  outgoing points are only a few frames apart. The advisor may prefer a minimum withheld interval (`withheld_pad_s`).
- One development match only. Duplicate artifact, event counts and runtimes may differ elsewhere.
- Samples of one pair cluster are strongly dependent; any training/evaluation must group by match (and should report by
  cluster) to avoid optimistic estimates.
- Instance JSON is large (≈ 6 kB per sample); compression or a columnar format would help at scale.
