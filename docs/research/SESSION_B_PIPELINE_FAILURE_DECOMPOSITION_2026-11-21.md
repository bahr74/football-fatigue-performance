# Session B: why players disappear (pipeline-v1, engine-t0l-v1, heavy-v1), 2026-11-21

**Scope.** Development matches only:
- tune: 117093, 118575
- check: 117092 (low light), 118576

Inputs are the existing YOLO26-L detection cache, the existing tracks and the existing engine v2 runs, plus the new T0L tracks and engine runs (CPU only). No GPU, no downloads, no training, no Session A changes. All outputs are new and write-once under `data/processed/video/_diag/`.

The ground truth is SoccerTrack: 22 annotated players per match, with no referees. Every rate below is a share of **evaluable GT player observations**, one per GT player per frame. An observation is not evaluable when the box is off the pitch or on the frame border.

## 1. Method: one outcome per GT player per frame (`python -m football_fatigue.video.failures decompose`)

Each GT box is matched to the cached boxes by Hungarian assignment at IoU ≥ 0.5. Its fate is then followed in pipeline order through each tracker-input profile:

| profile | tracker input |
|---|---|
| B0 | person class, conf > 0.4, turf filter on (production replay) |
| T0 | person class, conf > 0.4, turf filter off |
| T0L | person class, conf > 0.1, turf filter off |

| code | group | meaning | evidence |
|---|---|---|---|
| ok | A | detected, tracked, same temporary ID as last time | cache + tracks |
| rej_class / rej_conf / rej_turf | B | a matching box exists but the profile's filter removed it before ByteTrack | cache (conf, class, turf flag) |
| untracked | C | the box reached ByteTrack, no track output on it (track box IoU > 0.99) | cache + tracks |
| id_changed | D | tracked, but under a different temporary ID than this GT player's previous one | tracks |
| miss | E | **no post-merge box at IoU ≥ 0.5** | cache |
| not_evaluable | F | off pitch / frame border | GT geometry |

**Not measurable: "miss" cannot be split into model miss vs. merge-NMS deletion.** The cache stores boxes *after* the tile merge (merge NMS IoU 0.5, conf ≥ 0.1), so the decomposition never claims NMS. Each miss gets a geometry *hint*:
- `merge_suspect`: another GT player overlaps and a cached box covers one of the two.
- `poor_localisation`: a cached box with IoU 0.1–0.5 exists.
- `none`: neither.

Hints are not causes. §4 gives the capture that would measure the split.

## 2. Results

### 2.1 Observations (pooled, `_diag/pipeline-v1/aggregate.json`)

| profile | role | evaluable | ok | B turf | B conf | C untracked | D ID changed | E no box |
|---|---|---|---|---|---|---|---|---|
| B0 | tune | 263,420 | 85.73 % | **7.74 %** | 1.82 % | 0.60 % | 0.36 % | 3.75 % |
| B0 | check | 263,340 | 82.88 % | 2.41 % | 4.18 % | 1.10 % | 0.42 % | 9.00 % |
| T0 | tune | 263,420 | 93.60 % | 0 | 1.82 % | 0.52 % | 0.31 % | 3.75 % |
| T0 | check | 263,340 | 85.33 % | 0 | 4.18 % | 1.08 % | 0.41 % | 9.00 % |
| T0L | tune | 263,420 | 94.80 % | 0 | 0 | 1.14 % | 0.31 % | 3.75 % |
| T0L | check | 263,340 | 87.60 % | 0 | 0 | 3.00 % | 0.40 % | 9.00 % |

Lost share per match, in the order 117093 / 118575 / 117092 / 118576:

| profile | 117093 | 118575 | 117092 | 118576 |
|---|---|---|---|---|
| B0 | 11.2 % | 16.7 % | 23.6 % | 9.8 % |
| T0 | 5.4 % | 6.8 % | 22.9 % | 5.6 % |
| T0L | 4.6 % | 5.2 % | 19.1 % | 4.9 % |

The detector stage is identical for every profile, and a test checks this. "No box" observations per match are 4,921 / 4,955 / **18,649** / 5,054. The low-light match 117092 alone holds 56 % of all misses.

**Composition of lost observations, all four matches:**

| profile | turf | no box | conf | untracked |
|---|---|---|---|---|
| B0 | 33.2 % | 41.7 % | 19.6 % | 5.5 % |
| T0 | – | 62.6 % | 29.5 % | 7.9 % |
| T0L | – | 75.5 % | – | 24.5 % |

On the tune matches, the turf filter alone causes 55.6 % of B0's losses.

**Episodes** are contiguous lost runs for one GT player. T0 has 8,610 episodes in total:
- E (no box): 4,252
- B (conf): 4,088
- C (untracked): 270

On 117093 the durations are mostly short: 1 frame 231, 2–5 frames 342, 6–25 frames 185, 26–75 frames 41, 76–250 frames 4 (E group). After an episode, 867 players reappear with the same temporary ID and 323 with a new one.

**Per-frame coverage, T0:** the mean number of missing GT players per frame is 1.18 / 1.50 / 5.04 / 1.23.

### 2.2 Where the misses happen (T0, all matches)

- **By GT-derived visibility.** Visibility is measured by cover from other GT boxes. These are not human-confirmed occlusions.

  | visibility | miss rate |
  |---|---|
  | visible | 4.0 % |
  | partial | 10.1 % |
  | heavy | 43.2 % (plus rej_conf 10.7 %, id_changed 2.8 %) |
  | hidden | 78.3 % |

- **By GT box height:**

  | height | miss rate |
  |---|---|
  | < 35 px | 21.1 % (plus rej_conf 9.3 %) |
  | 35–50 px | 5.2 % |
  | 50–80 px | 0.9 % |

- **Miss hints, pooled (33,579 misses):**
  - `none`: 16,959 (50.5 %; 14,006 of them on 117092, consistent with low light and small players)
  - `merge_suspect`: 11,662 (34.7 %)
  - `poor_localisation`: 4,958 (14.8 %)
- **Median confidence of the matched box, by code (heavy-v1):**

  | code | median conf |
  |---|---|
  | ok | 0.85–0.87 |
  | id_changed | 0.69–0.74 |
  | untracked | 0.46–0.47 |
  | rej_conf | 0.24–0.26 |

- **Overlapping pairs (heavy-v1).** Overlapping GT pairs at IoU ≥ 0.5 are frequent. Overlapping pairs of cached person boxes at IoU ≥ 0.5 are almost absent:

  | match | GT pairs | cached pairs |
  |---|---|---|
  | 117093 | 1,948 | 18 |
  | 118575 | 1,321 | 44 |
  | 117092 | 695 | 252 |
  | 118576 | 1,934 | 22 |

  This is *consistent with* the IoU 0.5 merge (or the model's own NMS) collapsing two overlapping players into one box. It is not proof, because pre-merge boxes were not saved.

### 2.3 Tracker failures despite a valid detection (T0)

- **untracked: 4,209 observations (0.80 %).** 74 % of these have conf ≤ 0.5. ByteTrack (activation 0.5) only uses such boxes in its second, low-score association and never starts a track from them. Only 1,077 untracked observations have conf > 0.5.
- **Temporary-ID changes: 1,890 observations; 455 / 360 / 549 / 526 events per match.**
  - 132 / 102 / 113 / 174 are *direct* changes, with no gap.
  - 323 / 258 / 436 / 352 happen *after a gap*. Most of those gaps are "no box" episodes, i.e. the tracker lost the player during a detector gap and re-acquired a new ID.
  - In 209 / 144 / 164 / 216 changes the previous ID **continued on another box**. These are GT-verified swaps or theft, the most damaging case for identity.

### 2.4 Shares, answered

| question (T0 = recommended profile, all 4 matches) | share of evaluable GT obs. |
|---|---|
| no post-merge box (YOLO26 + merge combined) | **6.37 %** (tune 3.75, check 9.00; 117092 14.1 %) |
| correct detection removed by the turf filter (B0 only) | **5.07 %** (tune 7.74, check 2.41) |
| correct detection below conf 0.4 | 3.00 % |
| ByteTrack had a valid detection but output no track | 0.80 % |
| tracked under a changed temporary ID | 0.36 % (1,890 obs., 1,890 events) |

Detector stage (E) vs downstream (B + C) shares of T0 lost observations: **62.6 % vs 37.4 %**. Counting B0's turf filter as downstream: 41.7 % vs 58.3 %.

## 3. T0 vs T0L (`_diag/engine-t0l-v1`, same detections, same ByteTrack, same engine v2 / ff8815ed)

| role | profile | HOTA | DetA | AssA | IDF1 | IDSW | Frag | PO P | PO R | PO F1 | false PO alerts | missed occl. episodes | events (all classes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune | B0 | 0.371 | 0.636 | 0.217 | 0.361 | 756 | 5,948 | 0.705 | 0.663 | 0.683 | 194 | 264 | 6,018 |
| tune | T0 | **0.416** | 0.680 | 0.256 | 0.413 | 614 | 2,373 | 0.709 | 0.678 | 0.693 | 196 | 255 | 3,183 |
| tune | T0L | 0.414 | 0.675 | 0.254 | 0.414 | 608 | **1,601** | 0.705 | 0.696 | **0.700** | 203 | 246 | 2,323 |
| check | B0 | 0.345 | 0.649 | 0.184 | 0.326 | 896 | 6,895 | 0.656 | 0.702 | 0.678 | 251 | 240 | 5,921 |
| check | T0 | 0.363 | 0.655 | 0.204 | 0.363 | 859 | 5,145 | 0.659 | 0.704 | 0.680 | 248 | 238 | 4,700 |
| check | T0L | **0.368** | 0.661 | 0.207 | 0.364 | 840 | **3,232** | 0.650 | 0.715 | 0.680 | 262 | 230 | 3,352 |

Metric notes:
- HOTA is the mean over matches.
- PO P/R/F1 are means over matches, measured against GT-derived occlusion episodes (eval-v1). These are not human-confirmed occlusions.
- Counts are pooled.

PO start/end timing is unchanged: median absolute error 4–7 frames in both profiles.

**What changes:**
- **Fragmentation drops 33 % (tune) and 37 % (check)**, and "detection_loss" events fall by about half:

  | role | T0 | T0L |
  |---|---|---|
  | tune | 1,669 | 786 |
  | check | 3,069 | 1,719 |

  Most of the removed events were tracker gaps on a player whose box was below 0.4.
- **Probable-occlusion output barely moves.** Of the PO events that map to GT players, about 90 % keep the same class (T0 → T0L: 454 / 285 / 376 / 414 unchanged, and only 17–40 per match appear or disappear).
  - PO recall rises by 1–2 points: 9 fewer missed occlusion episodes on tune, 8 fewer on check.
  - PO precision falls by 0.4–0.9 points: 7 more false alerts on tune, 14 more on check.
  - PO F1: tune +0.007, check ±0.000.
- **HOTA:** tune −0.003, check +0.005, i.e. ±0.005. IDSW falls by 6 (tune) and 19 (check).
- **Pipeline view.** The 15,826 conf-rejected observations become:
  - about 9,100 tracked;
  - about 6,700 "untracked". 91 % of T0L's untracked observations have conf ≤ 0.5, because ByteTrack cannot start tracks from them.

**Verdict.** T0L reduces fragmentation and halves spurious detection-loss events, which makes the engine's timeline much cleaner. It does **not** measurably improve occlusion detection (PO F1 +0.007 / 0.000) or identity (HOTA ±0.005, IDF1 +0.001). It mostly changes event counts. **T0 remains the recommended profile, and T0L is not promoted.** T0L remains a useful option for a less fragmented review timeline. If wanted, the next CPU experiment is to lower ByteTrack's activation threshold to 0.3 *together with* T0L, isolated as one change.

A note on correctness: the first T0L engine run was refused ("3862 tracked boxes are not in the detection cache"). The T0L manifest had copied the production conf threshold 0.4, which the engine uses to verify boxes. `tracking.build` now records the profile's own threshold, plus `production_confidence_threshold`. The T0 manifest still says 0.4, and a test checks this.

## 4. NMS / heavy overlap: what cannot be measured yet, and the bounded capture

**Unknown:** for each of the 11,662 `merge_suspect` misses, whether YOLO26-L proposed the hidden player at all. It cannot be answered from the cache.

**Capture spec** (`_diag/heavy-v1.json → capture_plan`; tool already exists and refuses to run without `--approved`):

| item | value |
|---|---|
| tool | `python -m football_fatigue.video.detection capture --approved --data-root F:/Football-Fatigue-Project/data --source <sid> --start <s> --frames 125` |
| settings | production settings copied from each cache manifest: YOLO26-L, 1280×1280 tiles, 20 % overlap, imgsz 1920, batch 2 |
| what it stores | every per-tile box **before** the merge (conf ≥ 0.05), with tile and turf decision |
| windows (most merge_suspect GT-frames) | 117093 @1404, 118575 @4745, 117092 @5875, 118576 @4980; 125 frames each |
| cost | about 4 min per window on the GTX 1650 Ti (1.77 s/frame measured), < 3 GB VRAM, about 16 min total |
| answers | share of misses with a raw box removed by the merge (NMS loss) vs no raw box (model miss); offline replay of merge IoU 0.5 / 0.6 / 0.7 on the same raw boxes (no new inference); NMS parameters stay unchanged in production |

## 5. Limitations

- GT has players only. Referee boxes count as "not GT" everywhere, and engine events on referees show up as `non_gt_person`.
- "heavy/hidden" visibility is derived from GT-GT box cover. It is not a human-confirmed occlusion.
- Hungarian matching at IoU ≥ 0.5 can give a shared box to the neighbour, so some "no box" cases have a box that covers two players. This is exactly the `merge_suspect` hint.
- There are only four matches, from one camera system. 117092 (low light) dominates the check role.
- Tracker IDs (T##) are temporary ByteTrack IDs. "ID changed" means a change of temporary ID verified against GT (G##), not a player-identity claim.

Outputs:
- `_diag/pipeline-v1/` (per match and profile: `frames.jsonl`, `episodes.json`, `summary.json`; plus `aggregate.json` and `report.md`)
- `_diag/engine-t0l-v1/`
- `_diag/heavy-v1.json`
- `_diag/gallery-v2/` (22 strips, 11 categories, seed 11)
