# Handoff: Session B detection-miss investigation, T0L and Player Detection & Identity Debugger (2026-11-21)

**Status: done, awaiting review. Nothing is committed (Git is read-only for Claude).**

Details:
- [Failure decomposition report](../research/SESSION_B_PIPELINE_FAILURE_DECOMPOSITION_2026-11-21.md)
- [Multi-YOLO + GTA plan](../research/SESSION_B_MULTI_YOLO_AND_GTA_PLAN_2026-11-21.md)

Constraints respected:
- dev matches only (117093, 118575, 117092, 118576); reserved matches are refused by the server and the CLI;
- CPU only; no downloads, installs or training;
- Session A untouched (a test checks that its `vision/` and `models/` are clean);
- production config unchanged;
- new write-once outputs only.

## Launch

```powershell
cd F:\Football-Fatigue-Video
$env:PYTHONPATH="F:\Football-Fatigue-Video\src;F:\Football-Fatigue-Project\src"
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.player --data-root F:/Football-Fatigue-Project/data --port 5831
```

Open http://127.0.0.1:5831/?source=soccertrack-117093&ti=T0&debug=1&gid=4&frame=31 and press **🔍 Debugger** in the header (or keep `debug=1` in the URL).

The Streamlit dashboard (`python -m football_fatigue.video.dashboard --data-root F:/Football-Fatigue-Project/data`, port 5830) embeds the same player. Its "Detector & Tracker Benchmark" tab now also shows the decomposition and the B0 / T0 / T0L engine tables.

### What the Debugger shows

- **Three kinds of ID, never mixed.**
  - `G##` is the SoccerTrack GT player (annotation).
  - `T##` is the temporary ByteTrack ID of the selected profile. It is not an identity.
  - Detector boxes carry only a confidence.
- **GT boxes are coloured by what happened to that player in the selected profile:** tracked / ID changed / passed but not tracked / below conf / turf-filtered / non-person class / no box after merge / not evaluable.
- **Pipeline at this frame:** boxes per stage. The stages are raw cache (conf > 0.1, post-merge), person, conf > 0.4, turf filter, tracker output, and GT. Below them are the outcome counts. The detector dropdown shows each stage's boxes (`raw`, `T0L`, `T0`, `B0` input).
- **Failure inspector.** Filters by group (E no box / B rejected / C untracked / D ID change / reappearance / leaves view), by minimum length and by G id. Clicking an item:
  - jumps to its first frame and switches the view to "Follow selected GT player";
  - shows match, profile, missing interval, the last tracked frame with its T id and conf, the first reappearance with its T id and conf, same/changed ID, cause and evidence, size and visibility, the nearest GT player, and miss hints labelled as hints;
  - offers buttons for **before / during / after / ▶ play through** (loops from 1 s before to 1 s after) and a before/during/after strip image.
- **Timeline lanes** for the engine events plus per-player rows: no box, rejected, untracked, ID change, reappears, leaves view. Overlapping items are packed into separate lanes and none are drawn over each other. Selecting a G id keeps only that player's rows, and multiple episodes per player stay separate. Clicking any bar jumps to the exact frame.
- **A | B side by side** with any two tracker inputs (B0, T0, T0L, T0-BOT, T0-OCS), synchronized. Each pane has its own outcomes.
- **Detector comparison:** choose a benchmark run, for example YOLOv8m. Its boxes are drawn violet and dashed, only on frames where both that run and production exist. Other frames show a message instead of empty boxes.
- **Gallery** (gallery-v2): 11 categories, 2 seeded items each, drawn from all qualifying cases. Each item opens its match, frame, GT player and profile in the player.

## Three recommended failure examples

| # | open | before | during | after | what to see |
|---|---|---|---|---|---|
| 1 | `?source=soccertrack-117093&ti=T0&debug=1&gid=4&frame=31` | f30: G4 tracked as T3 (conf 0.55) | f31–37: no box after merge; G4 and G9 overlap heavily (`merge_suspect` hint) | f38: G4 reappears as **T5**, G9 as **T3** | a GT-verified swap caused by a 7-frame detector gap in an overlap. It is the archetypal identity failure. |
| 2 | `?source=soccertrack-118575&ti=B0&cmp=T0&debug=1&gid=2&frame=2463` | f2463: G2 tracked as T127 | f2464–2470: box conf 0.82–0.88 **removed by the turf filter** in B0; in T0 (bottom pane) G2 stays tracked as T56 every frame | f2471: back | a correct, confident detection deleted by post-processing, not by YOLO |
| 3 | `?source=soccertrack-118576&ti=T0&debug=1&gid=14&frame=5609` | f5609: G14 = T91 (conf 0.76) | f5614–5617: not tracked for 4 frames (box conf 0.34 at f5617, below 0.4) | f5618: G14 becomes **T23** while T91 continues on another box (GT-verified) | a short confidence dip during an overlap produces an ID swap |

To use the examples: select the item (failure list, or the gallery). Then press **before**, **during** and **after**, or **▶ play through**. Use ←/→ for single frames and Shift+arrow for ±25 frames.

## Measured answers (details in the decomposition report)

Shares are of evaluable GT-player observations; all four dev matches pooled.

| finding | value |
|---|---|
| no post-merge box (YOLO26 + merge, not separable) | **6.37 %** (tune 3.75, check 9.00; low-light 117092 alone 14.1 %) |
| correct detection deleted by the turf filter (B0) | **5.07 %** (tune 7.74) |
| correct detection below conf 0.4 | 3.00 % |
| ByteTrack received a valid box but output no track (T0) | 0.80 % (74 % of those have conf ≤ 0.5, which cannot start tracks) |
| temporary-ID changes (T0) | 1,890; 72 % after a gap; 733 GT-verified swaps |

T0 vs T0L:
- Frag −33 % (tune) / −37 % (check); detection-loss events about −50 %.
- HOTA −0.003 / +0.005; PO F1 +0.007 / 0.000.
- False PO alerts +7 / +14; missed occlusion episodes −9 / −8.
- **T0L is not promoted.**

## Answers to the 10 questions

1. **Main causes of disappearance:**
   - In B0, the turf filter (33 % of lost observations) and no box after merge (42 %).
   - In T0, no box after merge (63 %), below conf 0.4 (30 %) and ByteTrack not tracking a passed box (8 %).
   - Misses concentrate in GT-heavy overlap (43 % miss rate), hidden players (78 %), players under 35 px (21 %) and low light (117092).
2. **How often YOLO26 genuinely missed:** at most **6.37 %** of observations had no post-merge box. How much of that is the model and how much is merge-NMS is **unknown** until the approved raw capture. 35 % of misses carry the `merge_suspect` hint, and 51 % have no geometry hint (mostly low light).
3. **How often a filter rejected a correct detection:** turf filter 5.07 % (B0 only; T0 removes it) and conf threshold 3.00 %. No class rejections were observed.
4. **How often ByteTrack failed despite a valid detection:** 0.80 % untracked plus 0.36 % ID changed (T0). Most identity damage follows *detector* gaps (72 % of ID changes are after a gap).
5. **Does T0L improve the engine?** It cleans the timeline (Frag −35 %, about half the detection-loss events) but does not improve occlusion detection or identity. It mainly changes event counts. Not promoted.
6. **Best current profile:** **T0** (YOLO26-L cache, turf filter off, conf > 0.4, sv-bytetrack buffer 75). It has the best tune HOTA (0.416 vs B0 0.371 and T0L 0.414) and is +0.018 over B0 on check, with the same detections.
7. **Can the user inspect players visually?** Yes. The Debugger was browser-verified: 12 dedicated checks passed (failure selection, follow-GT, strip, timeline lanes and real mouse click, play-through, T0 | T0L, YOLOv8m overlay, gallery, reserved refusal).
8. **What fair YOLO testing requires:** same tiling, merge and frames; a stratified larger dev sample; the architecture-matched yolo26l-coco control; detector and downstream errors reported separately; then tracking only for a shortlist. This needs approvals A1, A2 and optionally A3/A4.
9. **What remains before identity recovery:**
   - the model-vs-NMS split (raw capture);
   - a fix for re-association after short gaps in overlap (GTA benchmark, A5);
   - human-confirmed occlusion labels (currently GT-derived only);
   - referee labels.
10. **Next to implement (after approval):**
    1. The 16-min raw per-tile capture (A2-cap) to split misses into model vs merge.
    2. The GTA benchmark on T0 tracks (A5).
    3. In parallel on CPU: ByteTrack activation 0.3 together with T0L as one isolated change.

## Approvals requested

| id | request |
|---|---|
| A2-cap | GPU raw capture, about 16 min |
| A1 | COCO weights |
| A2 | GPU detector runs |
| A3 | RF-DETR |
| A4 | Hugging Face football checkpoints |
| A5 | GTA (repo, separate venv, torchreid, OSNet sports checkpoint) |
| A6 | Roboflow `trackers` |

Exact scope is in the plan document §4.

## Unknowns

- Model miss vs merge-NMS for the 33,579 "no box" observations.
- Real (human-confirmed) occlusion vs GT-derived overlap.
- Referee behaviour.
- VRAM and speed of every non-production detector.
- Whether the OSNet sports checkpoint is publicly downloadable.

## Code changes (all uncommitted)

| area | change |
|---|---|
| `video/detection/replay.py` | added the `T0L` input |
| `video/tracking.py` | T0L profile; the manifest records the profile's own conf threshold plus the production one (fix: the engine refused T0L) |
| new `video/failures/` | `pipeline.py` (decompose), `aggregate.py`, `engine_compare.py`, `heavy.py`, `gallery.py`, `__main__.py` |
| new `video/player/debug.py` | debugger back end |
| `player/data.py` | per-frame outcomes `o` in chunks |
| `player/server.py` | `/api/diag`, `/api/strip`, `/api/bench`, `/api/gallery`, `/gallery/<file>` |
| `player/player.html` | Debugger UI |
| `player/qa.py` | `debugger_check`; the Streamlit walkthrough turns on the benchmark tab |
| `dashboard/bench_tab.py` | decomposition and engine tables |
| tests | new `tests/test_video_failures.py` (17 tests); `tests/test_video_player.py` updated for the `o` layer |

Commands:

```bash
python -m football_fatigue.video.failures decompose|aggregate|engine|heavy|gallery --data-root F:/Football-Fatigue-Project/data
```

## Checks that ran

- **Focused Session B tests:** `tests/test_video_{failures,bench,player,step2,detection,occlusion}.py` → **111 passed**.
- **Browser QA** (`python -m football_fatigue.video.player.qa --streamlit-url http://127.0.0.1:5830`, report `_qa/player-qa-v1-20261121-035909`): **73 passed, 1 failed**.
  - The failure was the new Streamlit benchmark-tab check. It read the page before the first-time load of the benchmark ground truth (about 1 min) finished.
  - The check now waits for it. The re-run of the Streamlit walkthrough alone (`_qa/streamlit-bench-recheck`) gave **19/19 passed**, with no exceptions in any tab.
- **Full suite:** see the section below.

### Full suite

`pytest` (`FOOTBALL_FATIGUE_DATA_ROOT` set): **1081 passed, 2 failed, 63 skipped** (3 min 31 s). Both failures are environmental and are the same two as on 2026-11-20:
- `test_proximity_audit::test_validation_and_test_matches_refused`
- `test_proximity_context::test_v3_refuses_validation_and_test_matches`

Both are Session A tests that need the repo-relative `data/processed/m11-encounters-v1/splits.json`, which is absent in this worktree.
