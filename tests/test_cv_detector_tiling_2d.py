"""2D tiling, tile micro-batching and the person-agnostic merge policy, with a stub model (CPU only).

The stub detects, in every tile it is given, the visible part of each ground-truth box (clipped to the
tile, with lower confidence when truncated), so the tests exercise the real tile geometry, global
coordinate mapping, batching and merging code without model weights.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from football_fatigue.vision.detection_cache import DetectionCacheReader, DetectionCacheWriter  # noqa: E402
from football_fatigue.vision.detector import (  # noqa: E402
    EDGE_CONFIDENCE_MARGIN,
    EDGE_PX,
    YOLOPlayerDetector,
)

NAMES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper", 7: "chair"}
BALL, PLAYER, REF, GK, CHAIR = 0, 1, 2, 3, 7


def encoded_frame(width, height):
    """RGB frame whose first pixel of any crop reveals the crop's origin (x < 4096, y < 4096)."""
    ys, xs = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = xs & 255
    frame[..., 1] = (xs >> 8) | ((ys >> 8) << 4)
    frame[..., 2] = ys & 255
    return frame


class Tensor(list):
    def tolist(self):
        return list(self)


class StubModel:
    """Callable like an Ultralytics model: list of BGR crops in, per-crop results out."""

    names = NAMES

    def __init__(self, truth, min_visible=0.35):
        self.truth = truth  # [(class_id, confidence, (x1, y1, x2, y2))] in global coordinates
        self.min_visible = min_visible
        self.batches = []

    def __call__(self, crops, **kwargs):
        crops = crops if isinstance(crops, list) else [crops]
        self.batches.append(len(crops))
        out = []
        for crop in crops:
            y0 = int(crop[0, 0, 0]) | ((int(crop[0, 0, 1]) >> 4) << 8)
            x0 = int(crop[0, 0, 2]) | ((int(crop[0, 0, 1]) & 15) << 8)
            h, w = crop.shape[:2]
            cls, conf, xyxy = Tensor(), Tensor(), Tensor()
            for c, p, (bx1, by1, bx2, by2) in self.truth:
                cx1, cy1, cx2, cy2 = max(bx1, x0), max(by1, y0), min(bx2, x0 + w), min(by2, y0 + h)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                visible = (cx2 - cx1) * (cy2 - cy1) / ((bx2 - bx1) * (by2 - by1))
                if visible < self.min_visible:
                    continue
                cls.append(float(c))
                conf.append(p if visible >= 0.999 else p * visible ** 0.5)
                xyxy.append([cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0])
            out.append(SimpleNamespace(boxes=SimpleNamespace(cls=cls, conf=conf, xyxy=xyxy, __len__=lambda: 0)
                                       if False else _Boxes(cls, conf, xyxy)))
        return out


class _Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, xyxy

    def __len__(self):
        return len(self.cls)


def detector(truth, **kwargs):
    kwargs.setdefault("tile_width", 400)
    det = YOLOPlayerDetector(confidence_threshold=0.05, image_size=640, **kwargs)
    det._model = StubModel(truth)
    return det


def infer(det, width=1000, height=700):
    return sorted(det._infer(encoded_frame(width, height)), key=lambda o: (o[2][0], o[2][1], o[0]))


def person(x, y, conf=0.9, cls=PLAYER, w=20, h=40):
    return (cls, conf, (float(x), float(y), float(x + w), float(y + h)))


# --- grid geometry ---------------------------------------------------------------

@pytest.mark.parametrize("width, height, tile_w, tile_h, overlap", [
    (1000, 700, 400, 300, 0.2), (1000, 700, 400, None, 0.2), (3840, 1906, 1280, 1280, 0.2),
    (3840, 1906, 960, 960, 0.25), (1000, 700, 5000, 5000, 0.2), (1000, 700, 333, 111 + 64, 0.0), (1001, 701, 400, 300, 0.5),
])
def test_grid_covers_every_pixel_without_gaps_or_duplicate_tiles(width, height, tile_w, tile_h, overlap):
    det = YOLOPlayerDetector(tile_width=tile_w, tile_height=tile_h, tile_overlap=overlap)
    tiles = det._tile_grid(width, height)
    assert len(tiles) == len(set(tiles))
    coverage = np.zeros((height, width), dtype=int)
    for x, y, w, h in tiles:
        assert 0 <= x and 0 <= y and x + w <= width and y + h <= height and w == tiles[0][2] and h == tiles[0][3]
        coverage[y:y + h, x:x + w] += 1
    assert coverage.min() >= 1
    xs, ys = sorted({t[0] for t in tiles}), sorted({t[1] for t in tiles})
    assert len(tiles) == len(xs) * len(ys) and tiles == [(x, y, tiles[0][2], tiles[0][3]) for y in ys for x in xs]  # row-major
    assert xs[0] == 0 and ys[0] == 0 and xs[-1] + tiles[0][2] == width and ys[-1] + tiles[0][3] == height
    for axis in (xs, ys):
        size = tiles[0][2] if axis is xs else tiles[0][3]
        assert all(b - a <= max(1, int(size * (1 - overlap))) for a, b in zip(axis, axis[1:]))  # neighbours overlap


def test_default_grid_is_the_historical_set_of_full_height_strips():
    def historical(width, tile_width, overlap):
        tile = min(tile_width, width)
        step = max(1, int(tile * (1 - overlap)))
        xs = list(range(0, max(1, width - tile + 1), step))
        if xs[-1] + tile < width:
            xs.append(width - tile)
        return xs

    for width, tile_w in ((3840, 1280), (4096, 1280), (1000, 400), (500, 800)):
        det = YOLOPlayerDetector(tile_width=tile_w)
        assert det._tile_origins(width) == historical(width, tile_w, 0.2)
        assert det._tile_grid(width, 1906) == [(x, 0, min(tile_w, width), 1906) for x in historical(width, tile_w, 0.2)]


# --- coordinates, batching ---------------------------------------------------------

def players_across_the_frame():
    return [person(x, y) for x in (3, 190, 380, 391, 590, 777, 975) for y in (4, 130, 290, 520, 655)]


@pytest.mark.parametrize("policy", ["class_aware", "person_agnostic", "edge_aware"])
def test_every_player_gets_its_global_coordinates_from_any_tile_position(policy):
    truth = players_across_the_frame()
    det = detector(truth, tile_height=300, tile_merge=policy)
    found = infer(det)
    full = [o for o in found if o[1] == pytest.approx(0.9)]
    assert sorted(tuple(o[2]) for o in full) == sorted(t[2] for t in truth)  # exact global boxes, one each
    if policy != "class_aware":
        assert len(found) == len(truth)  # truncated duplicates at tile edges are gone
    assert len(det._tile_grid(1000, 700)) > 6 and det._model.batches == [len(det._tile_grid(1000, 700))]


def test_class_aware_merge_keeps_truncated_edge_duplicates_that_the_new_policy_removes():
    truth = [person(390, 100)]  # straddles the cut at x=400... and lies fully inside the next overlapping tile
    default = infer(detector(truth, tile_height=None))
    agnostic_det = detector(truth, tile_height=None, tile_merge="person_agnostic")
    agnostic = infer(agnostic_det)
    assert len(default) == 2 and len(agnostic) == 1 and agnostic[0][2] == [390.0, 100.0, 410.0, 140.0]
    assert agnostic_det.merge_statistics["suppressed_edge_truncated"] == 1


@pytest.mark.parametrize("policy", ["class_aware", "person_agnostic", "edge_aware"])
def test_results_do_not_depend_on_the_micro_batch_size(policy):
    truth = players_across_the_frame() + [(GK, 0.7, (500.0, 300.0, 520.0, 340.0)), (BALL, 0.8, (395.0, 100.0, 405.0, 110.0))]
    reference = infer(detector(truth, tile_height=300, tile_merge=policy))
    total = len(YOLOPlayerDetector(tile_width=400, tile_height=300)._tile_grid(1000, 700))
    for batch in (1, 2, 3, 5, total, total + 4):
        det = detector(truth, tile_height=300, tile_merge=policy, tile_batch=batch)
        assert infer(det) == reference
        assert max(det._model.batches) <= batch and sum(det._model.batches) == total


# --- merge policy -----------------------------------------------------------------

def test_edge_truncation_is_handled_conservatively():
    tile_a, tile_b = (0, 0, 400, 700), (320, 0, 400, 700)
    frame = (1000, 700)
    truncated = (PLAYER, 0.62, [390.0, 100.0, 400.0, 140.0])
    complete = (PLAYER, 0.9, [390.0, 100.0, 410.0, 140.0])
    cut = YOLOPlayerDetector._truncated_by
    assert cut(truncated, tile_a, complete, tile_b, frame)
    assert not cut(complete, tile_b, truncated, tile_a, frame)                     # never the larger box
    assert not cut(truncated, tile_a, complete, tile_a, frame)                     # needs a different tile
    weak_big = (PLAYER, 0.62 - EDGE_CONFIDENCE_MARGIN - 0.01, [390.0, 100.0, 410.0, 140.0])
    assert not cut(truncated, tile_a, weak_big, tile_b, frame)                     # the bigger box must be credible
    same_side = (PLAYER, 0.9, [380.0, 100.0, 399.0, 140.0])
    assert not cut(truncated, tile_a, same_side, tile_b, frame)                    # bigger box must extend past the cut
    not_on_the_cut = (PLAYER, 0.62, [300.0, 100.0, 380.0 - EDGE_PX - 1, 140.0])
    assert not cut(not_on_the_cut, tile_a, (PLAYER, 0.9, [290.0, 90.0, 420.0, 150.0]), tile_b, frame)  # interior box, no cut
    frame_border = (PLAYER, 0.62, [0.0, 100.0, 12.0, 140.0])
    assert not cut(frame_border, (0, 0, 400, 700), (PLAYER, 0.9, [-5.0, 95.0, 20.0, 145.0]), tile_b, frame)  # frame edge is no cut
    neighbour = (PLAYER, 0.9, [410.0, 100.0, 430.0, 140.0])
    assert not cut(truncated, tile_a, neighbour, tile_b, frame)                    # a distinct neighbour is not a container
    # horizontal cuts work the same way: a box cut at the top of a lower tile, completed by a tile above it
    cut_top = (PLAYER, 0.6, [100.0, 300.0, 120.0, 326.0])
    assert cut(cut_top, (0, 300, 400, 400), (PLAYER, 0.9, [100.0, 290.0, 120.0, 326.0]), (0, 0, 400, 400), (1000, 700))
    assert not cut(cut_top, (0, 300, 400, 400), (PLAYER, 0.9, [100.0, 300.0, 120.0, 336.0]), (0, 0, 400, 400), (1000, 700))


def test_distinct_players_next_to_a_cut_are_both_kept():
    truth = [person(380, 100, w=18), person(399, 100, w=18), person(418, 100, w=18)]
    for policy in ("class_aware", "person_agnostic", "edge_aware"):
        found = infer(detector(truth, tile_height=None, tile_merge=policy))
        full = {tuple(o[2]) for o in found if o[1] == pytest.approx(0.9)}
        assert full == {t[2] for t in truth}


def test_cross_role_duplicates_collapse_only_when_boxes_nearly_coincide():
    twin = [person(100, 100, 0.9, PLAYER), person(100, 100, 0.6, GK)]               # same box, two role labels
    near = [person(100, 100, 0.9, PLAYER), person(102, 100, 0.6, GK)]               # IoU 0.82
    apart = [person(100, 100, 0.9, PLAYER), person(110, 100, 0.6, GK)]              # IoU 0.33: two players
    crowded = [person(100, 100, 0.9, PLAYER), person(103, 102, 0.6, REF)]           # IoU 0.68: ambiguous, keep both
    default = detector(twin, tile_height=None)
    assert len(infer(default)) == 2
    agnostic = detector(twin, tile_height=None, tile_merge="person_agnostic")
    kept = infer(agnostic)
    assert [(o[0], round(o[1], 2)) for o in kept] == [(PLAYER, 0.9)]                 # the more confident role is kept
    assert agnostic.merge_statistics["suppressed_cross_role:player>goalkeeper"] == 1
    assert len(infer(detector(near, tile_height=None, tile_merge="person_agnostic"))) == 1
    assert len(infer(detector(apart, tile_height=None, tile_merge="person_agnostic"))) == 2
    mid = detector(crowded, tile_height=None, tile_merge="person_agnostic")
    iou = YOLOPlayerDetector._box_iou(crowded[0][2], crowded[1][2])
    assert 0.5 <= iou < 0.7 and len(infer(mid)) == 2                                  # below cross_role_iou: both roles kept
    assert len(infer(detector(crowded, tile_height=None, tile_merge="person_agnostic", cross_role_iou=0.5))) == 1


def test_the_ball_and_unmapped_labels_never_merge_with_people():
    truth = [person(100, 100, 0.9), (BALL, 0.8, (100.0, 100.0, 120.0, 140.0)), (CHAIR, 0.7, (100.0, 100.0, 120.0, 140.0)),
             (BALL, 0.5, (393.0, 300.0, 403.0, 310.0))]  # a ball in the tile overlap, seen twice (once cut by the tile edge)
    det = detector(truth, tile_height=None, tile_merge="person_agnostic")
    found = infer(det)
    assert sorted(o[0] for o in found) == sorted([PLAYER, BALL, CHAIR, BALL])
    assert sum(1 for o in found if o[0] == BALL and o[2][0] > 300) == 1               # duplicated ball merged class-aware
    assert not [k for k in det.merge_statistics if k.startswith("suppressed_cross_role")]


def test_observe_reports_global_coordinates_and_filter_decisions():
    truth = [person(390, 100), (BALL, 0.8, (600.0, 200.0, 610.0, 210.0))]
    det = detector(truth, tile_height=300, tile_merge="person_agnostic")
    observations = det.observe(encoded_frame(1000, 700))
    assert {(o.class_label, o.xyxy) for o in observations} == {
        ("player", (390.0, 100.0, 410.0, 140.0)), ("ball", (600.0, 200.0, 610.0, 210.0))}
    assert all(o.passes_turf_filter is None for o in observations if o.class_label == "ball")


# --- defaults and validation --------------------------------------------------------

def test_default_settings_reproduce_the_historical_strip_pipeline():
    truth = players_across_the_frame() + [(GK, 0.7, (500.0, 300.0, 520.0, 340.0))]
    det = detector(truth)  # tile_width only, as in the B2 run
    new = infer(det)
    # historical algorithm, reimplemented: full-height strips in one call, class-aware NMS
    import torch
    from torchvision.ops import batched_nms

    reference_det = detector(truth)
    frame_bgr = np.ascontiguousarray(encoded_frame(1000, 700)[:, :, ::-1])
    offsets = reference_det._tile_origins(1000)
    results = reference_det._model([np.ascontiguousarray(frame_bgr[:, x:x + 400]) for x in offsets])
    items = [(int(c), float(p), [b[0] + x, b[1], b[2] + x, b[3]])
             for r, x in zip(results, offsets) for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    keep = batched_nms(torch.tensor([i[2] for i in items], dtype=torch.float32), torch.tensor([i[1] for i in items], dtype=torch.float32),
                       torch.tensor([i[0] for i in items]), 0.5).tolist()
    expected = sorted((items[k] for k in keep), key=lambda o: (o[2][0], o[2][1], o[0]))
    assert new == expected
    assert det._model.batches == [len(offsets)]  # one model call with every strip
    assert set(det.merge_statistics) <= {"suppressed_class_aware"}


def test_untiled_detection_still_uses_one_whole_frame_call():
    det = YOLOPlayerDetector(image_size=640)
    det._model = StubModel([person(10, 10)])
    assert len(det._infer(encoded_frame(300, 200))) == 1 and det._model.batches == [1]


@pytest.mark.parametrize("kwargs", [
    dict(tile_width=400, tile_height=32), dict(tile_width=400, tile_batch=0), dict(tile_width=400, tile_merge="newest"),
    dict(tile_height=300), dict(tile_batch=2), dict(tile_merge="person_agnostic"),
    dict(tile_width=400, cross_role_iou=0.4), dict(tile_width=400, cross_role_iou=1.2),
])
def test_invalid_tiling_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        YOLOPlayerDetector(**kwargs)


# --- provenance ---------------------------------------------------------------------

def test_describe_records_tiling_batching_merge_and_preprocessing():
    default = YOLOPlayerDetector(tile_width=1280).describe()
    assert default["tile_width"] == 1280 and default["tile_height"] is None and default["tile_batch"] is None
    assert default["tile_merge"] == {"policy": "class_aware", "merge_iou": 0.5} and default["merge_iou"] == 0.5
    assert default["preprocessing"]["tiles"] == "full-height strips" and default["confidence_floor"] == 0.4
    new = YOLOPlayerDetector(tile_width=1280, tile_height=1280, tile_batch=2, tile_merge="person_agnostic").describe()
    assert new["tile_height"] == 1280 and new["tile_batch"] == 2 and new["preprocessing"]["tiles"] == "2D grid"
    assert new["tile_merge"]["policy"] == "person_agnostic" and new["tile_merge"]["cross_role_iou"] == 0.7
    assert new["tile_merge"]["edge_px"] == EDGE_PX and "class-aware" in new["tile_merge"]["ball_and_unmapped_labels"]
    assert YOLOPlayerDetector().describe()["preprocessing"]["tiles"] == "whole frame"
    json.dumps(new, allow_nan=False)


def test_cache_manifest_carries_the_detector_settings_and_merge_statistics(tmp_path):
    truth = [person(390, 100), person(100, 100, 0.9), person(100, 100, 0.6, GK)]
    det = detector(truth, tile_height=300, tile_batch=2, tile_merge="person_agnostic")
    observations = det.observe(encoded_frame(1000, 700))
    writer = DetectionCacheWriter(tmp_path / "cache", {
        "video": {"sha256": "a" * 64, "fps": 25.0, "width": 1000, "height": 700},
        "detector": det.describe() | {"weights_sha256": "b" * 64},
        "tracking_input": {"confidence_threshold": 0.4}, "window": {"start_frame": 0},
    })
    writer.write_frame(0, 0.0, observations)
    writer.close(complete=True, detector_extra={"merge_statistics": dict(det.merge_statistics)})
    reader = DetectionCacheReader(tmp_path / "cache")
    saved = reader.manifest["detector"]
    assert saved["tile_height"] == 300 and saved["tile_batch"] == 2 and saved["tile_merge"]["policy"] == "person_agnostic"
    assert saved["merge_statistics"]["suppressed_edge_truncated"] >= 1
    assert saved["merge_statistics"]["suppressed_cross_role:player>goalkeeper"] == 1
    assert [o.xyxy for _, _, os_ in reader.iter_records() for o in os_ if o.confidence == pytest.approx(0.9)]  # round-trips


# --- edge-aware merge: truncation is resolved before same-role NMS -----------------------

UPPER, LOWER = (0, 0, 400, 400), (0, 300, 400, 400)   # the lower tile is cut at y=300 and the upper one extends to y=400
FRAME = (1000, 700)


def merge(policy, items, tiles, **kwargs):
    det = YOLOPlayerDetector(tile_width=400, tile_height=400, tile_merge=policy, **kwargs)
    det._model = StubModel([])
    return det._merge_person_agnostic(items, tiles, FRAME), det


def test_a_more_confident_truncated_box_no_longer_beats_the_complete_box():
    complete = (PLAYER, 0.69, [100.0, 290.0, 120.0, 330.0])      # seen whole by the upper tile
    truncated = (PLAYER, 0.78, [100.0, 300.0, 120.0, 330.0])     # same player, cut at the top of the lower tile
    old, old_det = merge("person_agnostic", [complete, truncated], [UPPER, LOWER])
    assert old == [truncated] and old_det.merge_statistics["suppressed_same_role"] == 1  # the defect: the complete box is lost
    new, new_det = merge("edge_aware", [complete, truncated], [UPPER, LOWER])
    assert new == [complete]
    assert new_det.merge_statistics["suppressed_edge_truncated_before_nms"] == 1
    assert new_det.merge_statistics["suppressed_same_role"] == 0 and "edge_truncated_role_differs" not in new_det.merge_statistics


def test_edge_aware_keeps_the_truncated_box_when_the_evidence_does_not_support_dropping_it():
    truncated = (PLAYER, 0.9, [100.0, 300.0, 120.0, 330.0])
    weak_complete = (PLAYER, 0.9 - EDGE_CONFIDENCE_MARGIN - 0.05, [100.0, 290.0, 120.0, 330.0])
    for items in ([truncated, weak_complete], [truncated, (PLAYER, 0.95, [100.0, 300.0, 120.0, 340.0])]):
        # a much weaker larger box, or a larger box that does not extend past the cut: no pre-NMS removal
        tiles = [LOWER, UPPER]
        old, _ = merge("person_agnostic", items, tiles)
        new, det = merge("edge_aware", items, tiles)
        assert new == old and "suppressed_edge_truncated_before_nms" not in det.merge_statistics
    alone, det = merge("edge_aware", [truncated], [LOWER])
    assert alone == [truncated] and not any(det.merge_statistics.values())                                 # nothing to complete it with


def test_edge_aware_never_removes_a_distinct_neighbour_or_a_box_from_the_same_tile():
    left = (PLAYER, 0.8, [100.0, 300.0, 120.0, 330.0])
    neighbour = (PLAYER, 0.7, [140.0, 285.0, 160.0, 330.0])      # next to it, not containing it
    same_tile_big = (PLAYER, 0.7, [100.0, 290.0, 120.0, 330.0])
    assert merge("edge_aware", [left, neighbour], [LOWER, UPPER])[0] == [left, neighbour]
    assert len(merge("edge_aware", [left, same_tile_big], [LOWER, LOWER])[0]) == 1           # same tile: ordinary NMS only


def test_edge_aware_keeps_the_ball_out_of_truncation_decisions_and_reports_role_differences():
    ball = (BALL, 0.9, [100.0, 300.0, 110.0, 310.0])
    big_person = (PLAYER, 0.8, [95.0, 280.0, 130.0, 330.0])
    kept, det = merge("edge_aware", [ball, big_person], [LOWER, UPPER])
    assert sorted(o[0] for o in kept) == [BALL, PLAYER] and not any(det.merge_statistics.values())       # the ball is never truncated by a person
    gk_cut = (GK, 0.8, [100.0, 300.0, 120.0, 330.0])
    player_whole = (PLAYER, 0.7, [100.0, 290.0, 120.0, 330.0])
    kept, det = merge("edge_aware", [gk_cut, player_whole], [LOWER, UPPER])
    assert kept == [player_whole]                                                              # the complete box wins ...
    assert det.merge_statistics["edge_truncated_role_differs"] == 1                            # ... and the lost role label is counted


def test_edge_aware_never_leaves_a_chain_of_truncated_boxes_without_a_survivor():
    a = (PLAYER, 0.8, [100.0, 300.0, 120.0, 330.0])              # cut by LOWER, completed by b
    b = (PLAYER, 0.75, [100.0, 290.0, 120.0, 330.0])             # itself cut by its tile's edge, completed by c
    c = (PLAYER, 0.7, [100.0, 280.0, 120.0, 335.0])
    tiles = [LOWER, (0, 290, 400, 400), UPPER]
    kept, _ = merge("edge_aware", [a, b, c], tiles)
    assert kept == [c]


def test_edge_aware_equals_person_agnostic_when_no_box_is_cut():
    truth = players_across_the_frame() + [(GK, 0.7, (500.0, 300.0, 520.0, 340.0)), (BALL, 0.8, (395.0, 100.0, 405.0, 110.0))]
    straddling = [person(390, 100), person(100, 296)]            # the stub makes truncated views less confident, so the old rule works
    for scene in (truth, straddling):
        assert infer(detector(scene, tile_height=300, tile_merge="edge_aware")) == infer(detector(scene, tile_height=300, tile_merge="person_agnostic"))


def test_edge_aware_is_opt_in_and_recorded():
    assert YOLOPlayerDetector(tile_width=400).tile_merge == "class_aware"
    d = YOLOPlayerDetector(tile_width=1280, tile_height=1280, tile_merge="edge_aware").describe()["tile_merge"]
    assert d["policy"] == "edge_aware" and d["edge_px"] == EDGE_PX and "before same-role NMS" in d["edge_truncation"]
    assert "edge_truncation" not in YOLOPlayerDetector(tile_width=1280, tile_merge="person_agnostic").describe()["tile_merge"]
    with pytest.raises(ValueError):
        YOLOPlayerDetector(tile_merge="edge_aware")                                            # still needs --tile-width
