"""Tiling geometry and benchmark matching (no model weights needed)."""

import pytest

from football_fatigue.vision.benchmark import iou, match
from football_fatigue.vision.detector import YOLOPlayerDetector


def test_tile_origins_cover_full_width():
    det = YOLOPlayerDetector(tile_width=1280, tile_overlap=0.2)
    xs = det._tile_origins(4096)
    assert xs[0] == 0 and xs[-1] + 1280 == 4096
    assert all(b - a <= 1280 * 0.8 + 1 for a, b in zip(xs, xs[1:]))  # overlapping
    assert all(a < b for a, b in zip(xs, xs[1:]))


def test_tile_larger_than_frame_gives_single_tile():
    assert YOLOPlayerDetector(tile_width=5000)._tile_origins(4096) == [0]


def test_invalid_tile_arguments():
    with pytest.raises(ValueError):
        YOLOPlayerDetector(tile_width=10)
    with pytest.raises(ValueError):
        YOLOPlayerDetector(tile_width=640, tile_overlap=1.0)


def test_match_counts_and_one_to_one():
    refs = [(0, 0, 10, 20), (100, 0, 110, 20)]
    preds = [(0.9, (0, 0, 10, 20)), (0.8, (1, 0, 11, 20)), (0.7, (500, 0, 510, 20))]
    tp, fp, fn = match(preds, refs, 0.5)
    assert (tp, fp, fn) == (1, 2, 1)  # duplicate and distant box are FPs
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
