"""Kit-colour classification: adaptive blue/white split, 'other' kits, 'unknown' honesty."""

import numpy as np
import pytest

from football_fatigue.vision.models import BoundingBox
from football_fatigue.vision.team_classifier import KitColorClassifier
from football_fatigue.vision.tracker import MultiObjectTracker

BOX = BoundingBox(10, 10, 40, 80)


def _frame(rgb):
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    frame[:] = (40, 150, 40)  # grass
    frame[10:80, 10:40] = rgb
    return frame


BLUE = (20, 60, 220)
WHITE = (235, 235, 240)
WHITE_IN_SHADE = (150, 170, 235)  # bluish white under a lighting shift
YELLOW = (240, 220, 20)


def _feed(kc, tid, rgb, n=12):
    for _ in range(n):
        label = kc.observe(tid, _frame(rgb), BOX)
    return label


def _warm(kc, white=WHITE):
    for i in range(120):  # both kits in view so the adaptive threshold has two clusters
        kc.observe(1000 + i, _frame(BLUE if i % 2 else white), BOX)


def test_blue_and_white_separated():
    kc = KitColorClassifier()
    _warm(kc)
    assert _feed(kc, 1, BLUE)[0] == "blue"
    assert _feed(kc, 2, WHITE)[0] == "white"


def test_adapts_when_white_kit_gets_bluish_in_shade():
    kc = KitColorClassifier()
    _warm(kc, white=WHITE_IN_SHADE)
    assert _feed(kc, 1, BLUE)[0] == "blue"
    assert _feed(kc, 2, WHITE_IN_SHADE)[0] == "white"


def test_goalkeeper_style_kit_is_other_not_a_team():
    kc = KitColorClassifier()
    _warm(kc)
    assert _feed(kc, 3, YELLOW)[0] == "other"


def test_unknown_until_enough_evidence_and_without_two_clusters():
    kc = KitColorClassifier()
    assert _feed(kc, 1, BLUE, n=3)[0] == "unknown"  # too few observations
    only_blue = KitColorClassifier()
    for i in range(150):
        only_blue.observe(i, _frame(BLUE), BOX)
    assert _feed(only_blue, 999, BLUE)[0] == "unknown"  # no separable second kit: refuse to guess


def test_tracker_defaults_are_the_tuned_values():
    tr = MultiObjectTracker()
    assert (tr.track_activation_threshold, tr.track_buffer, tr.matching_threshold) == (0.5, 75, 0.95)
