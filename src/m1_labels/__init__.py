"""M1 answer key: per-frame possession labels built from PFF tags (evaluation only).

This package reads PFF event tags. The tag-free model package ``m1_possession`` must never import it.
"""
from .labeller import (DEAD, CONTROL, LOOSE, STATE_NAMES, TOLERANCE_S, TRAVELLING, UNKNOWN, Labels, Row, Spell,
                       build_spells, label_match)

__all__ = ["DEAD", "CONTROL", "TRAVELLING", "LOOSE", "UNKNOWN", "STATE_NAMES", "TOLERANCE_S", "Labels", "Row", "Spell",
           "build_spells", "label_match"]
