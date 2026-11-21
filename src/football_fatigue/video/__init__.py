"""Session B video engine (V1): image-space occlusion events from cached detections and tracks.

The package imports, and never modifies, the validated pieces of ``football_fatigue.vision``
(detection cache, tracker output) and ``football_fatigue.multicam_lab.media`` (exact frame access).
It does not do identity, roles, calibration or Session A's trajectory models.
"""

ENGINE_VERSION = "0.2.0"
