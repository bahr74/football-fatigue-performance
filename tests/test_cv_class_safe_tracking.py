"""Class-safe person tracking: the ball never enters the person tracker, and a track's role is an
evidence-weighted estimate rather than whatever the detector said last."""

import numpy as np
import pytest

from football_fatigue.vision.detection_cache import DetectionCacheReader, DetectionCacheWriter
from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection, RawObservation
from football_fatigue.vision.pipeline import VisionTrackingPipeline
from football_fatigue.vision.tracker import MultiObjectTracker, RoleEstimator

PLAYER, GK, REF, BALL = DetectedClass.PLAYER, DetectedClass.GOALKEEPER, DetectedClass.REFEREE, DetectedClass.BALL
BACKENDS = [False, True]


def tracker(use_supervision, **kwargs):
    if use_supervision:
        pytest.importorskip("supervision")
    t = MultiObjectTracker(use_supervision=use_supervision, **kwargs)
    assert t.backend == ("bytetrack" if use_supervision else "iou_fallback")
    return t


def det(cls, x, y=100.0, w=20.0, h=45.0, conf=0.9):
    return Detection(cls, conf, BoundingBox(x, y, x + w, y + h))


def walk(t, frames, start=0):
    """frames: per-frame detection lists; returns the per-frame active tracks."""
    return [t.update(dets, start + i) for i, dets in enumerate(frames)]


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_ball_detections_are_never_tracked_and_never_change_person_ids(use_supervision):
    def stream(with_ball):
        frames = []
        for f in range(20):
            people = [det(PLAYER, 100 + 2 * f), det(PLAYER, 400 - f), det(REF, 700)]
            ball = [det(BALL, 100 + 2 * f, y=110, w=8, h=8, conf=0.95)] if with_ball else []  # sits on the first player
            frames.append(people + ball)
        return frames

    with_ball, without = tracker(use_supervision), tracker(use_supervision)
    a, b = walk(with_ball, stream(True)), walk(without, stream(False))
    assert [[(t.track_id, t.class_name, t.bbox) for t in f] for f in a] == [[(t.track_id, t.class_name, t.bbox) for t in f] for f in b]
    assert all(t.class_name != BALL for f in a for t in f)
    assert with_ball.ignored_non_person_detections == 20 and without.ignored_non_person_detections == 0
    assert {t.track_id for f in a for t in f} == {1, 2, 3}  # no id was spent on the ball


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_pipeline_keeps_the_ball_separate_from_players(use_supervision):
    pipe = VisionTrackingPipeline(None, tracker=tracker(use_supervision), detector=object())
    frames = []
    for f in range(12):
        frames.append([det(PLAYER, 100 + 2 * f), det(BALL, 100 + 2 * f, y=110, w=8, h=8, conf=0.9 - 0.01 * f), det(BALL, 600, y=300, w=8, h=8, conf=0.5)])
    out = [pipe.process_frame(None, i, detections=d) for i, d in enumerate(frames)]
    assert all(len(v.players) == 1 for v in out) and len({v.players[0].track_id for v in out}) == 1
    assert all(v.players[0].detected_class == PLAYER for v in out)
    assert all(v.ball is not None for v in out)
    assert out[0].ball.confidence == pytest.approx(0.9)       # the higher-confidence of the two ball detections
    assert pipe.ball_detections_seen == 24 and len(pipe.last_ball_detections) == 2
    ball_only = pipe.process_frame(None, 12, detections=[det(BALL, 50, y=50, w=8, h=8)])
    assert ball_only.players == [] and ball_only.ball is not None
    nothing = pipe.process_frame(None, 13, detections=[])
    assert nothing.players == [] and nothing.ball is None and pipe.last_ball_detections == []


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_role_flicker_neither_breaks_the_track_nor_changes_the_role_estimate(use_supervision):
    labels = [PLAYER] * 10 + [GK, GK] + [PLAYER] * 8
    stable, latest = tracker(use_supervision), tracker(use_supervision, role_policy="latest")
    frames = [[det(label, 100 + 2 * f)] for f, label in enumerate(labels)]
    a, b = walk(stable, frames), walk(latest, frames)
    assert {t.track_id for f in a for t in f} == {1} == {t.track_id for f in b for t in f}
    assert all(f[0].class_name == PLAYER for f in a)
    assert [f[0].observed_class for f in a] == labels
    assert [f[0].class_name for f in b] == labels                      # "latest" reports the raw label
    flicker = a[10][0]
    assert 0.5 < flicker.role_confidence < 1.0 and a[5][0].role_confidence == 1.0
    assert stable.roles.stats()["observed_label_changes"] == 2 and stable.roles.stats()["role_changes"] == 0


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_sustained_contrary_evidence_changes_the_role_but_not_the_id(use_supervision):
    t = tracker(use_supervision)
    labels = [PLAYER] * 5 + [REF] * 120
    out = walk(t, [[det(label, 100 + f)] for f, label in enumerate(labels)])
    assert {x.track_id for f in out for x in f} == {1}
    roles = [f[0].class_name for f in out]
    assert roles[5] == PLAYER and roles[-1] == REF                     # one contrary frame does not flip it; 120 do
    assert roles.count(PLAYER) + roles.count(REF) == len(roles) and roles == sorted(roles, key=lambda r: r == REF)  # one switch
    assert t.roles.stats()["role_changes"] == 1 and out[-1][0].role_confidence > 0.9


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_empty_frames_and_occlusions(use_supervision):
    t = tracker(use_supervision)
    assert t.update([], 0) == []                                      # nothing at all
    assert t.update([det(BALL, 10, w=8, h=8)], 1) == []               # only a ball
    out = walk(t, [[det(GK, 100 + f)] for f in range(8)], start=2)
    gk_id = next(f[0].track_id for f in out if f)  # ByteTrack confirms a new track on its second frame
    gap = walk(t, [[] for _ in range(10)], start=10)                  # occluded for 10 frames, inside the lost-track buffer
    assert all(f == [] for f in gap)
    after = t.update([det(PLAYER, 112, conf=0.9)], 20)                # reappears with a contrary single label
    assert [x.track_id for x in after] == [gk_id]
    assert after[0].class_name == GK and after[0].observed_class == PLAYER  # role evidence survived the gap
    assert t.roles.stats()["role_changes"] == 0


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_unflickering_streams_are_identical_under_both_policies(use_supervision):
    frames = [[det(PLAYER, 100 + f), det(PLAYER, 300), det(REF, 600 - f)] for f in range(15)]
    a = walk(tracker(use_supervision), frames)
    b = walk(tracker(use_supervision, role_policy="latest"), frames)
    assert [[(x.track_id, x.class_name, x.bbox, x.confidence) for x in f] for f in a] == \
           [[(x.track_id, x.class_name, x.bbox, x.confidence) for x in f] for f in b]
    assert all(x.observed_class == x.class_name and x.role_confidence == 1.0 for f in a for x in f)


def test_serialised_frames_keep_every_historical_key_and_add_the_role_fields():
    pipe = VisionTrackingPipeline(None, tracker=tracker(False), detector=object())
    record = pipe.process_frame(None, 0, detections=[det(PLAYER, 100)]).to_dict()
    player = record["players"][0]
    historical = {"track_id", "side", "jersey", "detected_class", "confidence", "bbox_xyxy_px", "pixel_x", "pixel_y", "x", "y",
                  "speed_m_s", "estimated", "position_usable", "kit_color", "kit_confidence", "kit_vote", "identity_usable", "coord_source"}
    assert historical <= set(player) and set(player) - historical == {"observed_class", "role_confidence"}
    assert player["detected_class"] == "player" == player["observed_class"] and player["role_confidence"] == 1.0
    assert set(record) == {"frame_num", "timestamp_s", "fps", "players", "ball"}


def test_referee_role_still_maps_to_the_referee_side():
    pipe = VisionTrackingPipeline(None, tracker=tracker(False), detector=object())
    for f in range(3):
        frame = pipe.process_frame(None, f, detections=[det(REF, 100 + f)])
    assert frame.players[0].team_side.value == "referee" and frame.players[0].detected_class == REF


def test_role_estimator_decay_margin_ties_and_pruning():
    est = RoleEstimator(frame_rate=10.0, half_life_s=1.0, switch_margin=1.5)   # half-life = 10 frames
    assert est.observe(1, PLAYER, 0.8, 0) == (PLAYER, 1.0)
    role, share = est.observe(1, GK, 0.8, 1)
    assert role == PLAYER and 0.4 < share < 0.6                  # one contrary vote is not 1.5x the current evidence
    for f in range(2, 40):
        role, share = est.observe(1, GK, 0.8, f)
    assert role == GK and share > 0.9
    assert est.stats()["role_changes"] == 1 and est.stats()["tracks_with_role_change"] == 1
    est.observe(2, REF, 0.5, 5)
    est.prune(frame_idx=100, max_age_frames=50)                   # track 1 last seen at 39, track 2 at 5
    assert est.observe(2, PLAYER, 0.5, 101) == (PLAYER, 1.0)      # forgotten, so it starts afresh
    # a long gap decays old evidence
    old = RoleEstimator(frame_rate=10.0, half_life_s=1.0)
    old.observe(7, PLAYER, 0.9, 0)
    assert old.observe(7, GK, 0.9, 100)[0] == GK
    for bad in (dict(half_life_s=0), dict(half_life_s=float("nan")), dict(switch_margin=0.9)):
        with pytest.raises(ValueError):
            RoleEstimator(25.0, **bad)
    with pytest.raises(ValueError, match="role_policy"):
        MultiObjectTracker(role_policy="newest")


def test_frame_rate_change_rescales_the_role_memory():
    t = tracker(False, frame_rate=25.0)
    assert t.roles.half_life_frames == pytest.approx(50.0)
    t.set_frame_rate(10.0)
    assert t.roles.half_life_frames == pytest.approx(20.0)


@pytest.mark.parametrize("use_supervision", BACKENDS)
def test_cache_replay_reproduces_class_safe_output_including_the_ball(tmp_path, use_supervision):
    def raw(cls, label, class_id, x, conf, y=100.0, w=20.0, h=45.0):
        return RawObservation(class_id, label, cls, conf, (x, y, x + w, y + h), None if cls == BALL else True, None)

    def observations(f):
        label = GK if f in (4, 5) else PLAYER
        return [
            raw(label, label.value, 3 if label == GK else 1, 100.0 + 2 * f, 0.9),
            raw(REF, "referee", 2, 400.0, 0.8),
            raw(BALL, "ball", 0, 100.0 + 2 * f, 0.9, y=110.0, w=8.0, h=8.0),   # on top of the first player
            raw(PLAYER, "player", 1, 700.0, 0.3),                                # below the tracking threshold
        ]

    writer = DetectionCacheWriter(tmp_path / "cache", {
        "run_id": "synthetic", "video": {"sha256": "a" * 64, "fps": 25.0, "width": 1000, "height": 400},
        "detector": {"weights_sha256": "b" * 64, "confidence_floor": 0.1}, "tracking_input": {"confidence_threshold": 0.4},
        "window": {"start_frame": 0},
    })
    saving = VisionTrackingPipeline(
        None, tracker=tracker(use_supervision), detector=type("D", (), {"observe": lambda self, *_a, **_k: observations(self.n)})(),
        observation_sink=writer.write_frame, tracking_confidence=0.4,
    )
    pixels = np.zeros((400, 1000, 3), dtype=np.uint8)
    saved = []
    for f in range(12):
        saving.detector.n = f
        saved.append(saving.process_frame(pixels, f).to_dict())
    writer.close(complete=True)

    source = DetectionCacheReader(tmp_path / "cache").detection_source(0, 0.4)
    replaying = VisionTrackingPipeline(None, tracker=tracker(use_supervision), detector=object())
    replayed = [replaying.process_frame(pixels, f, detections=source(f)).to_dict() for f in range(12)]
    assert replayed == saved
    assert replaying.ball_detections_seen == saving.ball_detections_seen == 12
    assert all(len(r["players"]) == 2 and r["ball"] is not None for r in replayed)
    assert {p["track_id"] for r in replayed for p in r["players"]} == {1, 2}
    assert [r["players"][0]["observed_class"] for r in replayed][4:6] == ["goalkeeper", "goalkeeper"]
    assert {r["players"][0]["detected_class"] for r in replayed} == {"player"}
