"""Detection cache and deterministic replay (synthetic fixtures only; no model weights or real footage).

The stub detector and the generated video are synthetic and say nothing about real detector output.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_fatigue.vision import __main__ as vision_main
from football_fatigue.vision.detection_cache import (
    CACHE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    DetectionCacheError,
    DetectionCacheReader,
    DetectionCacheWriter,
    detections_from_observations,
    resolve_replay,
)
from football_fatigue.vision.detector import YOLOPlayerDetector
from football_fatigue.vision.models import DetectedClass, RawObservation
from football_fatigue.vision.pipeline import VisionTrackingPipeline
from football_fatigue.vision.tracker import MultiObjectTracker

np = pytest.importorskip("numpy")

SHA_VIDEO = "a" * 64
SHA_WEIGHTS = "b" * 64


def obs(conf=0.9, mapped=DetectedClass.PLAYER, box=(10.0, 10.0, 20.0, 40.0), turf=True, mask=None,
        class_id=1, label="player"):
    return RawObservation(class_id, label, mapped, conf, box, turf, mask)


def provenance(floor=0.4, start=0, video_sha=SHA_VIDEO, weights_sha=SHA_WEIGHTS, tracking=None):
    return {
        "run_id": "synthetic",
        "video": {"sha256": video_sha, "fps": 25.0, "width": 160, "height": 96},
        "detector": {"weights_sha256": weights_sha, "confidence_floor": floor, "image_size": 640, "tile_width": None},
        "tracking_input": {"confidence_threshold": floor if tracking is None else tracking},
        "window": {"start_frame": start},
    }


def write_cache(directory, frames, **kwargs):
    writer = DetectionCacheWriter(directory, provenance(**kwargs))
    for number, observations in frames:
        writer.write_frame(number, number / 25.0, observations)
    writer.close(complete=True)
    return writer


def sample_frames(count=4, start=0):
    return [(start + i, [obs(0.9 - 0.01 * i), obs(0.5, DetectedClass.REFEREE, (50.0, 12.0, 60.0, 40.0), label="referee", class_id=2),
                         obs(0.8, None, (1.0, 1.0, 5.0, 5.0), None, None, 56, "chair")]) for i in range(count)]


# --- observation and detector -------------------------------------------------

@pytest.mark.parametrize("kwargs, kept", [
    ({}, True),
    ({"mapped": None}, False),
    ({"turf": False}, False),
    ({"mask": False}, False),
    ({"turf": None, "mask": None, "mapped": DetectedClass.BALL}, True),
])
def test_observation_kept_follows_recorded_decisions(kwargs, kept):
    observation = obs(**kwargs)
    assert observation.kept is kept
    assert (observation.to_detection() is not None) is kept


def test_detector_observe_records_decisions_and_detect_keeps_exactly_the_kept_ones():
    detector = YOLOPlayerDetector("x.pt", 0.4)
    detector._model = SimpleNamespace(names={0: "ball", 1: "player", 2: "referee", 3: "chair"})
    green_box, red_box, outside_box = (10, 10, 30, 50), (40, 10, 60, 50), (70, 10, 90, 50)
    detector._infer = lambda frame: [
        (1, 0.9, list(green_box)), (1, 0.8, list(red_box)), (2, 0.7, list(outside_box)),
        (0, 0.6, [1.0, 1.0, 4.0, 4.0]), (3, 0.5, [0.0, 0.0, 0.0, 0.0]),  # unmapped label, degenerate box
    ]
    frame = np.zeros((60, 100, 3), dtype=np.uint8)
    frame[:, 0:35] = (30, 160, 30)    # grass under the first box
    frame[:, 35:100] = (200, 20, 20)  # kit-coloured elsewhere
    polygon = [(35.0, 0.0), (65.0, 0.0), (65.0, 60.0), (35.0, 60.0)]  # excludes the third box's foot point

    observations = detector.observe(frame, pitch_mask_polygon=polygon)

    assert [o.class_label for o in observations] == ["player", "player", "referee", "ball", "chair"]
    assert [o.passes_turf_filter for o in observations] == [False, True, True, None, None]
    assert [o.inside_pitch_mask for o in observations] == [None, True, False, None, None]
    assert [o.kept for o in observations] == [False, True, False, True, False]
    detections = detector.detect(frame, pitch_mask_polygon=polygon)
    assert [d.class_name for d in detections] == [DetectedClass.PLAYER, DetectedClass.BALL]
    assert detections[0].bbox.x1 == 40.0 and detections[0].confidence == 0.8
    assert detector.describe()["class_names"] == {"0": "ball", "1": "player", "2": "referee", "3": "chair"}


def test_detections_from_observations_threshold_is_exclusive():
    observations = [obs(0.40), obs(0.41), obs(0.9, turf=False)]
    assert [d.confidence for d in detections_from_observations(observations, 0.40)] == [0.41]
    assert len(detections_from_observations(observations)) == 2


# --- writer / reader ----------------------------------------------------------

def test_round_trip_is_exact_and_manifest_records_provenance(tmp_path):
    frames = sample_frames()
    frames[0][1][0] = obs(0.123456789012345678, box=(10.1234567891, 10.5, 20.25, 40.75))
    write_cache(tmp_path / "cache", frames)

    reader = DetectionCacheReader(tmp_path / "cache")

    assert reader.manifest["schema_version"] == CACHE_SCHEMA_VERSION
    assert reader.manifest["status"] == "complete"
    assert (reader.first_frame, reader.last_frame) == (0, 3)
    assert reader.manifest["video"]["sha256"] == SHA_VIDEO and reader.manifest["detector"]["weights_sha256"] == SHA_WEIGHTS
    assert reader.manifest["counts"]["observations"] == 12 and reader.manifest["counts"]["kept"] == 8
    assert reader.manifest["counts"]["mapped_unmapped"] == 4
    read_back = [(n, o) for n, _, o in reader.iter_records()]
    assert read_back == [(n, o) for n, o in frames]  # dataclass equality: exact floats, decisions and labels


def test_reader_accepts_a_run_directory_containing_the_cache(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_cache(run / "detection_cache", sample_frames(2))
    assert DetectionCacheReader(run).last_frame == 1


def test_writer_never_overwrites_and_requires_consecutive_frames(tmp_path):
    write_cache(tmp_path / "cache", sample_frames(1))
    with pytest.raises(DetectionCacheError, match="never overwritten"):
        DetectionCacheWriter(tmp_path / "cache", provenance())
    writer = DetectionCacheWriter(tmp_path / "other", provenance(start=5))
    with pytest.raises(DetectionCacheError, match="expected 5"):
        writer.write_frame(6, 0.24, [])
    writer.write_frame(5, 0.2, [])
    with pytest.raises(DetectionCacheError, match="expected 6"):
        writer.write_frame(8, 0.32, [])
    writer.close(complete=False, error="stopped")


def test_writer_rejects_incomplete_provenance_and_non_finite_values(tmp_path):
    bad = provenance()
    del bad["video"]["sha256"]
    with pytest.raises(DetectionCacheError, match="video"):
        DetectionCacheWriter(tmp_path / "a", bad)
    writer = DetectionCacheWriter(tmp_path / "b", provenance())
    with pytest.raises(ValueError):
        writer.write_frame(0, 0.0, [obs(float("nan"))])
    writer.close(complete=False, error="nan")


@pytest.mark.parametrize("complete, frames", [(False, 2), (True, 0)])
def test_unfinished_or_empty_cache_is_refused(tmp_path, complete, frames):
    writer = DetectionCacheWriter(tmp_path / "cache", provenance())
    for number, observations in sample_frames(frames):
        writer.write_frame(number, 0.0, observations)
    writer.close(complete=complete, error="interrupted")
    with pytest.raises(DetectionCacheError, match="not complete"):
        DetectionCacheReader(tmp_path / "cache")


def test_cache_still_being_written_is_refused(tmp_path):
    writer = DetectionCacheWriter(tmp_path / "cache", provenance())
    writer.write_frame(0, 0.0, [obs()])
    with pytest.raises(DetectionCacheError, match="writing"):
        DetectionCacheReader(tmp_path / "cache")
    writer.close(complete=False)


def _manifest_edit(cache, **changes):
    path = cache / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("mutate, message", [
    (lambda cache: (cache / RECORDS_FILENAME).write_bytes((cache / RECORDS_FILENAME).read_bytes()[:-20]), "size"),
    (lambda cache: (cache / RECORDS_FILENAME).write_bytes((cache / RECORDS_FILENAME).read_bytes().replace(b"0.9", b"0.8", 1)), "SHA-256"),
    (lambda cache: _manifest_edit(cache, schema_version=99), "schema version"),
    (lambda cache: _manifest_edit(cache, columns=["x"]), "columns"),
    (lambda cache: _manifest_edit(cache, last_frame=7), "inconsistent"),
    (lambda cache: (cache / RECORDS_FILENAME).unlink(), "missing"),
    (lambda cache: (cache / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8"), "not valid JSON"),
    (lambda cache: (cache / MANIFEST_FILENAME).unlink(), "not a detection cache"),
])
def test_corrupted_or_altered_cache_is_detected(tmp_path, mutate, message):
    write_cache(tmp_path / "cache", sample_frames())
    mutate(tmp_path / "cache")
    with pytest.raises(DetectionCacheError, match=message):
        DetectionCacheReader(tmp_path / "cache")


def test_bad_records_are_caught_while_reading(tmp_path):
    write_cache(tmp_path / "cache", sample_frames())
    records = tmp_path / "cache" / RECORDS_FILENAME
    lines = records.read_text(encoding="utf-8").splitlines()
    cases = {
        "frame gap": [lines[0], lines[2], lines[3]],
        "bad json": [lines[0], "{oops", lines[2], lines[3]],
        "bad column count": [lines[0], lines[1].replace('"observations":[[', '"observations":[[7,', 1), lines[2], lines[3]],
        "bad mapped class": [lines[0], lines[1].replace('"player","player"', '"player","striker"', 1), lines[2], lines[3]],
    }
    for name, edited in cases.items():
        records.write_text("\n".join(edited) + "\n", encoding="utf-8")
        _manifest_edit(tmp_path / "cache", records_bytes=records.stat().st_size)  # let the size check pass
        reader = DetectionCacheReader(tmp_path / "cache", verify_hash=False)
        with pytest.raises(DetectionCacheError, match="line|fewer"):
            list(reader.iter_records())


# --- replay rules ---------------------------------------------------------------

def _manifest(floor, tracking=None):
    return {"detector": {"confidence_floor": floor}, "tracking_input": {"confidence_threshold": floor if tracking is None else tracking}}


def test_replay_modes():
    same = resolve_replay(_manifest(0.4), "compatibility", 0.4)
    assert same["experimental"] is False and same["equals_saving_run_tracking_confidence"] is True
    # a cache saved at floor 0.1 for a 0.4 tracking run: compatibility reconstructs the 0.4 input
    assert resolve_replay(_manifest(0.1, 0.4), "compatibility", 0.4)["cache_confidence_floor"] == 0.1
    experimental = resolve_replay(_manifest(0.1, 0.4), "experimental", 0.25)
    assert experimental["experimental"] is True and experimental["equals_saving_run_tracking_confidence"] is False
    assert experimental["saving_run_tracking_confidence"] == 0.4 and experimental["tracking_confidence"] == 0.25
    assert resolve_replay(_manifest(0.1, 0.4), "experimental", 0.5)["tracking_confidence"] == 0.5
    assert resolve_replay(_manifest(0.1, 0.4), "experimental", 0.1)["tracking_confidence"] == 0.1  # exactly the floor
    with pytest.raises(DetectionCacheError, match="compatibility replay reconstructs"):
        resolve_replay(_manifest(0.1, 0.4), "compatibility", 0.5)
    with pytest.raises(DetectionCacheError, match="compatibility replay reconstructs"):
        resolve_replay(_manifest(0.1, 0.4), "compatibility", 0.1)  # the floor is not the saving run's tracker input
    for mode in ("compatibility", "experimental"):
        with pytest.raises(DetectionCacheError, match="below the cache floor"):
            resolve_replay(_manifest(0.1, 0.4), mode, 0.05)
    with pytest.raises(DetectionCacheError, match="unknown replay mode"):
        resolve_replay(_manifest(0.4), "lenient", 0.4)
    with pytest.raises(DetectionCacheError, match="below the cache floor"):
        resolve_replay(_manifest(0.4), "experimental", float("nan"))


def test_two_thresholds_are_validated_and_recorded_separately(tmp_path):
    with pytest.raises(DetectionCacheError, match="below the cache floor"):
        DetectionCacheWriter(tmp_path / "bad", provenance(floor=0.4, tracking=0.2))
    frames = [(0, [obs(0.9), obs(0.3, box=(30.0, 10.0, 40.0, 40.0)), obs(0.25, DetectedClass.REFEREE, label="referee", class_id=2)])]
    write_cache(tmp_path / "cache", frames, floor=0.2, tracking=0.4)
    reader = DetectionCacheReader(tmp_path / "cache")
    assert reader.confidence_floor == 0.2 and reader.tracking_confidence == 0.4
    assert reader.manifest["tracking_input"]["confidence_threshold"] == 0.4
    assert reader.manifest["detector"]["confidence_floor"] == 0.2
    assert reader.manifest["counts"]["kept"] == 3 and reader.manifest["counts"]["tracking_input"] == 1
    _manifest_edit(tmp_path / "cache", tracking_input={"confidence_threshold": 0.1})
    with pytest.raises(DetectionCacheError, match="threshold below its floor"):
        DetectionCacheReader(tmp_path / "cache")
    _manifest_edit(tmp_path / "cache", tracking_input={})
    with pytest.raises(DetectionCacheError, match="tracking_input is missing"):
        DetectionCacheReader(tmp_path / "cache")


def test_old_schema_version_one_caches_are_refused(tmp_path):
    write_cache(tmp_path / "cache", sample_frames(2))
    _manifest_edit(tmp_path / "cache", schema_version=1)
    with pytest.raises(DetectionCacheError, match="schema version"):
        DetectionCacheReader(tmp_path / "cache")


def test_replay_is_refused_for_the_wrong_video_geometry_weights_or_window(tmp_path):
    write_cache(tmp_path / "cache", sample_frames(10, start=100), start=100)
    reader = DetectionCacheReader(tmp_path / "cache")
    ok = dict(video_sha256=SHA_VIDEO, fps=25.0, width=160, height=96, first_frame=102, last_frame=105)
    reader.validate_for_replay(**ok, weights_sha256=SHA_WEIGHTS)
    for change, message in [
        ({"video_sha256": "c" * 64}, "different video"),
        ({"width": 320}, "dimensions"),
        ({"fps": 30.0}, "frame rate"),
        ({"weights_sha256": "d" * 64}, "different|--weights"),
        ({"first_frame": 99}, "outside the cached window"),
        ({"last_frame": 110}, "outside the cached window"),
    ]:
        with pytest.raises(DetectionCacheError, match=message):
            reader.validate_for_replay(**{**ok, **change})


def test_detection_source_is_sequential_and_starts_where_asked(tmp_path):
    write_cache(tmp_path / "cache", sample_frames(5))
    reader = DetectionCacheReader(tmp_path / "cache")
    source = reader.detection_source(start_frame=2, min_confidence=0.4)
    first = source(2)
    assert [d.class_name for d in first] == [DetectedClass.PLAYER, DetectedClass.REFEREE]  # the unmapped chair is dropped
    with pytest.raises(DetectionCacheError, match="out of step"):
        source(4)  # skipped frame 3
    exhausted = reader.detection_source(start_frame=3, min_confidence=0.4)
    exhausted(3), exhausted(4)
    with pytest.raises(DetectionCacheError, match="no record for frame 5"):
        exhausted(5)


# --- pipeline determinism -------------------------------------------------------

class ScriptedDetector:
    """Synthetic detector: raw observations are a deterministic function of the call count."""

    def __init__(self, floor=0.4):
        self.confidence_threshold = floor
        self.calls = 0
        self.merge_statistics = {}

    def observe(self, frame_rgb, pitch_mask_polygon=None):
        n = self.calls
        self.calls += 1
        raw = [
            obs(0.9, box=(20.0 + 2 * n, 30.0, 32.0 + 2 * n, 60.0)),
            obs(0.8, box=(80.0 - n, 30.0, 92.0 - n, 62.0)),
            obs(0.5, DetectedClass.GOALKEEPER, (120.0, 28.0, 130.0, 58.0), label="goalkeeper", class_id=3),
            obs(0.45, DetectedClass.REFEREE, (60.0, 10.0, 70.0, 40.0), label="referee", class_id=2),
            obs(0.42, DetectedClass.BALL, (40.0 + n, 70.0, 44.0 + n, 74.0), None, None, 0, "ball"),
            obs(0.9, box=(5.0, 5.0, 15.0, 30.0), turf=False),
            obs(0.85, None, (1.0, 1.0, 9.0, 9.0), None, None, 56, "chair"),
            obs(0.3, box=(100.0, 50.0, 110.0, 80.0)),
        ]
        return [o for o in raw if o.confidence > self.confidence_threshold]

    def detect(self, frame_rgb, pitch_mask_polygon=None):
        return detections_from_observations(self.observe(frame_rgb, pitch_mask_polygon))

    def describe(self):
        return {
            "model_path": "scripted.pt", "confidence_floor": self.confidence_threshold, "image_size": 640,
            "tile_width": None, "tile_overlap": 0.2, "merge_iou": 0.5, "half": False, "device": "cpu",
            "tile_height": None, "tile_batch": None, "tile_merge": {"policy": "class_aware", "merge_iou": 0.5},
            "turf_filter": {"name": "looks_like_player", "version": 1}, "class_names": {"0": "ball"},
        }


def _frames(count=12):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, (96, 160, 3), dtype=np.uint8) for _ in range(count)]


@pytest.mark.parametrize("use_supervision", [False, True])
def test_replayed_pipeline_output_equals_the_saving_runs_output(tmp_path, use_supervision):
    if use_supervision:
        pytest.importorskip("supervision")
    frames = _frames()
    writer = DetectionCacheWriter(tmp_path / "cache", provenance())
    saved = VisionTrackingPipeline(
        None, detector=ScriptedDetector(), tracker=MultiObjectTracker(use_supervision=use_supervision),
        observation_sink=writer.write_frame,
    )
    saved_output = [saved.process_frame(f, i).to_dict() for i, f in enumerate(frames)]
    writer.close(complete=True)

    plain = VisionTrackingPipeline(None, detector=ScriptedDetector(), tracker=MultiObjectTracker(use_supervision=use_supervision))
    assert [plain.process_frame(f, i).to_dict() for i, f in enumerate(frames)] == saved_output  # sink changes nothing

    source = DetectionCacheReader(tmp_path / "cache").detection_source(0, 0.4)
    replay = VisionTrackingPipeline(
        None, detector=SimpleNamespace(detect=lambda *a, **k: pytest.fail("the detector must not run during replay")),
        tracker=MultiObjectTracker(use_supervision=use_supervision),
    )
    replayed = [replay.process_frame(f, i, detections=source(i)).to_dict() for i, f in enumerate(frames)]
    assert replayed == saved_output
    assert any(r["players"] for r in replayed)


def test_lower_cache_floor_stores_more_but_tracks_exactly_the_same(tmp_path):
    frames = _frames()
    plain = VisionTrackingPipeline(None, detector=ScriptedDetector(0.4), tracker=MultiObjectTracker(use_supervision=False))
    expected = [plain.process_frame(f, i).to_dict() for i, f in enumerate(frames)]

    writer = DetectionCacheWriter(tmp_path / "cache", provenance(floor=0.2, tracking=0.4))
    low = VisionTrackingPipeline(
        None, detector=ScriptedDetector(0.2), tracker=MultiObjectTracker(use_supervision=False),
        observation_sink=writer.write_frame, tracking_confidence=0.4,
    )
    assert [low.process_frame(f, i).to_dict() for i, f in enumerate(frames)] == expected
    writer.close(complete=True)

    reader = DetectionCacheReader(tmp_path / "cache")
    stored = [o for _, _, observations in reader.iter_records() for o in observations]
    assert any(o.confidence == 0.3 for o in stored) and all(o.confidence > 0.2 for o in stored)

    def replay(threshold):
        source = DetectionCacheReader(tmp_path / "cache").detection_source(0, threshold)
        pipe = VisionTrackingPipeline(None, detector=SimpleNamespace(), tracker=MultiObjectTracker(use_supervision=False))
        return [pipe.process_frame(f, i, detections=source(i)).to_dict() for i, f in enumerate(frames)]

    assert replay(0.4) == expected                    # compatibility: the saving run's tracker input
    experimental = replay(0.25)
    assert experimental == replay(0.25)               # deterministic
    assert sum(len(r["players"]) for r in experimental) > sum(len(r["players"]) for r in expected)
    with pytest.raises(DetectionCacheError):          # nothing below the floor can be replayed
        resolve_replay(reader.manifest, "experimental", 0.1)


# --- CLI integration (stub detector, synthetic video) -----------------------------

@pytest.fixture
def cli(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")

    def make_video(name, seed):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 96))
        rng = np.random.default_rng(seed)
        for _ in range(12):
            writer.write(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8))
        writer.release()
        capture = cv2.VideoCapture(str(path))
        usable = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        capture.release()
        if not usable:
            pytest.skip("OpenCV cannot write/read mp4v video here")
        return path

    class StubYOLO(ScriptedDetector):
        def __init__(self, model_path, confidence, device, image_size=640, tile_width=None, half=False, **tiling):
            super().__init__(confidence)

    monkeypatch.setattr(vision_main, "YOLOPlayerDetector", StubYOLO)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"synthetic weights")
    video = make_video("a.mp4", 1)
    other_video = make_video("b.mp4", 2)
    out = tmp_path / "out"

    def run(run_id, *extra, video_path=video, use_weights=True, confidence="0.4"):
        argv = [str(video_path), "--detection-only", "--output-root", str(out), "--run-id", run_id,
                "--tracker", "iou", "--confidence", confidence, *extra]
        if use_weights:
            argv += ["--weights", str(weights)]
        return vision_main.main(argv)

    return SimpleNamespace(run=run, out=out, video=video, other_video=other_video, weights=weights)


def _frames_bytes(cli, run_id):
    return (cli.out / run_id / "frames.jsonl").read_bytes()


def _manifest_of(cli, run_id):
    return json.loads((cli.out / run_id / "manifest.json").read_text(encoding="utf-8"))


def test_cli_default_output_is_unchanged_by_saving_and_replay_reproduces_it(cli):
    assert cli.run("plain") == 0
    assert cli.run("saved", "--save-detections") == 0
    assert _frames_bytes(cli, "saved") == _frames_bytes(cli, "plain")
    assert "detection_cache" not in _manifest_of(cli, "plain")
    saved = _manifest_of(cli, "saved")["detection_cache"]
    assert saved["role"] == "saved" and saved["cache_confidence_floor"] == 0.4 and saved["tracking_confidence"] == 0.4
    assert saved["frames_written"] == 12 and "tracking run" in saved["interpretation"]

    cache = cli.out / "saved" / "detection_cache"
    assert cli.run("replay", "--replay-detections", str(cache), use_weights=False) == 0
    assert _frames_bytes(cli, "replay") == _frames_bytes(cli, "plain")
    replayed = _manifest_of(cli, "replay")
    assert replayed["inputs"]["weights_sha256"] == _manifest_of(cli, "saved")["inputs"]["weights_sha256"]
    assert replayed["inputs"]["video_sha256"] == _manifest_of(cli, "saved")["inputs"]["video_sha256"]
    assert replayed["detection_cache"]["replay"]["mode"] == "compatibility"
    assert replayed["detection_cache"]["manifest_sha256"] == saved["manifest_sha256"]
    assert replayed["configuration"]["device"].startswith("replay")
    assert any("replayed from a cache" in text for text in replayed["limitations"])
    assert cli.run("replay2", "--replay-detections", str(cli.out / "saved"), use_weights=False) == 0  # a run directory also works
    assert _frames_bytes(cli, "replay2") == _frames_bytes(cli, "replay")


def test_cli_replay_can_vary_the_tracker_without_the_detector(cli):
    assert cli.run("saved", "--save-detections") == 0
    cache = str(cli.out / "saved" / "detection_cache")
    assert cli.run("buffered", "--replay-detections", cache, "--track-buffer", "5", use_weights=False) == 0
    assert _manifest_of(cli, "buffered")["configuration"]["tracker_params"]["buffer_frames"] == 5


def test_cli_lower_cache_floor_keeps_the_tracking_input_and_output_of_the_plain_run(cli):
    assert cli.run("plain") == 0
    assert cli.run("low", "--save-detections", "--cache-confidence-floor", "0.2") == 0
    assert _frames_bytes(cli, "low") == _frames_bytes(cli, "plain")  # the lower floor never reaches the tracker

    manifest = _manifest_of(cli, "low")
    assert manifest["configuration"]["confidence_threshold"] == 0.4
    assert manifest["configuration"]["cache_confidence_floor"] == 0.2
    assert manifest["detection_cache"]["cache_confidence_floor"] == 0.2
    assert manifest["detection_cache"]["tracking_confidence"] == 0.4
    cache = json.loads((cli.out / "low" / "detection_cache" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert cache["detector"]["confidence_floor"] == 0.2 and cache["tracking_input"]["confidence_threshold"] == 0.4
    assert cache["counts"]["observations"] > cache["counts"]["tracking_input"]
    stored = [o for _, _, observations in DetectionCacheReader(cli.out / "low").iter_records() for o in observations]
    assert any(o.confidence == 0.3 for o in stored)  # the below-threshold observation was kept in the cache

    # without the flag the cache floor equals the tracking threshold, as before
    assert cli.run("equal", "--save-detections", "--cache-confidence-floor", "0.4") == 0
    assert _frames_bytes(cli, "equal") == _frames_bytes(cli, "plain")
    assert _manifest_of(cli, "equal")["detection_cache"]["cache_confidence_floor"] == 0.4


def test_cli_compatibility_and_experimental_replay_from_a_lower_floor_cache(cli):
    assert cli.run("plain") == 0
    assert cli.run("low", "--save-detections", "--cache-confidence-floor", "0.2") == 0
    cache = str(cli.out / "low" / "detection_cache")

    assert cli.run("compat", "--replay-detections", cache, use_weights=False) == 0
    assert _frames_bytes(cli, "compat") == _frames_bytes(cli, "plain")  # reconstructs the saving run's tracker input
    info = _manifest_of(cli, "compat")["detection_cache"]
    assert info["replay"]["mode"] == "compatibility" and info["replay"]["saving_run_tracking_confidence"] == 0.4
    assert "reconstructs" in info["interpretation"]

    assert cli.run("exp-a", "--replay-detections", cache, "--replay-mode", "experimental",
                   use_weights=False, confidence="0.25") == 0
    assert cli.run("exp-b", "--replay-detections", cache, "--replay-mode", "experimental",
                   use_weights=False, confidence="0.25") == 0
    assert _frames_bytes(cli, "exp-a") == _frames_bytes(cli, "exp-b")  # deterministic
    assert _frames_bytes(cli, "exp-a") != _frames_bytes(cli, "plain")  # the 0.3 observation now reaches the tracker
    experimental = _manifest_of(cli, "exp-a")
    assert experimental["detection_cache"]["replay"]["experimental"] is True
    assert experimental["detection_cache"]["replay"]["tracking_confidence"] == 0.25
    assert experimental["configuration"]["confidence_threshold"] == 0.25
    assert any("Experimental replay" in text for text in experimental["limitations"])

    assert cli.run("exp-high", "--replay-detections", cache, "--replay-mode", "experimental",
                   use_weights=False, confidence="0.5") == 0

    # compatibility replay at another threshold, or any replay below the floor, is refused before output exists
    for run_id, mode, confidence in (("compat-wrong", "compatibility", "0.5"), ("compat-floor", "compatibility", "0.2"),
                                     ("below-c", "compatibility", "0.15"), ("below-e", "experimental", "0.15")):
        assert cli.run(run_id, "--replay-detections", cache, "--replay-mode", mode,
                       use_weights=False, confidence=confidence) == 2
        assert not (cli.out / run_id).exists()


@pytest.mark.parametrize("extra, confidence", [
    (("--cache-confidence-floor", "0.2"), "0.4"),                     # needs --save-detections
    (("--save-detections", "--cache-confidence-floor", "0.5"), "0.4"),  # above the tracking threshold
    (("--save-detections", "--cache-confidence-floor", "0"), "0.4"),
    (("--save-detections", "--cache-confidence-floor", "-0.1"), "0.4"),
    (("--save-detections", "--cache-confidence-floor", "1.5"), "0.4"),
])
def test_cli_rejects_invalid_cache_floor_combinations(cli, extra, confidence):
    assert cli.run("invalid", *extra, confidence=confidence) == 2
    assert not (cli.out / "invalid").exists()


def test_cli_cache_floor_is_rejected_with_a_replay(cli):
    assert cli.run("saved", "--save-detections") == 0
    assert cli.run("replay", "--replay-detections", str(cli.out / "saved"), "--cache-confidence-floor", "0.2",
                   use_weights=False) == 2
    assert not (cli.out / "replay").exists()


def test_cli_cannot_recover_detections_below_the_floor(cli):
    assert cli.run("saved", "--save-detections") == 0
    cache = str(cli.out / "saved" / "detection_cache")
    for mode in ("compatibility", "experimental"):
        assert cli.run(f"below-{mode}", "--replay-detections", cache, "--replay-mode", mode,
                       use_weights=False, confidence="0.3") == 2
        assert not (cli.out / f"below-{mode}").exists()


def test_cli_refuses_mismatched_inputs_before_creating_output(cli, tmp_path):
    assert cli.run("saved", "--save-detections") == 0
    assert cli.run("short", "--save-detections", "--max-frames", "6") == 0
    cache = str(cli.out / "saved" / "detection_cache")
    other_weights = tmp_path / "other.pt"
    other_weights.write_bytes(b"different weights")
    cases = {
        "wrong-video": dict(video_path=cli.other_video),
        "beyond-cache": dict(cache="short"),
        "wrong-weights": dict(extra=("--weights", str(other_weights))),
        "detector-flag": dict(extra=("--tile-width", "640")),
        "imgsz-flag": dict(extra=("--imgsz", "1920")),
        "both-flags": dict(extra=("--save-detections",)),
    }
    for run_id, case in cases.items():
        result = cli.run(run_id, "--replay-detections",
                         str(cli.out / case["cache"] / "detection_cache") if "cache" in case else cache, *case.get("extra", ()),
                         video_path=case.get("video_path", cli.video), use_weights=False)
        assert result == 2, run_id
        assert not (cli.out / run_id).exists(), run_id
    assert cli.run("no-weights", use_weights=False) == 2  # weights stay mandatory without a replay


def test_cli_refuses_a_corrupted_cache(cli):
    assert cli.run("saved", "--save-detections") == 0
    records = cli.out / "saved" / "detection_cache" / RECORDS_FILENAME
    records.write_bytes(records.read_bytes().replace(b"0.9", b"0.8", 1))
    assert cli.run("replay", "--replay-detections", str(records.parent), use_weights=False) == 2
    assert not (cli.out / "replay").exists()


def test_cli_replay_window_inside_the_cache_is_allowed(cli):
    assert cli.run("saved", "--save-detections") == 0
    cache = str(cli.out / "saved" / "detection_cache")
    assert cli.run("inside", "--replay-detections", cache, "--start-frame", "4", "--max-frames", "6", use_weights=False) == 0
    records = [json.loads(line) for line in (cli.out / "inside" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["frame_num"] for r in records] == list(range(4, 10))


def test_cli_failed_saving_run_leaves_a_cache_that_cannot_be_replayed(cli, monkeypatch):
    class Failing(vision_main.YOLOPlayerDetector):
        def observe(self, frame_rgb, pitch_mask_polygon=None):
            if self.calls == 5:
                raise RuntimeError("synthetic failure")
            return super().observe(frame_rgb, pitch_mask_polygon)

    monkeypatch.setattr(vision_main, "YOLOPlayerDetector", Failing)
    assert cli.run("broken", "--save-detections") == 1
    manifest = json.loads((cli.out / "broken" / "detection_cache" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete" and "synthetic failure" in manifest["error"]
    assert cli.run("replay", "--replay-detections", str(cli.out / "broken"), use_weights=False) == 2


def test_cli_manifest_records_the_class_safety_policy_and_replay_matches_it(cli):
    assert cli.run("stable", "--save-detections") == 0
    assert cli.run("latest", "--role-policy", "latest") == 0
    stable, latest = _manifest_of(cli, "stable"), _manifest_of(cli, "latest")
    assert stable["configuration"]["class_safety"]["role_policy"] == "stable"
    assert latest["configuration"]["class_safety"]["role_policy"] == "latest"
    assert "ball is never submitted" in stable["configuration"]["class_safety"]["person_tracker_inputs"]
    assert stable["summary"]["ball_detections_not_tracked"] == 12 and "role_changes" in stable["summary"]["role_statistics"]
    assert cli.run("replay", "--replay-detections", str(cli.out / "stable" / "detection_cache"), use_weights=False) == 0
    assert _frames_bytes(cli, "replay") == _frames_bytes(cli, "stable")


def test_cli_tiling_options_are_validated_recorded_and_fixed_by_a_cache(cli):
    for run_id, extra in (("h", ("--tile-height", "300")), ("b", ("--tile-batch", "2")), ("m", ("--tile-merge", "person_agnostic")),
                          ("small", ("--tile-width", "640", "--tile-height", "32")), ("zero", ("--tile-width", "640", "--tile-batch", "0")),
                          ("iou", ("--tile-width", "640", "--tile-merge-cross-role-iou", "0.4"))):
        assert cli.run(run_id, *extra) == 2
        assert not (cli.out / run_id).exists()
    assert cli.run("tiled", "--tile-width", "640", "--tile-height", "480", "--tile-batch", "2", "--save-detections") == 0
    manifest = _manifest_of(cli, "tiled")
    assert manifest["configuration"]["tile_width"] == 640 and manifest["configuration"]["tile_height"] == 480
    assert manifest["configuration"]["tile_batch"] == 2 and manifest["configuration"]["tile_merge"]["policy"] == "class_aware"
    assert manifest["summary"]["detector_merge_statistics"] == {}
    cache = json.loads((cli.out / "tiled" / "detection_cache" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert cache["detector"]["tile_height"] is None  # the stub detector describes itself; real detectors record their own
    assert cache["detector"]["merge_statistics"] == {}
    cache_dir = str(cli.out / "tiled" / "detection_cache")
    for run_id, extra in (("r-h", ("--tile-height", "300")), ("r-b", ("--tile-batch", "2")),
                          ("r-m", ("--tile-merge", "person_agnostic")), ("r-i", ("--tile-merge-cross-role-iou", "0.9"))):
        assert cli.run(run_id, "--replay-detections", cache_dir, *extra, use_weights=False) == 2
        assert not (cli.out / run_id).exists()
    assert cli.run("replayed", "--replay-detections", cache_dir, use_weights=False) == 0
    assert _manifest_of(cli, "replayed")["summary"]["detector_merge_statistics"] is None
