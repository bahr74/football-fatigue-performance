import av
import pytest

from football_fatigue.multicam_lab.media import parse_range, probe_video


def _encode(path, n=30, valid_pts=True):
    out = av.open(str(path), "w"); s = out.add_stream("mpeg4", rate=30); s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    for i in range(n):
        f = av.VideoFrame(64, 48, "yuv420p"); f.pts = i if valid_pts else None
        for p in s.encode(f):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()


def test_probe_accepts_valid_clip(tmp_path):
    p = tmp_path / "ok.mp4"; _encode(p)
    info = probe_video(p)
    assert info["width"] == 64 and abs(info["fps"] - 30) < 0.5 and 0.8 < info["duration_s"] < 1.2


def test_probe_rejects_non_video_content(tmp_path):
    p = tmp_path / "fake.mp4"; p.write_text("this is not a video")
    with pytest.raises(ValueError, match="not a recognised video container"):
        probe_video(p)


def test_probe_rejects_wrong_extension(tmp_path):
    p = tmp_path / "x.txt"; p.write_text("x")
    with pytest.raises(ValueError, match="unsupported file extension"):
        probe_video(p)


def test_probe_rejects_broken_timestamps(tmp_path):
    p = tmp_path / "zero_pts.mp4"; _encode(p, valid_pts=False)
    try:
        info = probe_video(p)
    except ValueError as exc:
        assert "timestamps" in str(exc)
    else:                                   # some muxers repair missing pts; then timing must be plausible
        assert info["fps"] <= 240 and info["duration_s"] > 0


def test_parse_range():
    assert parse_range("bytes=0-99", 1000) == (0, 99)
    assert parse_range("bytes=900-", 1000) == (900, 999)
    assert parse_range(None, 1000) is None
