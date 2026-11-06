from football_fatigue.multicam_lab.timeline import CameraClock, master_grid, observation_state, resolve


def test_offset_camera_resolves_shifted_frame():
    fixed = CameraClock("fixed", 30.0, 1000, offset_s=-0.1)
    r = resolve(fixed, 10.0)
    assert r.status == "ok" and r.index == 297            # 9.9 s * 30
    assert abs(r.residual_s) < 1e-9
    assert abs(r.requested_source_s - 9.9) < 1e-9


def test_residual_is_reported_and_within_half_frame():
    cam = CameraClock("moving", 30.0, 1000)
    r = resolve(cam, 10.01)
    assert r.status == "ok" and r.index == 300 and abs(r.residual_s + 0.01) < 1e-9


def test_out_of_file_is_unavailable_not_clamped():
    cam = CameraClock("moving", 30.0, 100)
    assert resolve(cam, 10.0).status == "unavailable"
    assert resolve(cam, -1.0).index is None


def test_tolerance_violation_is_unavailable():
    cam = CameraClock("fixed", 5.0, 100)                     # 0.2 s frames
    r = resolve(cam, 1.09, tolerance_s=0.05)
    assert r.status == "unavailable" and r.index == 5


def test_observation_state():
    assert observation_state(1.0, 1.0, 0.017) == "current"
    assert observation_state(0.8, 1.0, 0.017) == "stale"
    assert observation_state(None, 1.0, 0.017) == "missing"


def test_master_grid_has_no_float_drift():
    g = master_grid(1825.0, 1835.0, 0.2)
    assert len(g) == 51 and g[-1] == 1835.0 and g[25] == 1830.0
