"""M13: the GPU pilot stays refused without approval and fails closed when the GPU state cannot be read."""

from __future__ import annotations

import subprocess

from football_fatigue.m11_fair import gpu_pilot as G


def test_gpu_busy_fails_closed_when_nvidia_smi_is_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", boom)
    assert G.gpu_busy() and "unknown" in G.gpu_busy()[0]


def test_gpu_busy_fails_closed_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 9, stdout="", stderr="err"))
    assert G.gpu_busy() and "unknown" in G.gpu_busy()[0]


def test_gpu_busy_empty_when_no_compute_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="\n", stderr=""))
    assert G.gpu_busy() == []


def test_pilot_plan_is_the_m12_package():
    e = G.estimate()
    assert len(e["configs"]) == 7 and e["epoch_cap"] == 400 and e["patience"] == 25
    assert e["gpu_hours_worst_case"] == 2.51 and e["gpu_hours_if_best_epoch_about_225"] == 1.57
