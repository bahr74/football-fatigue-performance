"""Verify the installed package contract without dataset access."""

from importlib import import_module
from importlib.metadata import distribution


def test_installed_distribution_is_importable():
    package = import_module("football_fatigue")
    installed = distribution("football-fatigue")
    assert package.__file__ is not None
    assert installed.metadata["Name"] == "football-fatigue"
