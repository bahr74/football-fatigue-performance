"""Entry point: ``python -m football_fatigue.interaction_catalogue`` (implementation in ``run.py``, importable for multiprocessing)."""

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
