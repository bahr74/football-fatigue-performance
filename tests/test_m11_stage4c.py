"""Stage 4C unit tests (synthetic data only; no project datasets required)."""

import numpy as np
import pytest

from football_fatigue.m11_benchmark import shiftaug as SA
from football_fatigue.m11_benchmark.stage4c import draw_rows
from football_fatigue.window_intersections import samples as S


def _pool(n0=400, seed=0):
    rng = np.random.default_rng(seed)
    n_aug = rng.integers(0, 7, n0)
    V = np.full((n0, 6), -1)
    owner, r = [], 0
    for i in range(n0):
        for k in range(n_aug[i]):
            V[i, k] = r
            owner.append(i)
            r += 1
    return n_aug, V, np.array(owner)


@pytest.mark.parametrize("config", ["B", "C"])
def test_draw_rows_one_row_per_encounter_of_its_own(config):
    n_aug, V, owner = _pool()
    n0 = len(n_aug)
    cands = np.arange(0, n0, 2)
    rows = draw_rows(config, cands, n_aug, V, n0, np.random.default_rng(1))
    assert rows.shape == cands.shape
    cen = rows < n0
    assert (rows[cen] == cands[cen]).all()                                   # centred rows are the encounter itself
    assert (owner[rows[~cen] - n0] == cands[~cen]).all()                     # shifted rows belong to the same encounter
    assert (cen[n_aug[cands] == 0]).all()                                    # no valid shift -> centred


def test_draw_rows_mixture_proportions():
    n_aug, V, _ = _pool(4000)
    n0 = len(n_aug)
    cands = np.flatnonzero(n_aug == 6)
    rb = np.concatenate([draw_rows("B", cands, n_aug, V, n0, np.random.default_rng(s)) for s in range(20)])
    rc = np.concatenate([draw_rows("C", cands, n_aug, V, n0, np.random.default_rng(s)) for s in range(20)])
    assert abs((rb < n0).mean() - 1 / 7) < 0.02                              # B: uniform over centred + 6 shifts
    assert abs((rc < n0).mean() - 0.5) < 0.03                                # C: half centred


def test_draw_rows_centred_configs_unchanged():
    cands = np.array([3, 1, 2])
    for cf in ("A", "Anoise", "Asmooth"):
        assert (draw_rows(cf, cands, None, None, 10, np.random.default_rng(0)) == cands).all()


def test_shift_order_tables_distinct_and_order_bits_vary():
    names = [SA.table_name(s) for s in SA.SHIFTS]
    assert len(set(names)) == len(SA.SHIFTS) and 0.0 not in SA.SHIFTS
    bits = {tuple(S.order_bits(f"cand{i}", n) for n in names) for i in range(200)}
    assert len(bits) > 150                                                   # order is not a function of the shift alone
