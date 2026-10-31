"""The phase fix replaces only near-zero phasors far from the order's modal phase."""

import numpy as np

import train as T


def test_fix_replaces_outlying_phases_of_near_zero_magnitudes_only():
    rng = np.random.default_rng(0)
    a = np.full((50, 4, 6), 1.0 * np.exp(0.3j), dtype=np.complex128)   # modal phase 0.3 at every order
    a[:, :, 2] *= 1 + 0.01 * rng.standard_normal((50, 4))            # order 3 (index 2) jitters in magnitude
    a[7, 1, 2] = 1e-12 * np.exp(2.5j)     # near zero, 2.2 rad off: replaced
    a[9, 3, 2] = 1e-12 * np.exp(0.5j)     # near zero, 0.2 rad off: kept
    a[11, 0, 2] = 5.0 * np.exp(2.5j)      # large: kept
    a[13, 2, 3] = 1e-12 * np.exp(2.5j)    # order 4 is not listed: kept
    before = a.copy()
    out, n = T.fix_phase_angles(a, harmonics=[2])
    assert out is a and n == 1
    assert np.isclose(np.angle(a[7, 1, 2]), 0.3) and np.isclose(abs(a[7, 1, 2]), 1e-12)
    # the rewritten order is recomputed from magnitude and phase (as theirs is): rounding only
    changed = np.argwhere(np.abs(a - before) > 1e-12 * np.abs(before))
    assert changed.tolist() == [[7, 1, 2]]
