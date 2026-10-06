import numpy as np
import pytest

from src.synthetic import make_synthetic_signal, make_synthetic_table


def test_shape_and_metadata():
    t = make_synthetic_table(n_people=3, n_sessions=2, windows_per_session=5, seed=1)
    assert t.windows.shape == (3 * 2 * 5, 1, 650)
    assert t.windows.dtype == np.float32
    assert t.persons() == ["p0", "p1", "p2"]
    assert t.sessions_of("p0") == ["s1", "s2"]


def test_windows_are_zscored():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=4, seed=2)
    assert np.allclose(t.windows.mean(axis=2), 0, atol=1e-4)
    assert np.allclose(t.windows.std(axis=2), 1, atol=1e-3)


def test_same_seed_same_data_different_seed_different_data():
    a = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=5)
    b = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=5)
    c = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=6)
    assert np.array_equal(a.windows, b.windows)
    assert not np.array_equal(a.windows, c.windows)


def test_prefix_changes_ids():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="pub_")
    assert t.persons() == ["pub_0", "pub_1"]


def test_synthetic_signal_heart_rate_and_irregularity_are_controllable():
    from scipy.signal import find_peaks

    def rr_stats(sig):
        pk, _ = find_peaks(sig, distance=39, prominence=0.5)
        rr = np.diff(pk) / 130
        return 60 / np.median(rr), rr.std() / rr.mean()

    hr, var = rr_stats(make_synthetic_signal(40, 130, seed=1, hr_bpm=100))
    assert hr == pytest.approx(100, abs=4) and var < 0.05
    _, var_irregular = rr_stats(make_synthetic_signal(40, 130, seed=1, hr_bpm=70, rr_jitter=0.5))
    assert var_irregular > 0.15


def test_long_synthetic_signal_is_generated_quickly():
    import time

    t0 = time.time()
    sig = make_synthetic_signal(900, 130, seed=0)       # una sentada simulada necesita ~6 min de señal
    assert len(sig) == 900 * 130
    assert time.time() - t0 < 3.0                       # antes ~10 s (cada latido recorría toda la señal)
