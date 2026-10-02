import numpy as np

from src.synthetic import make_synthetic_table


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
