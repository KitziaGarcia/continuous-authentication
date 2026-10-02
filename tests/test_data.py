import numpy as np
import pytest

from src.data import (
    WindowTable,
    concat_tables,
    load_window_dir,
    save_window_file,
    table_fingerprint,
)
from src.errors import DataError
from src.synthetic import make_synthetic_table


def test_validate_rejects_wrong_window_length():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2)
    bad = WindowTable(t.windows[:, :, :100], t.participant_ids, t.session_ids, t.activities, "synthetic")
    with pytest.raises(DataError, match="650"):
        bad.validate()


def test_validate_rejects_nan():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2)
    t.windows[0, 0, 0] = np.nan
    with pytest.raises(DataError, match="NaN"):
        t.validate()


def test_subset_and_persons():
    t = make_synthetic_table(n_people=3, n_sessions=2, windows_per_session=2)
    only_p1 = t.subset(t.participant_ids == "p1")
    assert only_p1.persons() == ["p1"]
    assert len(only_p1) == 4


def test_concat_tables():
    a = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2, prefix="a")
    b = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3, prefix="b")
    c = concat_tables([a, b])
    assert len(c) == 5 and c.persons() == ["a0", "b0"]


def test_save_and_load_roundtrip(tmp_path):
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3, seed=3)
    save_window_file(tmp_path / "p0__s1.npz", t.windows, "p0", "s1", "rest", "synthetic")
    back = load_window_dir(tmp_path)
    assert back.persons() == ["p0"] and back.sessions_of("p0") == ["s1"]
    assert back.source == "synthetic"
    assert np.allclose(back.windows, t.windows)


def test_load_empty_dir_raises(tmp_path):
    with pytest.raises(DataError, match="npz"):
        load_window_dir(tmp_path)


def test_fingerprint_changes_when_data_changes():
    a = make_synthetic_table(n_people=2, n_sessions=2, windows_per_session=3)
    b = make_synthetic_table(n_people=2, n_sessions=2, windows_per_session=4)
    assert table_fingerprint(a) == table_fingerprint(a)
    assert table_fingerprint(a) != table_fingerprint(b)


from src.data import (
    PolarSplit,
    assert_disjoint_people,
    assert_no_holdout_leak,
    split_polar,
    split_public,
)
from src.errors import LeakageError


def _polar(n_people=5, n_sessions=3):
    return make_synthetic_table(n_people=n_people, n_sessions=n_sessions, windows_per_session=4, seed=11)


def test_split_polar_holdout_people_are_complete_and_separate():
    t = _polar()
    s = split_polar(t, holdout_ids=("p3", "p4"))
    assert isinstance(s, PolarSplit)
    assert s.holdout.persons() == ["p3", "p4"]
    assert set(s.train.persons()) == {"p0", "p1", "p2"}
    assert set(s.val.persons()) == {"p0", "p1", "p2"}
    assert len(s.holdout) == 2 * 3 * 4          # todas las sesiones de los hold-out


def test_split_polar_last_session_is_validation():
    s = split_polar(_polar(), holdout_ids=("p3", "p4"))
    assert set(s.val.session_ids.tolist()) == {"s3"}
    assert set(s.train.session_ids.tolist()) == {"s1", "s2"}


def test_split_polar_never_loses_or_duplicates_windows():
    t = _polar()
    s = split_polar(t, holdout_ids=("p3", "p4"))
    assert len(s.train) + len(s.val) + len(s.holdout) == len(t)


# --- Review Focus 1: una sola sesión ---
def test_person_with_single_session_raises_and_names_person():
    t = _polar()
    keep = ~((t.participant_ids == "p1") & (t.session_ids != "s1"))
    t = t.subset(keep)
    with pytest.raises(DataError, match="p1"):
        split_polar(t, holdout_ids=("p3", "p4"))


# --- Review Focus 2: hold-out inexistente (typo) ---
def test_unknown_holdout_id_raises():
    with pytest.raises(DataError, match="P7"):
        split_polar(_polar(), holdout_ids=("p3", "P7"))


def test_assert_no_holdout_leak_detects_leak():
    t = _polar()
    with pytest.raises(LeakageError, match="p4"):
        assert_no_holdout_leak(("p4",), {"train": t})


def test_assert_no_holdout_leak_passes_when_clean():
    s = split_polar(_polar(), holdout_ids=("p3", "p4"))
    assert_no_holdout_leak(("p3", "p4"), {"train": s.train, "val": s.val})


def test_assert_disjoint_people():
    a = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="x")
    b = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="x")
    with pytest.raises(LeakageError):
        assert_disjoint_people({"a": a, "b": b})
    c = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="y")
    assert_disjoint_people({"a": a, "c": c})


def test_split_public_is_by_person_and_deterministic():
    t = make_synthetic_table(n_people=10, n_sessions=1, windows_per_session=3, prefix="pub_")
    tr, va = split_public(t, val_fraction=0.2, seed=0)
    assert set(tr.persons()).isdisjoint(va.persons())
    assert len(va.persons()) == 2
    tr2, va2 = split_public(t, val_fraction=0.2, seed=0)
    assert va.persons() == va2.persons()


def test_split_public_needs_enough_people():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, prefix="pub_")
    with pytest.raises(DataError):
        split_public(t, val_fraction=0.5, seed=0)


import dataclasses

import torch

from src.config import Config
from src.data import Augmenter, WindowDataset, make_label_map, make_loader


def _window():
    return make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=1, seed=4).windows[0]


def test_augmenter_output_shape_dtype_and_zscore():
    out = Augmenter(Config(), seed=0)(_window())
    assert out.shape == (1, 650) and out.dtype == np.float32
    assert abs(out.mean()) < 1e-3 and out.std() == pytest.approx(1.0, abs=1e-2)


def test_augmenter_with_all_strengths_zero_is_identity():
    cfg = dataclasses.replace(
        Config(), jitter_std=0, magnitude_warp=0, warp_strength=0, wander_amplitude=0
    )
    w = _window()
    assert np.allclose(Augmenter(cfg, seed=0)(w), w, atol=1e-4)


def test_augmenter_changes_the_window_and_is_seeded():
    w = _window()
    a = Augmenter(Config(), seed=1)(w)
    b = Augmenter(Config(), seed=1)(w)
    c = Augmenter(Config(), seed=2)(w)
    assert not np.allclose(a, w, atol=1e-3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_label_map_is_sorted_and_contiguous():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=2)
    assert make_label_map(t) == {"p0": 0, "p1": 1, "p2": 2}


def test_dataset_items_and_loader_batches():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=4)
    lm = make_label_map(t)
    ds = WindowDataset(t, lm, augmenter=Augmenter(Config(), seed=0))
    x, y = ds[0]
    assert x.shape == (1, 650) and x.dtype == torch.float32 and y == 0
    xb, yb = next(iter(make_loader(ds, batch_size=5, shuffle=True, seed=0)))
    assert xb.shape == (5, 1, 650) and yb.shape == (5,)


def test_loader_shuffle_is_reproducible():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=4)
    ds = WindowDataset(t, make_label_map(t))
    a = next(iter(make_loader(ds, 12, True, seed=3)))[1]
    b = next(iter(make_loader(ds, 12, True, seed=3)))[1]
    assert torch.equal(a, b)


# --- Revisión final: contrato z-score y orden de sesiones ---
def test_validate_rejects_windows_that_are_not_zscored():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3)
    raw = WindowTable(t.windows * 500.0 + 3000.0, t.participant_ids, t.session_ids, t.activities, "polar")
    with pytest.raises(DataError, match="z-score"):
        raw.validate()


def test_validate_rejects_session_ids_that_do_not_sort_chronologically():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3)
    for ids in (["s1", "s2", "s10"], ["s10", "s9", "s1"]):
        n = len(ids)
        bad = WindowTable(t.windows[:n], t.participant_ids[:n], np.array(ids), t.activities[:n], "polar")
        with pytest.raises(DataError, match="ceros"):
            bad.validate()


def test_validate_accepts_zero_padded_and_iso_session_ids():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3)
    for ids in (["s01", "s02", "s10"], ["2026-10-01_s1", "2026-10-03_s2", "2026-11-02_s3"]):
        ok = WindowTable(t.windows[:3], t.participant_ids[:3], np.array(ids), t.activities[:3], "polar")
        ok.validate()
