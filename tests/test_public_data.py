import numpy as np

import src.public_data as pd_
from src.config import Config
from src.data import load_window_dir


def _fake_records():
    rng = np.random.default_rng(0)
    fs = 500.0
    t = np.arange(0, 20, 1 / fs)
    for person in ("Person_01", "Person_02"):
        for rec in ("rec_1", "rec_2", "rec_10"):
            beat = np.sin(2 * np.pi * 1.2 * t) ** 21           # picos tipo R a ~72 lpm
            yield person, rec, beat + rng.normal(0, 0.02, len(t)), fs


def test_build_public_windows_writes_loadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: _fake_records())
    n = pd_.build_public_windows("ignored", tmp_path, Config())
    assert n == 6
    t = load_window_dir(tmp_path)
    assert t.persons() == ["pub_Person_01", "pub_Person_02"]
    assert t.source == "public"
    assert t.windows.shape[1:] == (1, 650)
    # 20 s a 130 Hz con ventanas de 5 s y 50 % de traslape -> 7 ventanas por registro
    assert len(t) == 6 * 7


def test_session_ids_are_zero_padded_so_they_sort_chronologically(tmp_path, monkeypatch):
    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: _fake_records())
    pd_.build_public_windows("ignored", tmp_path, Config())
    t = load_window_dir(tmp_path)
    assert t.sessions_of("pub_Person_01") == ["rec_01", "rec_02", "rec_10"]


def test_records_too_short_are_skipped(tmp_path, monkeypatch):
    def short():
        yield "Person_01", "rec_1", np.random.default_rng(0).normal(size=500 * 2), 500.0

    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: short())
    assert pd_.build_public_windows("ignored", tmp_path, Config()) == 0
