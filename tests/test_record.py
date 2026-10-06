from datetime import datetime

import numpy as np
import pytest

from src.config import Config
from src.data import load_window_dir
from src.errors import DataError
from src.record import (
    default_session_id,
    main,
    plot_recording,
    rate_warning,
    save_recording,
)
from src.synthetic import make_synthetic_signal


def test_synthetic_signal_has_requested_length_and_is_finite():
    sig = make_synthetic_signal(seconds=10, fs=130, seed=0)
    assert sig.shape == (1300,) and np.isfinite(sig).all()
    assert not np.array_equal(sig, make_synthetic_signal(10, 130, seed=1))


def test_save_recording_writes_raw_and_loadable_windows(tmp_path):
    raw = make_synthetic_signal(seconds=60, fs=130, seed=0) * 400 + 80   # como µV con desfase
    info = save_recording(raw, 130, "ana", "2026-10-01_0930", "rest",
                          tmp_path / "raw", tmp_path / "proc", Config())
    assert info["n_samples"] == len(raw) and info["duration_s"] == pytest.approx(60.0)
    assert info["n_windows"] == 23                      # 60 s, ventanas de 5 s con 50 % de traslape
    assert np.array_equal(np.load(info["raw_path"]), raw)
    t = load_window_dir(tmp_path / "proc")              # pasa validate(): z-score y forma correctos
    assert t.persons() == ["ana"] and t.sessions_of("ana") == ["2026-10-01_0930"]
    assert len(t) == 23 and t.source == "polar"


def test_save_recording_rejects_recordings_with_no_usable_window(tmp_path):
    raw = make_synthetic_signal(seconds=3, fs=130, seed=0)           # < 5 s
    with pytest.raises(DataError, match="muy corta"):
        save_recording(raw, 130, "ana", "s1", "rest", tmp_path / "r", tmp_path / "p", Config())


def test_save_recording_rejects_flat_signal(tmp_path):
    with pytest.raises(DataError, match="ventanas"):
        save_recording(np.zeros(130 * 30), 130, "ana", "s1", "rest", tmp_path / "r", tmp_path / "p", Config())


def test_save_recording_refuses_to_overwrite_an_existing_session(tmp_path):
    raw = make_synthetic_signal(seconds=20, fs=130, seed=0)
    save_recording(raw, 130, "ana", "s1", "rest", tmp_path / "r", tmp_path / "p", Config())
    with pytest.raises(FileExistsError):
        save_recording(raw, 130, "ana", "s1", "rest", tmp_path / "r", tmp_path / "p", Config())


def test_plot_recording_writes_a_png(tmp_path):
    raw = make_synthetic_signal(seconds=30, fs=130, seed=0)
    out = plot_recording(raw, 130, tmp_path / "plot.png", "ana", "s1", n_windows=5)
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_default_session_id_sorts_chronologically():
    a = default_session_id(datetime(2026, 10, 1, 9, 30))
    b = default_session_id(datetime(2026, 10, 1, 14, 5))
    c = default_session_id(datetime(2026, 10, 3, 8, 0))
    assert a == "2026-10-01_0930" and sorted([c, b, a]) == [a, b, c]


def test_rate_warning_only_when_far_from_nominal():
    assert rate_warning(130.4) is None
    assert rate_warning(120.0) is None                     # dentro de ±10 %
    assert "130" in rate_warning(90.0)


def test_cli_simulate_end_to_end(tmp_path):
    rc = main(["--id", "ana", "--seconds", "20", "--simulate", "--no-open",
               "--raw-dir", str(tmp_path / "raw"), "--processed-dir", str(tmp_path / "proc"),
               "--plots-dir", str(tmp_path / "plots")])
    assert rc == 0
    assert len(list((tmp_path / "plots").glob("*.png"))) == 1
    assert len(load_window_dir(tmp_path / "proc")) > 0


def test_save_recording_tag_goes_into_both_file_names_so_activities_can_share_a_session(tmp_path):
    raw = make_synthetic_signal(seconds=30, fs=130, seed=0)
    a = save_recording(raw, 130, "P01", "2026-10-06", "reposo", tmp_path / "r", tmp_path / "p", Config(), tag="reposo")
    b = save_recording(raw, 130, "P01", "2026-10-06", "lectura", tmp_path / "r", tmp_path / "p", Config(), tag="lectura")
    assert a["raw_path"].name == "P01__2026-10-06__reposo.npy"
    assert a["windows_path"].name == "P01__2026-10-06__reposo.npz"
    assert b["windows_path"].name == "P01__2026-10-06__lectura.npz"
    t = load_window_dir(tmp_path / "p")
    assert t.sessions_of("P01") == ["2026-10-06"] and set(t.activities.tolist()) == {"reposo", "lectura"}
