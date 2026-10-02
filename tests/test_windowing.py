import numpy as np
import pytest

from src.config import Config
from src.errors import DataError
from src.windowing import (
    bandpass,
    resample_to_fs,
    segment,
    signal_to_windows,
    zscore_windows,
)


def test_bandpass_removes_dc_and_keeps_10hz():
    fs = 130
    t = np.arange(0, 10, 1 / fs)
    sig = 5.0 + np.sin(2 * np.pi * 10 * t)
    out = bandpass(sig, fs, 0.5, 40.0)
    assert abs(out.mean()) < 0.05
    assert out.std() == pytest.approx(np.sin(2 * np.pi * 10 * t).std(), rel=0.1)


def test_resample_500_to_130_length():
    sig = np.random.default_rng(0).normal(size=5000)  # 10 s a 500 Hz
    out = resample_to_fs(sig, 500, 130)
    assert len(out) == 1300


def test_resample_same_rate_is_identity():
    sig = np.arange(10.0)
    assert np.array_equal(resample_to_fs(sig, 130, 130), sig)


def test_segment_overlap_50_percent():
    sig = np.arange(650 * 3)
    w = segment(sig, 650, 0.5)
    assert w.shape == (5, 650)            # pasos de 325: 0, 325, 650, 975, 1300
    assert w[1, 0] == 325


def test_segment_short_signal_gives_zero_windows():
    assert segment(np.zeros(100), 650, 0.5).shape == (0, 650)


def test_zscore_windows():
    w = np.random.default_rng(0).normal(5, 3, size=(4, 650))
    z = zscore_windows(w)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_signal_to_windows_shape_dtype():
    cfg = Config()
    sig = np.random.default_rng(0).normal(size=130 * 20)
    w = signal_to_windows(sig, 130, cfg)
    assert w.ndim == 3 and w.shape[1:] == (1, 650)
    assert w.dtype == np.float32


# --- Review Focus 3: grabaciones problemáticas ---
def test_flat_segment_windows_are_dropped_not_nan():
    cfg = Config()
    rng = np.random.default_rng(1)
    good = rng.normal(size=130 * 10)
    flat = np.zeros(130 * 10)                 # banda desconectada: línea plana
    w = signal_to_windows(np.concatenate([good, flat]), 130, cfg)
    assert np.isfinite(w).all()
    assert len(w) < len(segment(np.zeros(130 * 20), 650, 0.5))


def test_nan_in_recording_raises():
    cfg = Config()
    sig = np.random.default_rng(2).normal(size=130 * 10)
    sig[100] = np.nan
    with pytest.raises(DataError, match="NaN"):
        signal_to_windows(sig, 130, cfg)


def test_recording_shorter_than_one_window_gives_zero_windows():
    cfg = Config()
    sig = np.random.default_rng(3).normal(size=130 * 2)   # 2 s < 5 s
    w = signal_to_windows(sig, 130, cfg)
    assert w.shape == (0, 1, 650)
