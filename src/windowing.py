"""Preprocesamiento mínimo: filtro, remuestreo, ventanas y z-score.

Es un reemplazo sencillo hasta que exista el módulo completo de preprocesamiento (con
detección de picos R y calidad de señal). Sirve para ECG-ID y para grabaciones del Polar.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.config import Config
from src.errors import DataError


def bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Pasa-banda de fase cero. Quita deriva de línea base (<0.5 Hz) y ruido muscular (>40 Hz)."""
    if high >= fs / 2:
        raise DataError(f"La frecuencia de corte {high} Hz excede Nyquist ({fs / 2} Hz)")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def resample_to_fs(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(sig, ratio.numerator, ratio.denominator)


def segment(sig: np.ndarray, window_samples: int, overlap: float) -> np.ndarray:
    step = max(1, int(round(window_samples * (1 - overlap))))
    starts = range(0, len(sig) - window_samples + 1, step)
    if len(starts) == 0:
        return np.zeros((0, window_samples))
    return np.stack([sig[s : s + window_samples] for s in starts])


def zscore_windows(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Media 0 y desviación 1 por ventana: la red ve la FORMA, no la amplitud absoluta."""
    return (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + eps)


def signal_to_windows(sig: np.ndarray, fs_in: float, cfg: Config) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    if not np.isfinite(sig).all():
        raise DataError("La grabación contiene NaN o inf; hay que limpiarla o recortarla antes")
    # Se filtra a la frecuencia ORIGINAL y luego se remuestrea (si no, el filtro anti-alias
    # de remuestrear descartaría banda útil o dejaría pasar ruido).
    raw = resample_to_fs(sig, fs_in, cfg.fs)
    filt = resample_to_fs(bandpass(sig, fs_in, cfg.band_low_hz, cfg.band_high_hz), fs_in, cfg.fs)
    w = segment(filt, cfg.window_samples, cfg.window_overlap)
    if len(w):
        # Ventanas planas = banda desconectada; z-score las convertiría en ruido. Se detectan en la
        # señal CRUDA porque el filtro deja "ecos" que hacen que una zona plana no salga con std 0.
        # (La calidad de señal más fina corresponde al módulo de preprocesamiento futuro.)
        raw_w = segment(raw, cfg.window_samples, cfg.window_overlap)
        w = w[raw_w.std(axis=1) > 1e-6]
    w = zscore_windows(w) if len(w) else w
    return w[:, None, :].astype(np.float32)
