import numpy as np
import pytest

from src.quality import check_quality
from src.synthetic import make_synthetic_signal

FS = 130


def _good(seconds=30, seed=0, **kw):
    return make_synthetic_signal(seconds, FS, seed=seed, **kw) * 400 + 50     # como µV con desfase


def test_clean_ecg_passes():
    r = check_quality(_good(hr_bpm=72))
    assert r.ok and r.issues == []
    assert r.hr_bpm == pytest.approx(72, abs=4) and r.rr_variation < 0.10 and r.spike_ratio < 2


def test_short_segment_of_30s_passes():
    assert check_quality(_good(seconds=30, seed=3, hr_bpm=85)).ok


def test_flat_signal_fails_with_plain_message():
    r = check_quality(np.zeros(FS * 30))
    assert not r.ok and any("plana" in i for i in r.issues)


def test_noise_only_fails():
    r = check_quality(np.random.default_rng(0).normal(0, 300, FS * 30))
    assert not r.ok and any("latidos" in i.lower() or "ritmo" in i.lower() for i in r.issues)


def test_large_spike_artifact_fails():
    raw = _good(hr_bpm=75)
    raw[1500:1506] += 15000                    # como el glitch real: miles de µV por encima de lo normal
    r = check_quality(raw)
    assert not r.ok and any("pico" in i.lower() and "artefacto" in i for i in r.issues)
    assert r.spike_ratio > 3


def test_very_irregular_rhythm_fails():
    r = check_quality(_good(hr_bpm=70, rr_jitter=0.5))
    assert not r.ok and any("irregular" in i for i in r.issues)


def test_gap_of_lost_beats_fails():
    raw = _good(hr_bpm=70)
    raw[1000:1000 + int(3.5 * FS)] = 50.0      # ~3.5 s sin señal (banda despegada)
    r = check_quality(raw)
    assert not r.ok and any("hueco" in i.lower() or "perdid" in i.lower() for i in r.issues)


def test_impossible_heart_rate_fails():
    r = check_quality(_good(hr_bpm=190))             # detectable (RR > 0.3 s) pero fuera de lo humano en reposo
    assert not r.ok and any("ritmo cardiaco" in i.lower() for i in r.issues)


def test_nan_fails():
    raw = _good()
    raw[10] = np.nan
    r = check_quality(raw)
    assert not r.ok and any("NaN" in i for i in r.issues)


def test_inverted_polarity_still_passes():
    assert check_quality(-_good(hr_bpm=72)).ok            # banda puesta al revés: la forma sigue siendo válida
