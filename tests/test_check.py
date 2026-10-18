import asyncio

import numpy as np
import pytest

from src.check import main, run_check
from src.session import ConnectionLost, PolarSource, SimClock, SimulatedSource, wait_for_ecg
from src.synthetic import make_synthetic_signal

FS = 130


def _run(coro):
    return asyncio.run(coro)


def _source(signal, clock):
    return SimulatedSource(clock, seed=0, signal=signal)


def _check(signal, seconds=12):
    clock = SimClock()
    return _run(run_check(_source(signal, clock), clock.sleep, clock, lambda *a, **k: None, seconds=seconds))


def test_good_signal_is_ready():
    raw, report = _check(make_synthetic_signal(60, FS, seed=1, hr_bpm=72) * 400 + 20)
    assert report.ok and 60 < report.hr_bpm < 85
    assert len(raw) == pytest.approx(12 * FS, abs=FS * 0.2)


def test_check_allows_ecg_stream_startup_delay():
    class DelayedSource(PolarSource):
        def __init__(self, clock):
            self.clock = clock
            self.source = SimulatedSource(clock, seed=1)

        async def start(self):
            await self.source.start()

        async def stop(self):
            await self.source.stop()

        def n_samples(self):
            if self.clock() < 7:
                return 0
            return self.source.n_samples()

        def take(self, i0, i1):
            return self.source.take(i0, i1)

    clock = SimClock()
    source = DelayedSource(clock)
    raw, report = _run(run_check(source, clock.sleep, clock, lambda *a, **k: None))
    assert report.ok
    assert len(raw) == pytest.approx(12 * FS, abs=FS * 0.2)


def test_wait_for_ecg_reports_when_stream_never_starts():
    class SilentSource:
        def n_samples(self):
            return 0

    clock = SimClock()
    with pytest.raises(ConnectionLost, match="no empezó a enviar señal ECG en 15 s"):
        _run(wait_for_ecg(SilentSource(), clock.sleep))
    assert clock() == pytest.approx(15)


def test_flat_signal_is_not_ready_and_says_why():
    _, report = _check(np.full(60 * FS, 20.0) + np.random.default_rng(0).normal(0, 1e-9, 60 * FS))
    assert not report.ok and any("plana" in i for i in report.issues)


def test_noise_signal_is_not_ready():
    _, report = _check(np.random.default_rng(0).normal(0, 300, 60 * FS))
    assert not report.ok


def test_dead_stream_raises_connection_lost():
    class Dead:
        async def start(self): pass
        async def stop(self): pass
        def n_samples(self): return 5
        def take(self, a, b): return np.zeros(0)

    clock = SimClock()
    with pytest.raises(ConnectionLost):
        _run(run_check(Dead(), clock.sleep, clock, lambda *a, **k: None, seconds=12))


def test_too_short_check_is_rejected():
    clock = SimClock()
    with pytest.raises(ValueError, match="8 s"):
        _run(run_check(_source(np.zeros(100), clock), clock.sleep, clock, lambda *a, **k: None, seconds=5))


def test_cli_good_signal_prints_ready_and_writes_picture(tmp_path, capsys):
    out = tmp_path / "check.png"
    assert main(["--simulate", "--no-open", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "LISTO" in text and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("kind", ["flat", "noise", "spike"])
def test_cli_bad_signal_says_to_check_and_returns_code_2(tmp_path, capsys, kind):
    out = tmp_path / "check.png"
    assert main(["--simulate", "--simulate-bad", kind, "--no-open", "--out", str(out)]) == 2
    text = capsys.readouterr().out
    assert "REVISA" in text and "NO graben todavía" in text and out.exists()


def test_cli_rejects_too_short_duration(tmp_path, capsys):
    assert main(["--simulate", "--seconds", "5", "--no-open", "--out", str(tmp_path / "c.png")]) == 1
    assert "8 s" in capsys.readouterr().out
