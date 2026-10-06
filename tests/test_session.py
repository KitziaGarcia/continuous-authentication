import asyncio

import numpy as np
import pytest

from src.config import Config
from src.data import load_window_dir
from src.protocol import SCHEDULE
from src.session import (
    ConnectionLost, Dirs, SimClock, SimulatedSource, collect, main, store_segment,
)
from src.synthetic import make_synthetic_signal

FS = 130


def _run(coro):
    return asyncio.run(coro)


def _collect(source_seed=0, segments=SCHEDULE, **kw):
    clock = SimClock()
    source = SimulatedSource(clock, seed=source_seed)
    got, say = [], []
    _run(collect(source, segments, clock.sleep, clock, lambda t, end="\n": say.append(t),
                 lambda seg, raw: got.append((seg, raw)), **kw))
    return clock, got, say


def test_collect_returns_one_trimmed_segment_per_activity_in_order():
    clock, got, say = _collect()
    assert [s.activity for s, _ in got] == ["reposo", "lectura", "problemas", "hablando"]
    for seg, raw in got:
        assert len(raw) == pytest.approx(seg.seconds * FS - 2 * FS, abs=FS * 0.2)    # recorta 1 s por lado
    # tiempo total simulado = 10 s estabilizando + (5 s de preparación + actividad) por cada una
    assert clock() == pytest.approx(10 + sum(5 + s.seconds for s in SCHEDULE))
    assert any("REPOSO" in line for line in say) and any("Siéntate" in line for line in say)


def test_collect_segments_come_from_the_matching_moment_of_the_stream():
    clock, got, _ = _collect()
    sig = make_synthetic_signal(900, FS, seed=0) * 400 + 20
    start_reposo = int((10 + 5) * FS) + FS                                           # tras estabilizar, preparar y recortar
    assert np.allclose(got[0][1][:50], sig[start_reposo:start_reposo + 50])


def test_collect_raises_connection_lost_when_the_stream_stops():
    class DeadSource:
        stopped = False

        async def start(self):
            pass

        async def stop(self):
            self.stopped = True

        def n_samples(self):
            return 100            # nunca crece

        def take(self, a, b):
            return np.zeros(0)

    clock, src = SimClock(), DeadSource()
    with pytest.raises(ConnectionLost, match="conexión"):
        _run(collect(src, SCHEDULE, clock.sleep, clock, lambda *a, **k: None, lambda *a: None))
    assert src.stopped                                # siempre se cierra la conexión, aunque falle


def test_collect_calls_on_segment_right_after_each_activity_so_nothing_is_lost_if_it_breaks_later():
    clock = SimClock()
    source = SimulatedSource(clock, seed=0)
    saved = []

    def on_segment(seg, raw):
        saved.append(seg.activity)
        if seg.activity == "lectura":
            raise KeyboardInterrupt                    # el usuario interrumpe a media sesión

    with pytest.raises(KeyboardInterrupt):
        _run(collect(source, SCHEDULE, clock.sleep, clock, lambda *a, **k: None, on_segment))
    assert saved == ["reposo", "lectura"]


def _dirs(tmp_path):
    return Dirs(raw=tmp_path / "raw", processed=tmp_path / "proc", discard=tmp_path / "disc", plots=tmp_path / "plots")


def _good_raw(seconds=30, seed=0):
    return make_synthetic_signal(seconds, FS, seed=seed, hr_bpm=75) * 400 + 20


def test_store_segment_saves_good_recordings_with_activity_in_the_name(tmp_path):
    r = store_segment(_good_raw(), SCHEDULE[0], "P01", "2026-10-06", _dirs(tmp_path), Config())
    assert r.saved and r.quality.ok
    assert (tmp_path / "proc" / "P01__2026-10-06__reposo.npz").exists()
    assert (tmp_path / "raw" / "P01__2026-10-06__reposo.npy").exists()


def test_store_segment_sends_bad_recordings_to_discard_and_not_to_the_dataset(tmp_path):
    bad = _good_raw()
    bad[1500:1506] += 15000
    r = store_segment(bad, SCHEDULE[0], "P01", "2026-10-06", _dirs(tmp_path), Config())
    assert not r.saved and not r.quality.ok
    assert not (tmp_path / "proc").exists() or not list((tmp_path / "proc").glob("*.npz"))
    assert len(list((tmp_path / "disc").glob("P01__2026-10-06__reposo__*.npy"))) == 1       # no se pierde el dato


def test_store_segment_replaces_a_previous_one_moving_it_to_discard(tmp_path):
    d = _dirs(tmp_path)
    store_segment(_good_raw(seed=1), SCHEDULE[0], "P01", "2026-10-06", d, Config())
    r = store_segment(_good_raw(seed=2), SCHEDULE[0], "P01", "2026-10-06", d, Config())
    assert r.saved
    assert len(list((tmp_path / "proc").glob("*.npz"))) == 1                                # sigue habiendo una sola
    moved = sorted(p.name for p in (tmp_path / "disc").iterdir())
    assert len(moved) == 2 and all("reemplazada" in n for n in moved)                       # .npz y .npy viejos


def test_store_segment_keeps_the_old_recording_when_the_new_one_is_bad(tmp_path):
    d = _dirs(tmp_path)
    store_segment(_good_raw(seed=1), SCHEDULE[0], "P01", "2026-10-06", d, Config())
    r = store_segment(np.zeros(30 * FS), SCHEDULE[0], "P01", "2026-10-06", d, Config())
    assert not r.saved
    assert (tmp_path / "proc" / "P01__2026-10-06__reposo.npz").exists()


def _cli(tmp_path, *extra, pid="P01", day="2026-10-06"):
    return main(["--id", pid, "--simulate", "--no-open", "--today", day,
                 "--raw-dir", str(tmp_path / "raw"), "--processed-dir", str(tmp_path / "proc"),
                 "--discard-dir", str(tmp_path / "disc"), "--plots-dir", str(tmp_path / "plots"), *extra])


def test_cli_full_session_end_to_end(tmp_path, capsys):
    assert _cli(tmp_path) == 0
    t = load_window_dir(tmp_path / "proc")                 # pasa validate: z-score, sesiones ordenadas
    assert t.persons() == ["P01"] and t.sessions_of("P01") == ["2026-10-06"]
    assert set(t.activities.tolist()) == {"reposo", "lectura", "problemas", "hablando"}
    assert (tmp_path / "plots" / "P01__2026-10-06.png").exists()
    out = capsys.readouterr().out
    assert "OK" in out and "1 de 3" in out                  # avance del participante


def test_cli_second_run_the_same_day_refuses_and_explains(tmp_path, capsys):
    _cli(tmp_path)
    capsys.readouterr()
    assert _cli(tmp_path) == 1
    out = capsys.readouterr().out
    assert "--redo" in out and "--another-sitting" in out


def test_cli_redo_replaces_only_that_activity(tmp_path):
    _cli(tmp_path)
    before = {p.name for p in (tmp_path / "proc").glob("*.npz")}
    assert _cli(tmp_path, "--redo", "reposo") == 0
    assert {p.name for p in (tmp_path / "proc").glob("*.npz")} == before
    assert any("reemplazada" in p.name for p in (tmp_path / "disc").iterdir())


def test_cli_rejects_invalid_id_without_recording_anything(tmp_path, capsys):
    assert _cli(tmp_path, pid="juanC2") == 1
    assert "P01" in capsys.readouterr().out and not (tmp_path / "proc").exists()


def test_cli_second_day_counts_progress_and_suggests_next_date(tmp_path, capsys):
    _cli(tmp_path, day="2026-10-06")
    capsys.readouterr()
    assert _cli(tmp_path, day="2026-10-09") == 0
    out = capsys.readouterr().out
    assert "2 de 3" in out and "2026-10-11" in out          # próxima sesión: 2 días después
