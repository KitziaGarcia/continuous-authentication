import asyncio

import numpy as np

from src.config import Config
from src.live import AUTH, BAD_SIGNAL, REJECT, Decider, Step, format_step, make_template, monitor, summary
from src.session import SimClock, SimulatedSource
from src.synthetic import make_synthetic_signal


def _embed(windows):
    # "Red" falsa: el embedding es la propia ventana recortada; basta para probar la lógica de decisión.
    return windows[:, 0, :16].astype(np.float64)


def test_decider_smooths_and_applies_threshold():
    d = Decider(threshold=0.5, smooth=3)
    assert d.update(0.9)[1] == AUTH
    assert d.update(0.1)[1] == AUTH            # promedio 0.5: una ventana mala no tumba la sesión
    assert d.update(0.1)[1] == REJECT          # promedio 0.37


def test_make_template_is_unit_norm():
    t = make_template(_embed, np.random.default_rng(0).normal(size=(5, 1, 650)))
    assert abs(np.linalg.norm(t) - 1) < 1e-9


def test_monitor_flags_flat_signal_as_bad_signal_not_reject():
    cfg, clock = Config(), SimClock()
    src = SimulatedSource(clock, signal=np.zeros(130 * 60))
    asyncio.run(src.start())
    steps = []
    asyncio.run(monitor(src, np.ones(16) / 4, _embed, cfg, 0.5, clock.sleep, clock, lambda *a, **k: None,
                        steps.append, duration=30))
    assert steps and all(s.status == BAD_SIGNAL and s.score is None for s in steps)


def test_monitor_decides_on_real_looking_signal():
    cfg, clock = Config(), SimClock()
    sig = make_synthetic_signal(60, 130, seed=3) * 400
    src = SimulatedSource(clock, signal=sig)
    asyncio.run(src.start())
    steps = []
    asyncio.run(monitor(src, np.ones(16) / 4, _embed, cfg, -1.0, clock.sleep, clock, lambda *a, **k: None,
                        steps.append, duration=40))
    decided = [s for s in steps if s.score is not None]
    assert decided and all(s.status == AUTH for s in decided)


def test_format_and_summary():
    ok, bad = Step(5, AUTH, 0.9, 0.9), Step(7, BAD_SIGNAL, None, None)
    assert "AUTENTICADO" in format_step(ok, 0.3) and "sin decisión" in format_step(bad, 0.3)
    assert "1 AUTENTICADO" in summary([ok, bad]) and "1 con señal mala" in summary([ok, bad])


def test_scores_before_a_signal_gap_do_not_carry_over():
    cfg, clock = Config(), SimClock()
    good = make_synthetic_signal(45, 130, seed=3) * 400
    sig = np.concatenate([good, np.zeros(130 * 20), good])
    src = SimulatedSource(clock, signal=sig)
    asyncio.run(src.start())
    tpl = np.eye(16)[0]

    def embed(w):                      # parecido a la huella hasta que acaba la pausa (t=65 s); después, ortogonal
        return np.eye(16)[[0 if clock() < 65 else 1]]

    steps = []
    asyncio.run(monitor(src, tpl, embed, cfg, 0.5, clock.sleep, clock, lambda *a, **k: None,
                        steps.append, duration=100))
    first_after_gap = next(s for i, s in enumerate(steps) if s.score is not None and steps[i - 1].score is None)
    assert first_after_gap.status == REJECT           # sin reinicio heredaría el promedio alto de antes
