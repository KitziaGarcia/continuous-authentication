"""ECG sintético: personas con forma de latido distinta y variación entre sesiones.

No es ECG real; sirve para probar el código y para que los notebooks corran sin datos.
Cada persona tiene su propia frecuencia cardiaca y amplitudes de las ondas P, QRS y T.
"""
from __future__ import annotations

import numpy as np

from src.data import WindowTable


def _gauss(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _one_window(rng, params: dict, window_samples: int, fs: int) -> np.ndarray:
    t = np.arange(window_samples) / fs
    period = 60.0 / (params["hr"] * (1 + rng.normal(0, 0.02)))
    first = -rng.uniform(0, period)             # fase aleatoria: la ventana no empieza en el pico R
    x = np.zeros(window_samples)
    for beat in np.arange(first, t[-1] + period, period):
        x += _gauss(t, beat - 0.16, 0.025, params["p_amp"])               # onda P
        x += _gauss(t, beat, params["qrs_w"], 1.0)                        # pico R
        x += _gauss(t, beat + 0.03, params["qrs_w"], -0.15)               # onda S
        x += _gauss(t, beat + params["t_off"], 0.05, params["t_amp"])     # onda T
    x += rng.normal(0, 0.02, window_samples)
    return (x - x.mean()) / (x.std() + 1e-8)


def make_synthetic_signal(seconds: float, fs: int = 130, seed: int = 0, hr_bpm: float | None = None,
                          rr_jitter: float = 0.0) -> np.ndarray:
    """Señal continua sintética (una sola persona): sirve para probar la grabación sin sensor.

    hr_bpm fija la frecuencia cardiaca; rr_jitter (0 a 1) hace irregular cada intervalo entre latidos.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    default_hr = rng.uniform(60, 85)           # se sortea siempre: así la semilla da la misma señal
    period = 60.0 / (hr_bpm if hr_bpm is not None else default_hr)
    if rr_jitter > 0:
        beats, b = [], 0.0
        while b < t[-1] + period:
            beats.append(b)
            b += period * (1 + rng.uniform(-rr_jitter, rr_jitter))
    else:
        beats = np.arange(0.0, t[-1] + period, period)
    x = np.zeros(n)
    for beat in beats:
        # Cada latido solo se calcula en su ventana local (de -0.5 s a +0.9 s): fuera de ahí las ondas
        # valen prácticamente 0. Recorrer toda la señal por cada latido era cuadrático (10 s para 15 min).
        i0, i1 = max(0, int((beat - 0.5) * fs)), min(n, int((beat + 0.9) * fs) + 1)
        if i0 >= i1:
            continue
        tl = t[i0:i1]
        x[i0:i1] += (_gauss(tl, beat - 0.16, 0.025, 0.15)      # onda P
                     + _gauss(tl, beat, 0.012, 1.0)            # pico R
                     + _gauss(tl, beat + 0.03, 0.012, -0.15)   # onda S
                     + _gauss(tl, beat + 0.30, 0.05, 0.30))    # onda T
    return x + rng.normal(0, 0.02, n)


def make_synthetic_table(
    n_people: int = 6,
    n_sessions: int = 3,
    windows_per_session: int = 40,
    seed: int = 0,
    prefix: str = "p",
    window_samples: int = 650,
    fs: int = 130,
) -> WindowTable:
    rng = np.random.default_rng(seed)
    wins, pids, sids = [], [], []
    for i in range(n_people):
        base = {
            "hr": rng.uniform(55, 95),
            "p_amp": rng.uniform(0.10, 0.30),
            "qrs_w": rng.uniform(0.008, 0.020),
            "t_amp": rng.uniform(0.20, 0.50),
            "t_off": rng.uniform(0.22, 0.38),
        }
        for s in range(n_sessions):
            # Variación entre sesiones: colocación de la banda y estado fisiológico distintos.
            sess = {k: v * (1 + rng.normal(0, 0.05)) for k, v in base.items()}
            sess["hr"] = base["hr"] + rng.normal(0, 3)
            for _ in range(windows_per_session):
                wins.append(_one_window(rng, sess, window_samples, fs))
                pids.append(f"{prefix}{i}")
                sids.append(f"s{s + 1}")
    n = len(wins)
    table = WindowTable(
        np.asarray(wins, dtype=np.float32)[:, None, :],
        np.array(pids),
        np.array(sids),
        np.full(n, "rest"),
        "synthetic",
    )
    table.validate(window_samples)
    return table
