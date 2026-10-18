"""Chequeo rápido de que el Polar H10 está leyendo bien, ANTES de grabar una sesión.

Uso (con la banda puesta y los electrodos húmedos):
    python -m src.check
Si dice REVISA, ajusten la banda y repitan el chequeo antes de correr  python -m src.session --id Pxx
Para ver cómo se ve un chequeo malo sin sensor:  python -m src.check --simulate-bad flat   (o noise / spike)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np
from matplotlib.figure import Figure

from src.plotting import save_figure
from src.protocol import Segment
from src.quality import QualityReport, check_quality
from src.record import POLAR_FS
from src.session import PolarSource, SimClock, SimulatedSource, collect
from src.synthetic import make_synthetic_signal

MIN_SECONDS = 8.0       # con menos tiempo hay muy pocos latidos para opinar con confianza
QUICK_MIN_PEAKS = 5     # en ~10 s útiles hay unos 10-12 latidos; se exigen al menos 5


async def run_check(source, sleep, clock, say, seconds: float = 12.0, settle_s: float = 2.0, fs: int = POLAR_FS):
    """Conecta, deja asentar la señal unos segundos y analiza `seconds`. Devuelve (señal, reporte)."""
    if seconds < MIN_SECONDS:
        raise ValueError(f"El chequeo necesita al menos {MIN_SECONDS:.0f} s para ver suficientes latidos "
                         "(con menos tiempo el veredicto no sería confiable).")
    got: list = []
    seg = Segment("chequeo", seconds, "Quédate quieto, respira normal y no hables.")
    await collect(source, [seg], sleep, clock, say, lambda s, raw: got.append(raw), fs=fs,
                  settle_s=settle_s, ready_s=0.0, edge_s=0.0)
    raw = got[0]
    return raw, check_quality(raw, fs, min_peaks=QUICK_MIN_PEAKS)


def plot_check(raw, report: QualityReport, path, fs: int = POLAR_FS):
    """La señal con los latidos detectados marcados: para ver con los ojos que sí está leyendo ECG."""
    raw = np.asarray(raw, dtype=np.float64)
    t = np.arange(len(raw)) / fs
    fig = Figure(figsize=(11, 4))
    ax = fig.subplots()
    ax.plot(t, raw, lw=0.9, color="tab:green" if report.ok else "tab:red")
    if len(report.peak_indices):
        pk = report.peak_indices
        ax.plot(t[pk], raw[pk], "v", color="black", ms=7, label="latido detectado")
        ax.legend(loc="upper right")
    if report.ok:
        ax.set_title(f"LISTO - {report.hr_bpm:.0f} lpm, {report.n_peaks} latidos detectados", color="tab:green")
    else:
        ax.set_title("REVISA - " + report.issues[0], color="tab:red", fontsize=10)
    ax.set_xlabel("segundos")
    ax.set_ylabel("µV")
    fig.tight_layout()
    return save_figure(fig, path)


def _bad_signal(kind: str, seconds: float, fs: int = POLAR_FS) -> np.ndarray:
    n = int((seconds + 60) * fs)
    if kind == "flat":
        return np.full(n, 20.0)
    if kind == "noise":
        return np.random.default_rng(0).normal(0, 300, n)
    sig = make_synthetic_signal(seconds + 60, fs, seed=7, hr_bpm=72) * 400 + 20       # spike: ECG bueno + artefacto
    sig[int(8 * fs):int(8 * fs) + 6] += 15000
    return sig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.check", description="Chequeo rápido del Polar H10")
    p.add_argument("--seconds", type=float, default=12.0, help=f"duración del análisis (mínimo {MIN_SECONDS:.0f})")
    p.add_argument("--simulate", action="store_true", help="sin sensor (señal buena simulada)")
    p.add_argument("--simulate-bad", choices=["flat", "noise", "spike"], help="sin sensor, con una señal mala simulada")
    p.add_argument("--no-open", action="store_true", help="no abrir la imagen")
    p.add_argument("--out", default="reports/check.png")
    a = p.parse_args(argv)

    try:
        if a.simulate or a.simulate_bad:
            clock = SimClock()
            sig = _bad_signal(a.simulate_bad, a.seconds) if a.simulate_bad else None
            source, sleep, clock_fn = SimulatedSource(clock, seed=7, signal=sig), clock.sleep, clock
        else:
            source, sleep, clock_fn = PolarSource(), asyncio.sleep, time.monotonic
        raw, report = asyncio.run(run_check(source, sleep, clock_fn,
                                            lambda text, end="\n": print(text, end=end, flush=True), seconds=a.seconds))
    except (ValueError, RuntimeError) as e:          # ConnectionLost y "no se encontró el sensor" son RuntimeError
        print("ERROR:", e)
        return 1

    png = plot_check(raw, report, a.out)
    if report.ok:
        print(f"\nLISTO: la señal se ve bien ({report.hr_bpm:.0f} lpm, {report.n_peaks} latidos detectados). "
              "Ya pueden grabar:  python -m src.session --id Pxx")
    else:
        print("\nREVISA: la señal NO se ve bien:")
        for issue in report.issues:
            print("  -", issue)
        print("Qué hacer: humedezcan los electrodos, ajusten la banda, quédense quietos y repitan este chequeo. "
              "NO graben todavía.")
    print(f"Imagen: {png}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(png)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
