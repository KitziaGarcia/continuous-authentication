"""Sentada de grabación guiada: 5 minutos, 4 actividades, revisión de calidad y archivos con nombre automático.

Uso (con la banda puesta y los electrodos húmedos):
    python -m src.session --id P03
Si una actividad sale mal:      python -m src.session --id P03 --redo reposo
Si te volviste a poner la banda: python -m src.session --id P03 --another-sitting
Sin sensor y sin esperar:        añade --simulate
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import time
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from src.config import Config
from src.errors import DataError
from src.plotting import save_figure
from src.protocol import (
    MIN_GAP_DAYS, SCHEDULE, TARGET_DAYS, existing_sessions, participant_days, plan_session, validate_id,
)
from src.quality import QualityReport, check_quality
from src.record import POLAR_FS, POLAR_RESOLUTION, save_recording
from src.synthetic import make_synthetic_signal


class ConnectionLost(RuntimeError):
    """Dejó de llegar señal de la banda a media sesión."""


@dataclass
class Dirs:
    raw: Path
    processed: Path
    discard: Path
    plots: Path


@dataclass
class SegmentResult:
    segment: object
    raw: np.ndarray
    quality: QualityReport
    saved: bool


# ---------------------------------------------------------------- fuentes de señal
class SimClock:
    """Reloj falso: `sleep` solo avanza el tiempo. Permite simular 5 minutos al instante."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, dt: float) -> None:
        self.t += dt


class SimulatedSource:
    def __init__(self, clock, fs: int = POLAR_FS, seed: int = 0):
        self.clock, self.fs, self._t0 = clock, fs, None
        self._sig = make_synthetic_signal(900, fs, seed=seed) * 400 + 20

    async def start(self) -> None:
        self._t0 = self.clock()

    async def stop(self) -> None:
        pass

    def n_samples(self) -> int:
        if self._t0 is None:
            return 0
        return min(int((self.clock() - self._t0) * self.fs), len(self._sig))

    def take(self, i0: int, i1: int) -> np.ndarray:
        return self._sig[i0:i1].copy()


class PolarSource:
    """Una sola conexión BLE para toda la sentada (reconectar entre actividades sería más frágil)."""

    def __init__(self, scan_timeout: float = 10.0):
        self.scan_timeout = scan_timeout
        self._samples: list = []
        self._cm = None
        self._dev = None

    async def start(self) -> None:
        from bleak import BleakScanner          # import perezoso: solo hace falta con el sensor
        from polar_python import PolarDevice

        print("Buscando Polar H10...")
        device = await BleakScanner.find_device_by_filter(
            lambda bd, ad: bd.name and "Polar H10" in bd.name, timeout=self.scan_timeout)
        if device is None:
            raise RuntimeError("No se encontró el sensor. Ponte la banda bien humedecida y vuelve a intentar.")
        self._cm = PolarDevice(device)
        self._dev = await self._cm.__aenter__()
        await self._dev.start_ecg_stream(ecg_callback=lambda d: self._samples.extend(d.data),
                                         sample_rate=POLAR_FS, resolution=POLAR_RESOLUTION)

    async def stop(self) -> None:
        if self._cm is None:
            return
        try:
            await self._dev.stop_ecg_stream()
        except Exception:                         # si ya se desconectó no importa: los datos ya están
            pass
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception:
            pass

    def n_samples(self) -> int:
        return len(self._samples)

    def take(self, i0: int, i1: int) -> np.ndarray:
        return np.array(self._samples[i0:i1], dtype=np.float64)


# ---------------------------------------------------------------- la sentada
async def collect(source, segments, sleep, clock, say, on_segment, fs: int = POLAR_FS,
                  settle_s: float = 10.0, ready_s: float = 5.0, edge_s: float = 1.0, stall_s: float = 3.0) -> None:
    """Corre la sentada completa con una sola conexión y llama a `on_segment` al terminar CADA actividad.

    Así, si algo se interrumpe a la mitad, lo ya grabado queda guardado y revisado.
    """

    async def wait(seconds: float, label: str) -> None:
        remaining, last_n, stalled = seconds, source.n_samples(), 0.0
        while remaining > 1e-9:
            step = min(1.0, remaining)
            await sleep(step)
            remaining -= step
            n = source.n_samples()
            stalled = stalled + step if n == last_n else 0.0
            last_n = n
            if stalled >= stall_s:
                raise ConnectionLost("Se perdió la conexión con la banda (dejó de llegar señal). "
                                     "Revisa que esté puesta y húmeda, y repite.")
            say(f"\r  {label}: {remaining:3.0f} s ", end="")
        say("\r" + " " * 40 + "\r", end="")

    await source.start()
    try:
        say("Conectado. Quédate quieto unos segundos mientras la señal se estabiliza...")
        await wait(settle_s, "estabilizando")
        for seg in segments:
            say(f"\n>> {seg.activity.upper()} ({seg.seconds:.0f} s): {seg.instruction}")
            await wait(ready_s, "prepárate")
            i0 = source.n_samples()
            await wait(seg.seconds, seg.activity)
            i1 = source.n_samples()
            trim = int(edge_s * fs)               # los bordes se descartan: llegan por lotes y hay transitorios
            on_segment(seg, source.take(i0 + trim, i1 - trim))
    finally:
        await source.stop()


def _discard_existing(pid: str, session_id: str, tag: str, dirs: Dirs, stamp: str) -> None:
    """Mueve (nunca borra) la grabación anterior de esa actividad antes de guardar la nueva."""
    base = f"{pid}__{session_id}__{tag}"
    for folder, suffix in ((dirs.processed, ".npz"), (dirs.raw, ".npy")):
        old = Path(folder) / f"{base}{suffix}"
        if old.exists():
            dirs.discard.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(dirs.discard / f"{base}__reemplazada_{stamp}{suffix}"))


def store_segment(raw, seg, pid: str, session_id: str, dirs: Dirs, cfg: Config, now: datetime | None = None
                  ) -> SegmentResult:
    """Revisa la calidad: si es buena la guarda en el conjunto de datos; si no, la aparta sin perderla."""
    quality = check_quality(raw, POLAR_FS)
    stamp = (now or datetime.now()).strftime("%H%M%S")
    if not quality.ok:
        dirs.discard.mkdir(parents=True, exist_ok=True)
        np.save(dirs.discard / f"{pid}__{session_id}__{seg.activity}__{stamp}.npy", raw)
        return SegmentResult(seg, raw, quality, False)
    _discard_existing(pid, session_id, seg.activity, dirs, stamp)
    save_recording(raw, POLAR_FS, pid, session_id, seg.activity, dirs.raw, dirs.processed, cfg, tag=seg.activity)
    return SegmentResult(seg, raw, quality, True)


def plot_session(results: list, pid: str, session_id: str, path) -> Path:
    """Una imagen con todas las actividades de esta sentada y el veredicto de cada una."""
    fig = Figure(figsize=(12, 2.6 * max(1, len(results))))
    axes = fig.subplots(max(1, len(results)), 1, squeeze=False)[:, 0]
    for ax, r in zip(axes, results):
        t = np.arange(len(r.raw)) / POLAR_FS
        ax.plot(t, r.raw, lw=0.6, color="tab:green" if r.saved else "tab:red")
        verdict = f"OK ({r.quality.hr_bpm:.0f} lpm)" if r.saved else "REPETIR: " + "; ".join(r.quality.issues)[:90]
        ax.set_title(f"{pid} · {session_id} · {r.segment.activity} — {verdict}", fontsize=10,
                     color="black" if r.saved else "tab:red")
        ax.set_ylabel("µV")
    axes[-1].set_xlabel("segundos")
    fig.tight_layout()
    return save_figure(fig, path)


# ---------------------------------------------------------------- línea de comandos
def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.session", description="Sentada de grabación guiada (5 min)")
    p.add_argument("--id", required=True, help="ID de la persona: P01, P02, ... (siempre el mismo)")
    p.add_argument("--redo", nargs="+", metavar="ACTIVIDAD", help="repetir solo esas actividades de hoy")
    p.add_argument("--another-sitting", action="store_true", help="otra sentada hoy (te quitaste y pusiste la banda)")
    p.add_argument("--simulate", action="store_true", help="sin sensor y sin esperar (para probar)")
    p.add_argument("--no-open", action="store_true", help="no abrir la imagen al terminar")
    p.add_argument("--today", default=None, help="fecha AAAA-MM-DD (solo para pruebas)")
    p.add_argument("--raw-dir", default="data/raw/polar")
    p.add_argument("--processed-dir", default=f"{cfg.data_dir}/polar")
    p.add_argument("--discard-dir", default="data/descartadas")
    p.add_argument("--plots-dir", default="reports/recordings")
    a = p.parse_args(argv)

    dirs = Dirs(Path(a.raw_dir), Path(a.processed_dir), Path(a.discard_dir), Path(a.plots_dir))
    today = date.fromisoformat(a.today) if a.today else date.today()
    results: list = []
    try:
        pid = validate_id(a.id)
        plan = plan_session(pid, today, existing_sessions(dirs.processed, pid), a.redo, a.another_sitting)
        print(f"\n{pid} · sesión {plan.session_id} · se grabará: " + ", ".join(s.activity for s in plan.segments))
        for note in plan.notes:
            print("NOTA:", note)

        def on_segment(seg, raw):
            r = store_segment(raw, seg, pid, plan.session_id, dirs, cfg)
            results.append(r)
            if r.saved:
                print(f"  {seg.activity}: OK ({r.quality.hr_bpm:.0f} lpm, guardada)")
            else:
                print(f"  {seg.activity}: REPETIR -> " + " | ".join(r.quality.issues))

        if a.simulate:
            clock = SimClock()
            source = SimulatedSource(clock, seed=zlib.crc32(pid.encode()) % 1000)
            sleep, clock_fn = clock.sleep, clock
        else:
            source, sleep, clock_fn = PolarSource(), asyncio.sleep, time.monotonic
        asyncio.run(collect(source, plan.segments, sleep, clock_fn,
                            lambda text, end="\n": print(text, end=end, flush=True), on_segment))
    except (ValueError, DataError, RuntimeError, FileExistsError) as e:
        print("ERROR:", e)
        if results:
            print("Lo grabado antes del error ya quedó guardado.")
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido. Lo grabado antes quedó guardado; para completar, corre el mismo comando otra vez.")
        return 1

    png = plot_session(results, pid, plan.session_id, dirs.plots / f"{pid}__{plan.session_id}.png")
    bad = [r.segment.activity for r in results if not r.saved]
    if bad:
        print(f"\nHay que repetir: {', '.join(bad)}. Corre:  python -m src.session --id {pid} --redo {' '.join(bad)}")
    else:
        print("\nSesión completa y con buena calidad.")
    days = participant_days(dirs.processed, pid)
    print(f"{pid} lleva {len(days)} de {TARGET_DAYS} días grabados ({', '.join(days)}).")
    if len(days) < TARGET_DAYS and not bad:
        print(f"Próxima sesión sugerida: a partir del {(date.fromisoformat(days[-1]) + timedelta(days=MIN_GAP_DAYS)).isoformat()}.")
    print(f"Imagen: {png}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
