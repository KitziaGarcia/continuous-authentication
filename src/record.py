"""Grabación de ECG con la banda Polar H10: guarda la señal y dibuja una gráfica de verificación.

Uso (con la banda puesta y los electrodos húmedos):
    python -m src.record --id kitzia --seconds 60
Sin sensor (para probar el flujo):  añade --simulate
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from src.config import Config
from src.data import save_window_file
from src.errors import DataError
from src.plotting import save_figure
from src.synthetic import make_synthetic_signal
from src.windowing import signal_to_windows

POLAR_FS = 130          # Hz que entrega el H10 en modo ECG
POLAR_RESOLUTION = 14   # bits


def default_session_id(now: datetime | None = None) -> str:
    """Fecha y hora: ordena cronológicamente como texto (la sesión 1 siempre es la más antigua)."""
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M")


def rate_warning(measured_hz: float, nominal: float = POLAR_FS) -> str | None:
    """Avisa si la frecuencia real de muestreo se aleja de la nominal (pérdida de paquetes BLE)."""
    if abs(measured_hz - nominal) / nominal <= 0.10:
        return None
    return (f"Frecuencia medida {measured_hz:.0f} Hz, muy lejos de {nominal} Hz: "
            "probablemente se perdieron paquetes Bluetooth (acércate al equipo y repite).")


def save_recording(raw, fs, participant_id, session_id, activity, raw_dir, processed_dir, cfg: Config,
                   tag: str | None = None) -> dict:
    """Guarda la señal cruda (.npy) y sus ventanas listas para el modelo (.npz).

    `tag` (p. ej. la actividad) va al final del nombre para que varias grabaciones convivan en una sesión.
    """
    raw = np.asarray(raw, dtype=np.float64)
    duration = len(raw) / fs
    if duration < cfg.window_samples / cfg.fs:
        raise DataError(f"Grabación muy corta ({duration:.1f} s): se necesitan al menos "
                        f"{cfg.window_samples / cfg.fs:.0f} s, idealmente 60 s o más.")
    stem = f"{participant_id}__{session_id}" + (f"__{tag}" if tag else "")
    raw_path = Path(raw_dir) / f"{stem}.npy"
    if raw_path.exists():
        raise FileExistsError(f"Ya existe {raw_path}; usa otro --session para no sobrescribir datos.")
    windows = signal_to_windows(raw, fs, cfg)
    if len(windows) == 0:
        raise DataError("No se obtuvieron ventanas válidas: la señal está plana "
                        "(¿banda mal puesta o electrodos secos?).")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    np.save(raw_path, raw)
    win_path = Path(processed_dir) / f"{stem}.npz"
    save_window_file(win_path, windows, participant_id, session_id, activity, "polar")
    return {"n_samples": len(raw), "duration_s": duration, "n_windows": len(windows),
            "raw_path": raw_path, "windows_path": win_path}


def plot_recording(raw, fs, out_path, participant_id, session_id, n_windows) -> Path:
    """Imagen de comprobación: toda la señal y un zoom de 10 s para ver los latidos."""
    raw = np.asarray(raw, dtype=np.float64)
    t = np.arange(len(raw)) / fs
    fig = Figure(figsize=(12, 6))
    top, bottom = fig.subplots(2, 1)
    top.plot(t, raw, lw=0.6)
    top.set_title(f"{participant_id} · {session_id} — {t[-1]:.0f} s, {len(raw)} muestras, "
                  f"{n_windows} ventanas utilizables")
    top.set_xlabel("segundos")
    top.set_ylabel("µV")
    start = max(0.0, t[-1] / 2 - 5)
    m = (t >= start) & (t <= start + 10)
    bottom.plot(t[m], raw[m], lw=0.9, color="tab:green")
    bottom.set_title("Zoom de 10 s (deberías ver latidos regulares con picos R)")
    bottom.set_xlabel("segundos")
    bottom.set_ylabel("µV")
    fig.tight_layout()
    return save_figure(fig, out_path)


async def record_polar(seconds: float, scan_timeout: float = 10.0):
    """Conecta al H10 por BLE y junta muestras de ECG durante `seconds`. Devuelve (señal, segundos reales)."""
    from bleak import BleakScanner          # import perezoso: solo hace falta con el sensor
    from polar_python import PolarDevice

    samples: list = []
    print("Buscando Polar H10...")
    device = await BleakScanner.find_device_by_filter(
        lambda bd, ad: bd.name and "Polar H10" in bd.name, timeout=scan_timeout)
    if device is None:
        raise RuntimeError("No se encontró el sensor. Ponte la banda bien humedecida y vuelve a intentar.")
    async with PolarDevice(device) as polar:
        await polar.start_ecg_stream(ecg_callback=lambda d: samples.extend(d.data),
                                     sample_rate=POLAR_FS, resolution=POLAR_RESOLUTION)
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < seconds:
            await asyncio.sleep(1)
            print(f"\rGrabando... {int(elapsed)}/{int(seconds)} s ({len(samples)} muestras)", end="", flush=True)
        try:
            await polar.stop_ecg_stream()
        except Exception:                    # si ya se desconectó, no importa: ya tenemos los datos
            pass
    print()
    return np.array(samples, dtype=np.float64), time.monotonic() - t0


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.record", description="Graba ECG con el Polar H10")
    p.add_argument("--id", required=True, help="identificador de la persona, p. ej. kitzia")
    p.add_argument("--session", default=None, help="por defecto fecha_hora actual")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--activity", default="rest")
    p.add_argument("--skip-seconds", type=float, default=3.0, help="se descartan al inicio (la banda se asienta)")
    p.add_argument("--simulate", action="store_true", help="señal sintética, sin sensor")
    p.add_argument("--no-open", action="store_true", help="no abrir la imagen al terminar")
    p.add_argument("--raw-dir", default="data/raw/polar")
    p.add_argument("--processed-dir", default=f"{cfg.data_dir}/polar")
    p.add_argument("--plots-dir", default="reports/recordings")
    a = p.parse_args(argv)

    session = a.session or default_session_id()
    try:
        if not re.fullmatch(r"[A-Za-z0-9-]+", a.id):
            raise ValueError("--id solo puede llevar letras, números y guiones (sin espacios ni guion bajo)")
        if a.simulate:
            seed = zlib.crc32(a.id.encode()) % 1000      # cada id simulado tiene su propio "corazón"
            raw, elapsed = make_synthetic_signal(a.seconds, POLAR_FS, seed=seed) * 400, a.seconds
        else:
            raw, elapsed = asyncio.run(record_polar(a.seconds))
        measured = len(raw) / elapsed if elapsed > 0 else 0.0
        print(f"Se recibieron {len(raw)} muestras en {elapsed:.1f} s ({measured:.0f} Hz).")
        if not a.simulate and (warning := rate_warning(measured)):
            print("AVISO:", warning)
        raw = raw[int(a.skip_seconds * POLAR_FS):]
        info = save_recording(raw, POLAR_FS, a.id, session, a.activity, a.raw_dir, a.processed_dir, cfg)
        png = plot_recording(raw, POLAR_FS, Path(a.plots_dir) / f"{a.id}__{session}.png", a.id, session,
                             info["n_windows"])
    except (DataError, RuntimeError, FileExistsError, ValueError) as e:
        print("ERROR:", e)
        return 1
    print(f"Guardado: {info['raw_path']}\n          {info['windows_path']}  ({info['n_windows']} ventanas)")
    print(f"Imagen:   {png}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(png)                    # abre la imagen con el visor de Windows
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
