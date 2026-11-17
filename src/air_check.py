"""Diagnóstico: ¿qué ve el sistema cuando la banda está al aire (sin tocar piel)?

Uso:  python -m src.air_check            # 60 s con la banda puesta en la mesa, sin tocar piel
      python -m src.air_check --seconds 90
Cada 5 s imprime si la banda dice que hay contacto y si la señal pasaría el chequeo de calidad.
"""
from __future__ import annotations

import argparse
import asyncio

from src.quality import check_quality
from src.record import POLAR_FS
from src.session import PolarSource, wait_for_ecg


async def run(seconds: float) -> None:
    src = PolarSource()
    await src.start()
    try:
        await wait_for_ecg(src)
        print("t(s)  contacto(banda)  calidad  detalle")
        for t in range(5, int(seconds) + 1, 5):
            await asyncio.sleep(5)
            n = src.n_samples()
            if n < 10 * POLAR_FS:
                continue
            q = check_quality(src.take(n - 10 * POLAR_FS, n), POLAR_FS, min_peaks=5)
            print(f"{t:4d}  {src.contact!s:15}  {'PASA' if q.ok else 'falla':7}  "
                  f"picos={q.n_peaks} lpm={q.hr_bpm:.0f} var={q.rr_variation:.2f}")
    finally:
        await src.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0)
    asyncio.run(run(p.parse_args().seconds))
