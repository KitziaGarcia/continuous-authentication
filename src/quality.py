"""Revisión automática de calidad de una grabación de ECG (para repetirla en el momento si falla).

Heurísticas sencillas calibradas con grabaciones reales del Polar H10: las buenas tuvieron variación
del ritmo <= 0.07 y razón de picos <= 1.3; la mala (contacto/movimiento) tuvo 0.22 y picos de 10x.
El módulo completo de preprocesamiento (NeuroKit2) podrá reemplazar esto más adelante.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from src.windowing import bandpass

HR_RANGE = (40.0, 180.0)      # latidos por minuto plausibles en una persona sentada
MAX_RR_VARIATION = 0.18       # desviación/media de los intervalos entre latidos
MAX_SPIKE_RATIO = 3.0         # amplitud máxima / percentil 99 de la amplitud
MAX_RR_GAP_S = 2.0            # más que esto sin latido = señal perdida
MIN_PEAKS = 8


@dataclass
class QualityReport:
    ok: bool
    issues: list = field(default_factory=list)
    n_peaks: int = 0
    hr_bpm: float = float("nan")
    rr_variation: float = float("nan")
    spike_ratio: float = float("nan")


def check_quality(raw, fs: int = 130) -> QualityReport:
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw).all():
        return QualityReport(False, ["La señal contiene NaN o valores no válidos."])
    if raw.std() < 1e-6:
        return QualityReport(False, ["La señal está plana: ¿banda mal puesta o electrodos secos?"])
    x = bandpass(raw, fs, 0.5, 40.0)
    edge = fs                                          # 1 s por lado: el filtro de 0.5 Hz deja un transitorio al empezar
    x = x[edge:-edge]
    if abs(x.min()) > abs(x.max()):                    # banda al revés: el latido sale invertido, sigue siendo válido
        x = -x
    # Amplitud de referencia ROBUSTA (percentil 99): un pico enorme y aislado no la infla, y así el
    # artefacto se detecta aunque sea tan grande que arruinaría la desviación estándar.
    ref = float(np.percentile(np.abs(x), 99))
    spike = float(np.max(np.abs(x)) / ref)
    issues = []
    if spike > MAX_SPIKE_RATIO:
        issues.append(f"Pico de amplitud anormal ({spike:.1f} veces lo normal): artefacto de contacto o movimiento.")
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=0.5 * ref)
    if len(peaks) < MIN_PEAKS:
        issues.append(f"Se detectan muy pocos latidos ({len(peaks)}): la señal parece ruido o no hay buen contacto.")
        return QualityReport(False, issues, n_peaks=len(peaks), spike_ratio=spike)
    rr = np.diff(peaks) / fs
    hr = 60.0 / float(np.median(rr))
    rr_var = float(rr.std() / rr.mean())
    if not HR_RANGE[0] <= hr <= HR_RANGE[1]:
        issues.append(f"Ritmo cardiaco imposible ({hr:.0f} lpm): probablemente se detectan mal los latidos.")
    if rr_var > MAX_RR_VARIATION:
        issues.append(f"Ritmo muy irregular (variación {rr_var:.2f}): movimiento o mal contacto.")
    if float(rr.max()) > MAX_RR_GAP_S:
        issues.append(f"Hueco de {rr.max():.1f} s sin latidos: se perdió la señal (banda despegada).")
    return QualityReport(not issues, issues, len(peaks), hr, rr_var, spike)
