"""Datos: tabla de ventanas con metadatos, lectura/escritura, particiones por persona y aumento."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.errors import DataError, LeakageError


def _natural_key(text: str) -> list:
    """Orden "natural": s2 < s10. Se compara contra el orden de texto para detectar ids mal rellenados."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


@dataclass
class WindowTable:
    """Ventanas de ECG (N, 1, 650) + quién, cuándo y haciendo qué.

    Guardamos los metadatos junto a las ventanas para poder particionar SIEMPRE por
    persona/sesión y no por ventana (ventanas vecinas se parecen mucho: mezclarlas
    entre train y test inflaría los resultados).
    """

    windows: np.ndarray
    participant_ids: np.ndarray
    session_ids: np.ndarray
    activities: np.ndarray
    source: str

    def __len__(self) -> int:
        return len(self.windows)

    def validate(self, window_samples: int = 650) -> None:
        w = self.windows
        if w.ndim != 3 or w.shape[1] != 1 or w.shape[2] != window_samples:
            raise DataError(
                f"Se esperaban ventanas de forma (N, 1, {window_samples}) y llegó {w.shape}"
            )
        for name in ("participant_ids", "session_ids", "activities"):
            if len(getattr(self, name)) != len(w):
                raise DataError(f"'{name}' no tiene la misma longitud que las ventanas")
        if not np.isfinite(w).all():
            raise DataError("Hay NaN o inf en las ventanas (¿pérdida de señal sin limpiar?)")
        if len(w) and (np.abs(w.mean(axis=2)).max() > 0.05 or np.abs(w.std(axis=2) - 1).max() > 0.05):
            # El contrato dice z-score por ventana. Datos en amplitud cruda harían que la red y la
            # línea base "señal cruda" trabajen sobre la amplitud y no sobre la forma del latido.
            raise DataError(
                "Las ventanas no están normalizadas con z-score (media 0, desviación 1 por ventana). "
                "Usa src/windowing.py::signal_to_windows para generarlas."
            )
        for pid in self.persons():
            sessions = self.sessions_of(pid)
            if sessions != sorted(sessions, key=_natural_key):
                raise DataError(
                    f"Los session_id de '{pid}' {sessions} no se ordenan cronológicamente como texto "
                    "(¿s10 antes que s2?). Rellena con ceros (s01, s02, ..., s10) o usa fechas ISO "
                    "(2026-10-01_s1)."
                )

    def subset(self, mask: np.ndarray) -> "WindowTable":
        return WindowTable(
            self.windows[mask],
            self.participant_ids[mask],
            self.session_ids[mask],
            self.activities[mask],
            self.source,
        )

    def persons(self) -> list[str]:
        return sorted(set(self.participant_ids.tolist()))

    def sessions_of(self, pid: str) -> list[str]:
        # Orden lexicográfico = cronológico (contrato: los session_id se ordenan por fecha).
        return sorted(set(self.session_ids[self.participant_ids == pid].tolist()))


def concat_tables(tables: list[WindowTable]) -> WindowTable:
    if not tables:
        raise DataError("No hay tablas que concatenar")
    sources = {t.source for t in tables}
    return WindowTable(
        np.concatenate([t.windows for t in tables]),
        np.concatenate([t.participant_ids for t in tables]),
        np.concatenate([t.session_ids for t in tables]),
        np.concatenate([t.activities for t in tables]),
        sources.pop() if len(sources) == 1 else "mixed",
    )


def save_window_file(path, windows, participant_id, session_id, activity, source) -> None:
    """Guarda UNA grabación ya procesada (ver contrato de datos en el plan)."""
    w = np.asarray(windows, dtype=np.float32)
    if w.ndim == 2:
        w = w[:, None, :]
    np.savez_compressed(
        path,
        windows=w,
        participant_id=participant_id,
        session_id=session_id,
        activity=activity,
        source=source,
    )


def load_window_dir(directory, window_samples: int = 650) -> WindowTable:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise DataError(f"No hay archivos .npz en {directory}")
    tables = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            w = z["windows"].astype(np.float32)
            if w.ndim == 2:
                w = w[:, None, :]
            n = len(w)
            tables.append(
                WindowTable(
                    w,
                    np.full(n, str(z["participant_id"])),
                    np.full(n, str(z["session_id"])),
                    np.full(n, str(z["activity"])),
                    str(z["source"]),
                )
            )
    table = concat_tables(tables)
    table.validate(window_samples)
    return table


def table_fingerprint(table: WindowTable) -> str:
    """Huella corta de qué datos se usaron; va en el JSON del checkpoint."""
    h = hashlib.sha256()
    pairs = sorted(set(zip(table.participant_ids.tolist(), table.session_ids.tolist())))
    for pid, sid in pairs:
        n = int(((table.participant_ids == pid) & (table.session_ids == sid)).sum())
        h.update(f"{pid}|{sid}|{n};".encode())
    return h.hexdigest()[:12]


@dataclass
class PolarSplit:
    train: WindowTable
    val: WindowTable
    holdout: WindowTable


def assert_no_holdout_leak(holdout_ids, tables: dict) -> None:
    """Falla si una persona reservada aparece en cualquier tabla de entrenamiento/validación."""
    for name, t in tables.items():
        bad = set(t.persons()) & set(holdout_ids)
        if bad:
            raise LeakageError(
                f"Participantes hold-out {sorted(bad)} aparecen en '{name}'. "
                "Eso invalida la evaluación con impostores no vistos."
            )


def assert_disjoint_people(tables: dict) -> None:
    """Ninguna persona puede estar en dos tablas (p. ej. público vs Polar)."""
    seen: dict[str, str] = {}
    for name, t in tables.items():
        for p in t.persons():
            if p in seen:
                raise LeakageError(f"La persona '{p}' está en '{seen[p]}' y en '{name}'")
            seen[p] = name


def split_polar(table: WindowTable, holdout_ids) -> PolarSplit:
    """Hold-out por persona; para el resto, la ÚLTIMA sesión es validación.

    Validar con una sesión posterior (otro día) mide lo que importa: que la huella
    sobreviva a cambios de colocación de la banda y de estado fisiológico.
    """
    present = set(table.persons())
    missing = [h for h in holdout_ids if h not in present]
    if missing:
        raise DataError(
            f"Los participantes hold-out {missing} no existen en los datos "
            f"(personas disponibles: {sorted(present)}). ¿Error de escritura en HOLDOUT_IDS?"
        )
    is_hold = np.isin(table.participant_ids, list(holdout_ids))
    holdout, rest = table.subset(is_hold), table.subset(~is_hold)

    single = [p for p in rest.persons() if len(rest.sessions_of(p)) < 2]
    if single:
        raise DataError(
            f"Las personas {single} tienen una sola sesión. Se necesitan al menos 2 sesiones "
            "en días distintos por persona (la última se usa como validación)."
        )
    train_mask = np.zeros(len(rest), dtype=bool)
    for p in rest.persons():
        last = rest.sessions_of(p)[-1]
        train_mask |= (rest.participant_ids == p) & (rest.session_ids != last)
    split = PolarSplit(rest.subset(train_mask), rest.subset(~train_mask), holdout)
    assert_no_holdout_leak(holdout_ids, {"train": split.train, "val": split.val})
    return split


def split_public(table: WindowTable, val_fraction: float, seed: int):
    """Separa personas (no ventanas) del dataset público para validar el preentrenamiento."""
    persons = table.persons()
    n_val = max(1, int(round(len(persons) * val_fraction)))
    if len(persons) - n_val < 2:
        raise DataError("Muy pocas personas públicas para separar entrenamiento y validación")
    rng = np.random.default_rng(seed)
    val_people = rng.choice(persons, size=n_val, replace=False).tolist()
    mask = np.isin(table.participant_ids, val_people)
    return table.subset(~mask), table.subset(mask)


class Augmenter:
    """Aumento de datos SOLO para entrenamiento (nunca en validación ni en evaluación).

    La idea: con pocas personas la red tiende a memorizar. Variar la señal de forma
    realista (otra colocación de banda, respiración, ruido) le enseña a fijarse en la
    forma del latido y no en detalles accidentales.
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _smooth_curve(self, n: int, strength: float, knots: int = 4) -> np.ndarray:
        """Curva suave alrededor de 1: pocos puntos aleatorios interpolados linealmente."""
        pos = np.linspace(0, n - 1, knots)
        vals = 1 + self.rng.uniform(-strength, strength, knots)
        return np.interp(np.arange(n), pos, vals)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = window[0].astype(np.float64)
        n = x.shape[0]
        idx = np.arange(n)
        if c.warp_strength > 0:
            # Deformación temporal: el ritmo "se acelera o frena" ligeramente.
            speed = self._smooth_curve(n, c.warp_strength)
            t = np.cumsum(speed)
            t = (t - t[0]) / (t[-1] - t[0]) * (n - 1)
            x = np.interp(t, idx, x)
        if c.magnitude_warp > 0:
            # Ganancia que varía suavemente (la ganancia global no serviría: el z-score la anula).
            x = x * self._smooth_curve(n, c.magnitude_warp)
        if c.wander_amplitude > 0:
            # Deriva de línea base por respiración / movimiento (0.1 a 0.5 Hz).
            f = self.rng.uniform(0.1, 0.5)
            phase = self.rng.uniform(0, 2 * np.pi)
            x = x + c.wander_amplitude * np.sin(2 * np.pi * f * idx / c.fs + phase)
        if c.jitter_std > 0:
            x = x + self.rng.normal(0, c.jitter_std, n)
        x = (x - x.mean()) / (x.std() + 1e-8)   # se conserva el contrato: ventana z-scored
        return x[None, :].astype(np.float32)


def make_label_map(table: WindowTable) -> dict:
    return {p: i for i, p in enumerate(table.persons())}


class WindowDataset(Dataset):
    def __init__(self, table: WindowTable, label_map: dict, augmenter: Augmenter | None = None):
        self.table = table
        self.augmenter = augmenter
        self.labels = np.array([label_map[p] for p in table.participant_ids], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int):
        w = self.table.windows[i]
        if self.augmenter is not None:
            w = self.augmenter(w)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), int(self.labels[i])


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)   # el orden de los lotes también es reproducible
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)
