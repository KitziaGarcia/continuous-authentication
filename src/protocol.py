"""Reglas del protocolo de grabación (sin hardware): IDs, actividades y qué toca grabar hoy.

Idea central: el ID es de la PERSONA y nunca cambia (P01, P02...). Lo que cambia entre grabaciones es la
sesión (la fecha). Así el programa nunca confunde "segunda vez de P03" con "otra persona".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ID_PATTERN = re.compile(r"P\d{2,3}")
TARGET_DAYS = 3          # sesiones (días distintos) por persona
MIN_GAP_DAYS = 2         # separación ideal entre sesiones


@dataclass(frozen=True)
class Segment:
    activity: str          # va en el nombre de los archivos: sin espacios ni acentos
    seconds: float
    instruction: str


SCHEDULE = (
    Segment("reposo", 120, "Siéntate derecho, respira normal y no te muevas ni hables."),
    Segment("lectura", 90, "Lee en silencio cualquier texto (celular o libro), sin hablar."),
    Segment("problemas", 60, "Resuelve de cabeza: 100-7, luego -7 otra vez, y sigue restando 7."),
    Segment("hablando", 30, "Habla con normalidad (cuenta qué hiciste hoy) sin moverte mucho."),
)


@dataclass
class SessionPlan:
    session_id: str
    segments: tuple
    notes: list = field(default_factory=list)
    mode: str = "new"      # new | continue | redo | another


def validate_id(pid: str) -> str:
    if not ID_PATTERN.fullmatch(pid):
        raise ValueError(
            f"ID inválido '{pid}'. Usa el formato P01, P02, ... (letra P mayúscula y el número de la persona). "
            "El ID es de la persona y es siempre el mismo; lo que cambia entre grabaciones es el día."
        )
    return pid


def existing_sessions(processed_dir, pid: str) -> dict:
    """{sesión: {actividades ya grabadas}} leyendo los nombres de archivo P03__2026-10-06__reposo.npz."""
    out: dict = {}
    for f in Path(processed_dir).glob(f"{pid}__*__*.npz"):
        parts = f.stem.split("__")
        if len(parts) == 3:
            out.setdefault(parts[1], set()).add(parts[2])
    return out


def plan_session(pid: str, today: date, existing: dict, redo, another_sitting: bool) -> SessionPlan:
    """Decide qué se graba hoy según lo que ya existe, sin dejar que se mezclen sesiones por error."""
    today_s = today.isoformat()
    today_sessions = sorted(s for s in existing if s[:10] == today_s)
    by_name = {s.activity: s for s in SCHEDULE}
    notes: list = []
    earlier = sorted(s[:10] for s in existing if s[:10] < today_s)
    if earlier:
        gap = (today - date.fromisoformat(earlier[-1])).days
        if gap < MIN_GAP_DAYS:
            notes.append(f"La última sesión de {pid} fue hace {gap} día(s); lo ideal es separarlas "
                         "2 o 3 días (puedes continuar igual).")
    if redo:
        unknown = [a for a in redo if a not in by_name]
        if unknown:
            raise ValueError(f"Actividad desconocida: {unknown}. Opciones: {', '.join(by_name)}.")
        if not today_sessions:
            raise ValueError(f"{pid} no tiene sesión de hoy: no hay nada que repetir. "
                             f"Corre sin --redo para grabar la sesión completa.")
        return SessionPlan(today_sessions[-1], tuple(by_name[a] for a in redo), notes, "redo")
    if another_sitting:
        if not today_sessions:
            raise ValueError("--another-sitting solo aplica si la persona ya grabó hoy; "
                             "corre sin esa opción para la primera sesión del día.")
        return SessionPlan(f"{today_s}_{len(today_sessions) + 1}", SCHEDULE, notes, "another")
    if not today_sessions:
        return SessionPlan(today_s, SCHEDULE, notes, "new")
    last = today_sessions[-1]
    missing = tuple(s for s in SCHEDULE if s.activity not in existing[last])
    if not missing:
        raise ValueError(
            f"{pid} ya completó la sesión de hoy ({last}). Si una actividad salió mal, repítela con "
            f"--redo <actividad> (opciones: {', '.join(by_name)}). Si te quitaste y volviste a poner la "
            "banda y es otra sentada, usa --another-sitting."
        )
    notes.append("La sesión de hoy estaba incompleta; faltan: " + ", ".join(s.activity for s in missing) + ".")
    return SessionPlan(last, missing, notes, "continue")


def participant_days(processed_dir, pid: str) -> list:
    """Fechas distintas (AAAA-MM-DD) en las que la persona tiene al menos una grabación."""
    return sorted({s[:10] for s in existing_sessions(processed_dir, pid)})
