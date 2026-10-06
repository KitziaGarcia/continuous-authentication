"""Control de avance de la recolección: quién ha grabado qué días y a quién le toca volver.

Uso:  python -m src.status
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.config import Config
from src.protocol import ID_PATTERN, MIN_GAP_DAYS, SCHEDULE, TARGET_DAYS


@dataclass
class ParticipantStatus:
    pid: str
    sessions: dict = field(default_factory=dict)       # sesión -> actividades grabadas
    minutes: float = 0.0
    days: list = field(default_factory=list)
    incomplete: dict = field(default_factory=dict)     # sesión -> actividades que faltan
    complete: bool = False
    next_date: str | None = None


@dataclass
class StatusReport:
    participants: list
    out_of_protocol: list
    total_minutes: float


def build_status(processed_dir, cfg: Config | None = None) -> StatusReport:
    cfg = cfg or Config()
    step_s = cfg.window_samples * (1 - cfg.window_overlap) / cfg.fs
    win_s = cfg.window_samples / cfg.fs
    by_pid: dict = {}
    foreign: list = []
    folder = Path(processed_dir)
    for f in sorted(folder.glob("*.npz")) if folder.exists() else []:
        parts = f.stem.split("__")
        if len(parts) != 3 or not ID_PATTERN.fullmatch(parts[0]):
            foreign.append(f.name)
            continue
        pid, session, activity = parts
        with np.load(f, allow_pickle=False) as z:
            n = int(z["windows"].shape[0])
        st = by_pid.setdefault(pid, ParticipantStatus(pid))
        st.sessions.setdefault(session, set()).add(activity)
        st.minutes += (((n - 1) * step_s + win_s) / 60.0) if n > 0 else 0.0
    participants = []
    for pid in sorted(by_pid):
        st = by_pid[pid]
        st.days = sorted({s[:10] for s in st.sessions})
        st.incomplete = {s: [a.activity for a in SCHEDULE if a.activity not in acts]
                         for s, acts in sorted(st.sessions.items()) if any(a.activity not in acts for a in SCHEDULE)}
        st.complete = len(st.days) >= TARGET_DAYS
        st.next_date = None if st.complete else (
            date.fromisoformat(st.days[-1]) + timedelta(days=MIN_GAP_DAYS)).isoformat()
        participants.append(st)
    return StatusReport(participants, foreign, sum(p.minutes for p in participants))


def format_status(report: StatusReport, today: date) -> str:
    if not report.participants and not report.out_of_protocol:
        return "No hay grabaciones todavía. Empieza con:  python -m src.session --id P01"
    lines = ["", f"{'ID':<6}{'días':<7}{'fechas':<34}{'min':<7}estado"]
    for p in report.participants:
        if p.complete:
            estado = "completo"
        elif today >= date.fromisoformat(p.next_date):
            estado = f"toca ya la sesión {len(p.days) + 1} (desde {p.next_date})"
        else:
            estado = f"esperar hasta {p.next_date} para la sesión {len(p.days) + 1}"
        lines.append(f"{p.pid:<6}{f'{len(p.days)}/{TARGET_DAYS}':<7}{', '.join(p.days):<34}{p.minutes:<7.1f}{estado}")
    done = sum(p.complete for p in report.participants)
    n = len(report.participants)
    lines += ["", f"{n} {'persona' if n == 1 else 'personas'}, {done} con los {TARGET_DAYS} días completos, "
                  f"{report.total_minutes:.0f} minutos grabados en total."]
    for p in report.participants:
        for session, missing in p.incomplete.items():
            lines.append(f"ATENCIÓN: {p.pid} tiene la sesión {session} incompleta: falta {', '.join(missing)}. "
                         f"Corre:  python -m src.session --id {p.pid}")
    if report.out_of_protocol:
        lines += ["", "Archivos fuera del protocolo (no cuentan en este estado; si son de pruebas, muévelos a "
                      "data/piloto/):"] + [f"  {name}" for name in report.out_of_protocol]
    return "\n".join(lines)


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.status", description="Avance de la recolección de datos")
    p.add_argument("--processed-dir", default=f"{cfg.data_dir}/polar")
    p.add_argument("--today", default=None, help="fecha AAAA-MM-DD (solo para pruebas)")
    a = p.parse_args(argv)
    today = date.fromisoformat(a.today) if a.today else date.today()
    print(format_status(build_status(a.processed_dir, cfg), today))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
