from datetime import date

import numpy as np

from src.data import save_window_file
from src.session import main as session_main
from src.status import build_status, format_status, main as status_main
from src.synthetic import make_synthetic_table


def _record(tmp_path, pid, day):
    assert session_main(["--id", pid, "--simulate", "--no-open", "--today", day,
                         "--raw-dir", str(tmp_path / "raw"), "--processed-dir", str(tmp_path / "proc"),
                         "--discard-dir", str(tmp_path / "disc"), "--plots-dir", str(tmp_path / "plots")]) == 0


def test_status_counts_days_minutes_and_sorts_participants(tmp_path):
    _record(tmp_path, "P02", "2026-10-06")
    _record(tmp_path, "P01", "2026-10-06")
    _record(tmp_path, "P01", "2026-10-09")
    r = build_status(tmp_path / "proc")
    assert [p.pid for p in r.participants] == ["P01", "P02"]
    p01, p02 = r.participants
    assert p01.days == ["2026-10-06", "2026-10-09"] and p02.days == ["2026-10-06"]
    assert 4.5 < p02.minutes < 5.0                       # una sesión de ~5 min
    assert p01.minutes == np.float64(p01.minutes) and 9.0 < p01.minutes < 10.0
    assert p01.next_date == "2026-10-11" and not p01.complete
    assert r.total_minutes == p01.minutes + p02.minutes


def test_status_marks_participant_complete_after_three_days(tmp_path):
    for day in ("2026-10-06", "2026-10-09", "2026-10-12"):
        _record(tmp_path, "P01", day)
    p = build_status(tmp_path / "proc").participants[0]
    assert p.complete and p.next_date is None


def test_status_flags_incomplete_sessions(tmp_path):
    _record(tmp_path, "P01", "2026-10-06")
    (tmp_path / "proc" / "P01__2026-10-06__hablando.npz").unlink()
    p = build_status(tmp_path / "proc").participants[0]
    assert p.incomplete == {"2026-10-06": ["hablando"]}


def test_status_separates_files_outside_the_protocol(tmp_path):
    _record(tmp_path, "P01", "2026-10-06")
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3)
    save_window_file(tmp_path / "proc" / "juanC__2026-10-01_2000.npz", t.windows, "juanC", "2026-10-01_2000", "rest", "polar")
    r = build_status(tmp_path / "proc")
    assert [p.pid for p in r.participants] == ["P01"]
    assert r.out_of_protocol == ["juanC__2026-10-01_2000.npz"]


def test_format_status_tells_who_can_record_today_and_who_must_wait(tmp_path):
    _record(tmp_path, "P01", "2026-10-06")
    _record(tmp_path, "P02", "2026-10-08")
    r = build_status(tmp_path / "proc")
    text = format_status(r, today=date(2026, 10, 9))
    p01_line = next(line for line in text.splitlines() if line.startswith("P01"))
    p02_line = next(line for line in text.splitlines() if line.startswith("P02"))
    assert "toca ya" in p01_line and "2026-10-08" in p01_line          # pasaron >= 2 días
    assert "esperar" in p02_line and "2026-10-10" in p02_line          # solo ha pasado 1 día
    assert "2 personas" in text


def test_format_status_warns_about_incomplete_sessions_and_foreign_files(tmp_path):
    _record(tmp_path, "P01", "2026-10-06")
    (tmp_path / "proc" / "P01__2026-10-06__hablando.npz").unlink()
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3)
    save_window_file(tmp_path / "proc" / "kitzia__x.npz", t.windows, "kitzia", "x", "rest", "polar")
    text = format_status(build_status(tmp_path / "proc"), today=date(2026, 10, 7))
    assert "incompleta" in text and "hablando" in text
    assert "fuera del protocolo" in text and "kitzia__x.npz" in text


def test_cli_prints_table_and_handles_empty_folder(tmp_path, capsys):
    assert status_main(["--processed-dir", str(tmp_path / "vacio")]) == 0
    assert "no hay grabaciones" in capsys.readouterr().out.lower()
    _record(tmp_path, "P01", "2026-10-06")
    capsys.readouterr()
    assert status_main(["--processed-dir", str(tmp_path / "proc"), "--today", "2026-10-07"]) == 0
    assert "P01" in capsys.readouterr().out


def test_format_status_uses_singular_for_one_person(tmp_path):
    _record(tmp_path, "P01", "2026-10-06")
    text = format_status(build_status(tmp_path / "proc"), today=date(2026, 10, 7))
    assert "1 persona," in text and "1 personas" not in text
