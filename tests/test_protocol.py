from datetime import date

import pytest

from src.protocol import SCHEDULE, existing_sessions, plan_session, validate_id

TODAY = date(2026, 10, 6)


def test_schedule_is_five_minutes_in_four_activities():
    assert [s.activity for s in SCHEDULE] == ["reposo", "lectura", "problemas", "hablando"]
    assert sum(s.seconds for s in SCHEDULE) == 300


@pytest.mark.parametrize("pid", ["P01", "P12", "P100"])
def test_valid_ids(pid):
    assert validate_id(pid) == pid


@pytest.mark.parametrize("pid", ["juanC", "kitzia2", "P1", "p01", "P01a", "P03 ", "01", "P", "P01_2"])
def test_invalid_ids_explain_the_format(pid):
    with pytest.raises(ValueError, match="P01"):
        validate_id(pid)


def test_existing_sessions_reads_file_names_and_ignores_other_files(tmp_path):
    for name in ("P03__2026-10-01__reposo", "P03__2026-10-01__lectura", "P03__2026-10-04__reposo",
                 "P04__2026-10-01__reposo", "juanC__2026-10-01_2000"):
        (tmp_path / f"{name}.npz").write_bytes(b"x")
    assert existing_sessions(tmp_path, "P03") == {"2026-10-01": {"reposo", "lectura"}, "2026-10-04": {"reposo"}}
    assert existing_sessions(tmp_path / "no_existe", "P03") == {}


def test_fresh_participant_records_everything_in_a_session_named_after_today():
    p = plan_session("P01", TODAY, {}, None, False)
    assert p.session_id == "2026-10-06" and p.mode == "new" and p.segments == SCHEDULE and p.notes == []


def test_incomplete_session_today_continues_with_the_missing_activities():
    p = plan_session("P01", TODAY, {"2026-10-06": {"reposo", "lectura"}}, None, False)
    assert p.mode == "continue" and p.session_id == "2026-10-06"
    assert [s.activity for s in p.segments] == ["problemas", "hablando"]
    assert any("faltan" in n.lower() for n in p.notes)


def test_complete_session_today_refuses_and_explains_options():
    done = {"2026-10-06": {s.activity for s in SCHEDULE}}
    with pytest.raises(ValueError) as e:
        plan_session("P01", TODAY, done, None, False)
    assert "--redo" in str(e.value) and "--another-sitting" in str(e.value)


def test_redo_repeats_only_the_chosen_activities_in_todays_session():
    done = {"2026-10-06": {s.activity for s in SCHEDULE}}
    p = plan_session("P01", TODAY, done, ["lectura", "reposo"], False)
    assert p.mode == "redo" and p.session_id == "2026-10-06"
    assert [s.activity for s in p.segments] == ["lectura", "reposo"]


def test_redo_without_a_session_today_is_an_error():
    with pytest.raises(ValueError, match="nada que repetir"):
        plan_session("P01", TODAY, {}, ["reposo"], False)


def test_redo_rejects_unknown_activity():
    with pytest.raises(ValueError, match="bailando"):
        plan_session("P01", TODAY, {"2026-10-06": {"reposo"}}, ["bailando"], False)


def test_another_sitting_opens_a_new_session_that_still_sorts_after_the_first():
    done = {"2026-10-06": {s.activity for s in SCHEDULE}}
    p = plan_session("P01", TODAY, done, None, True)
    assert p.mode == "another" and p.session_id == "2026-10-06_2" and p.segments == SCHEDULE
    p3 = plan_session("P01", TODAY, {**done, "2026-10-06_2": {"reposo"}}, None, True)
    assert p3.session_id == "2026-10-06_3"
    assert sorted(["2026-10-06_2", "2026-10-06"]) == ["2026-10-06", "2026-10-06_2"]


def test_another_sitting_without_a_session_today_is_an_error():
    with pytest.raises(ValueError, match="another-sitting"):
        plan_session("P01", TODAY, {}, None, True)


def test_note_when_previous_session_was_less_than_two_days_ago():
    p = plan_session("P01", TODAY, {"2026-10-05": {"reposo"}}, None, False)
    assert any("2 o 3 días" in n for n in p.notes)
    assert plan_session("P01", TODAY, {"2026-10-03": {"reposo"}}, None, False).notes == []
