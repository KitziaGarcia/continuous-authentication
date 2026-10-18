import pytest
from fastapi import HTTPException

from src.exam import ExamBank


def test_selected_exam_questions_do_not_include_the_answer_key():
    bank = ExamBank()
    questions = bank.select("mixto", "todas", 10)

    assert len(questions) == 10
    assert all("answer" not in question for question in questions)
    assert all("passage_text" in question and "choices" in question for question in questions)


def test_exam_grading_returns_score_and_correct_answers():
    bank = ExamBank()
    grade = bank.grade([
        {"question_id": "s01", "answer": "B"},
        {"question_id": "s02", "answer": "D"},
        {"question_id": "s03", "answer": None},
    ])

    assert grade["correct"] == 1
    assert grade["total"] == 3
    assert grade["percentage"] == 33
    assert grade["results"] == [
        {"question_id": "s01", "correct": True, "answer": "B"},
        {"question_id": "s02", "correct": False, "answer": "A"},
        {"question_id": "s03", "correct": False, "answer": "C"},
    ]


def test_grading_rejects_unknown_or_duplicate_questions():
    bank = ExamBank()

    with pytest.raises(HTTPException):
        bank.grade([{"question_id": "unknown", "answer": "A"}])
    with pytest.raises(HTTPException):
        bank.grade([{"question_id": "s01", "answer": "A"}, {"question_id": "s01", "answer": "B"}])


def test_level_and_question_count_are_validated():
    bank = ExamBank()

    with pytest.raises(HTTPException):
        bank.select("doctorado", "todas", 5)
    with pytest.raises(HTTPException):
        bank.select("mixto", "todas", 31)
    with pytest.raises(HTTPException):
        bank.select("mixto", "todas", 12)


@pytest.mark.parametrize("level", ["secundaria", "preparatoria", "universidad"])
@pytest.mark.parametrize("difficulty", ["facil", "intermedia", "avanzada"])
def test_bank_supports_30_questions_for_every_level_and_difficulty(level, difficulty):
    bank = ExamBank()

    available = sum(
        question["level"] == level and question["difficulty"] == difficulty
        for question in bank.questions.values()
    )
    assert available == 30

    questions = bank.select(level, difficulty, 30)

    assert len(questions) == 30
    assert all(question["level"] == level for question in questions)
    assert all(question["difficulty"] == difficulty for question in questions)


@pytest.mark.parametrize("count", [5, 10, 20, 30])
def test_bank_returns_exactly_the_requested_supported_count(count):
    bank = ExamBank()

    assert len(bank.select("secundaria", "todas", count)) == count
