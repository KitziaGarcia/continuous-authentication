"""Carga y calificación del banco JSON del examen; la clave no se envía al navegador."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi import HTTPException

LEVELS = {"secundaria", "preparatoria", "universidad"}
DIFFICULTIES = {"facil", "intermedia", "avanzada"}
QUESTION_COUNTS = {5, 10, 20, 30}
DEFAULT_BANK = Path(__file__).resolve().parent.parent / "data" / "exams" / "questions.json"


class ExamBank:
    def __init__(self, path: Path = DEFAULT_BANK):
        try:
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"No se pudo cargar el banco de preguntas {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("passages"), list):
            raise ValueError(f"El banco de preguntas {path} debe contener una lista 'passages'.")
        self.questions: dict[str, dict[str, Any]] = {}
        for passage in data["passages"]:
            self._add_passage(passage)
        if not self.questions:
            raise ValueError(f"El banco de preguntas {path} no contiene preguntas.")

    def _add_passage(self, passage: dict[str, Any]) -> None:
        required = {"id", "title", "text", "level", "questions"}
        if not isinstance(passage, dict) or not required <= passage.keys():
            raise ValueError("Cada lectura debe incluir id, title, text, level y questions.")
        if passage["level"] not in LEVELS:
            raise ValueError(f"Nivel de lectura inválido: {passage['level']!r}.")
        for question in passage["questions"]:
            if not isinstance(question, dict):
                raise ValueError(f"Pregunta inválida en la lectura {passage['id']}.")
            if not {"id", "difficulty", "prompt", "choices", "answer"} <= question.keys():
                raise ValueError(f"Pregunta incompleta en la lectura {passage['id']}.")
            if question["id"] in self.questions:
                raise ValueError(f"ID de pregunta duplicado: {question['id']}.")
            if question["difficulty"] not in DIFFICULTIES:
                raise ValueError(f"Dificultad inválida en la pregunta {question['id']}.")
            if set(question["choices"]) != {"A", "B", "C", "D"} or question["answer"] not in question["choices"]:
                raise ValueError(f"Opciones o respuesta inválidas en la pregunta {question['id']}.")
            self.questions[question["id"]] = {
                **question,
                "passage_id": passage["id"],
                "passage_title": passage["title"],
                "passage_text": passage["text"],
                "level": passage["level"],
            }

    def select(self, level: str, difficulty: str, count: int) -> list[dict[str, Any]]:
        if level not in LEVELS | {"mixto"}:
            raise HTTPException(422, "Nivel de examen inválido.")
        if difficulty not in DIFFICULTIES | {"todas"}:
            raise HTTPException(422, "Dificultad inválida.")
        if count not in QUESTION_COUNTS:
            raise HTTPException(422, "La cantidad debe ser 5, 10, 20 o 30 preguntas.")
        eligible = [
            question for question in self.questions.values()
            if (level == "mixto" or question["level"] == level)
            and (difficulty == "todas" or question["difficulty"] == difficulty)
        ]
        if len(eligible) < count:
            raise HTTPException(
                422,
                f"Hay {len(eligible)} preguntas para esa combinación; se solicitaron {count}.",
            )
        chosen = random.SystemRandom().sample(eligible, count)
        chosen.sort(key=lambda item: (item["passage_id"], item["id"]))
        return [
            {key: question[key] for key in (
                "id", "prompt", "choices", "difficulty", "level",
                "passage_id", "passage_title", "passage_text",
            )}
            for question in chosen
        ]

    def grade(self, answers: list[dict[str, str | None]]) -> dict[str, Any]:
        seen: set[str] = set()
        results = []
        for answer in answers:
            question_id, selected = answer["question_id"], answer["answer"]
            question = self.questions.get(question_id)
            if question is None or question_id in seen or selected not in {"A", "B", "C", "D", None}:
                raise HTTPException(422, "Respuestas inválidas o repetidas.")
            seen.add(question_id)
            results.append({
                "question_id": question_id,
                "correct": selected is not None and selected == question["answer"],
                "answer": question["answer"],
            })
        correct = sum(result["correct"] for result in results)
        total = len(results)
        return {
            "correct": correct,
            "total": total,
            "percentage": round(correct / total * 100) if total else 0,
            "results": results,
        }
