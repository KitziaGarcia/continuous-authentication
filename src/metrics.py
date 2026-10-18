"""Calculate an exploratory EER from labelled exam metric CSV files."""
from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path


SCORED_STATUSES = {"AUTHENTICATED", "REJECTED"}


def load_readings(root: Path) -> tuple[list[dict[str, str]], int]:
    """Load metric rows from completed and interrupted exam CSVs."""
    readings: list[dict[str, str]] = []
    ignored = 0
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                readings.append(row)
    return readings, ignored


def _score_and_label(readings: list[dict[str, str]], score_column: str) -> tuple[list[float], list[bool], int]:
    scores: list[float] = []
    genuine: list[bool] = []
    ignored = 0
    for row in readings:
        label = row.get("ground_truth")
        if label not in {"genuine", "impostor"} or row.get("status") not in SCORED_STATUSES:
            ignored += 1
            continue
        try:
            score = float(row.get(score_column, ""))
        except (TypeError, ValueError):
            ignored += 1
            continue
        if not math.isfinite(score):
            ignored += 1
            continue
        scores.append(score)
        genuine.append(label == "genuine")
    return scores, genuine, ignored


def calculate_eer(scores: list[float], genuine: list[bool]) -> dict[str, float | int]:
    """Interpolate the empirical FAR/FRR crossing over all unique score boundaries."""
    if len(scores) != len(genuine):
        raise ValueError("Scores and labels must have the same number of items.")
    genuine_scores = [score for score, is_genuine in zip(scores, genuine) if is_genuine]
    impostor_scores = [score for score, is_genuine in zip(scores, genuine) if not is_genuine]
    if not genuine_scores or not impostor_scores:
        raise ValueError("Se necesitan al menos una lectura genuine y una impostor.")

    unique_scores = sorted(set(scores))
    score_range = unique_scores[-1] - unique_scores[0]
    margin = max(score_range * 1e-9, 1e-12)
    thresholds = [unique_scores[-1] + margin]
    thresholds.extend(
        (left + right) / 2
        for left, right in reversed(list(zip(unique_scores, unique_scores[1:])))
    )
    thresholds.append(unique_scores[0] - margin)

    far = [
        sum(score >= threshold for score in impostor_scores) / len(impostor_scores)
        for threshold in thresholds
    ]
    frr = [
        sum(score < threshold for score in genuine_scores) / len(genuine_scores)
        for threshold in thresholds
    ]
    gaps = [false_accept - false_reject for false_accept, false_reject in zip(far, frr)]

    for index, gap in enumerate(gaps):
        if gap == 0:
            return {
                "eer": far[index],
                "threshold": thresholds[index],
                "genuine_trials": len(genuine_scores),
                "impostor_trials": len(impostor_scores),
            }
        if index + 1 < len(gaps) and gap < 0 < gaps[index + 1]:
            fraction = -gap / (gaps[index + 1] - gap)
            eer = far[index] + fraction * (far[index + 1] - far[index])
            threshold = thresholds[index] + fraction * (thresholds[index + 1] - thresholds[index])
            return {
                "eer": eer,
                "threshold": threshold,
                "genuine_trials": len(genuine_scores),
                "impostor_trials": len(impostor_scores),
            }

    raise RuntimeError("No se encontró el cruce entre FAR y FRR.")


def calculate_auc(scores: list[float], genuine: list[bool]) -> float:
    """Calculate ROC AUC as the probability a genuine score outranks an impostor score."""
    if len(scores) != len(genuine):
        raise ValueError("Scores and labels must have the same number of items.")
    genuine_scores = [score for score, is_genuine in zip(scores, genuine) if is_genuine]
    impostor_scores = sorted(score for score, is_genuine in zip(scores, genuine) if not is_genuine)
    if not genuine_scores or not impostor_scores:
        raise ValueError("Se necesitan al menos una lectura genuine y una impostor.")

    concordant_pairs = 0.0
    for score in genuine_scores:
        below = bisect.bisect_left(impostor_scores, score)
        tied = bisect.bisect_right(impostor_scores, score) - below
        concordant_pairs += below + 0.5 * tied
    return concordant_pairs / (len(genuine_scores) * len(impostor_scores))


def calculate_operating_rates(readings: list[dict[str, str]]) -> dict[str, float | int | None]:
    """Calculate observed FAR/FRR from the actual authenticated/rejected decisions."""
    genuine = [row for row in readings if row.get("ground_truth") == "genuine"
               and row.get("status") in SCORED_STATUSES]
    impostor = [row for row in readings if row.get("ground_truth") == "impostor"
                and row.get("status") in SCORED_STATUSES]
    thresholds = []
    for row in readings:
        try:
            threshold = float(row.get("threshold", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(threshold):
            thresholds.append(threshold)

    false_accepts = sum(row["status"] == "AUTHENTICATED" for row in impostor)
    false_rejects = sum(row["status"] == "REJECTED" for row in genuine)
    return {
        "far": false_accepts / len(impostor) if impostor else None,
        "frr": false_rejects / len(genuine) if genuine else None,
        "operating_threshold": sum(thresholds) / len(thresholds) if thresholds else None,
        "genuine_decisions": len(genuine),
        "impostor_decisions": len(impostor),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def calculate_detection_times(readings: list[dict[str, str]]) -> dict[str, float | int | None]:
    """Measure first REJECTED latency within every contiguous impostor-labelled episode."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for index, row in enumerate(readings):
        grouped.setdefault(row.get("session_id") or f"row-{index}", []).append(row)

    detections: list[float] = []
    episodes = 0
    undetected = 0
    for session_rows in grouped.values():
        episode: list[dict[str, str]] = []
        for row in [*session_rows, {"ground_truth": "unknown"}]:
            if row.get("ground_truth") == "impostor":
                episode.append(row)
                continue
            if episode:
                episodes += 1
                detected_at = next(
                    (item for item in episode if item.get("status") == "REJECTED"),
                    None,
                )
                if detected_at is None:
                    undetected += 1
                else:
                    try:
                        latency = float(detected_at["time_seconds"]) - float(episode[0]["time_seconds"])
                    except (KeyError, TypeError, ValueError):
                        undetected += 1
                    else:
                        if math.isfinite(latency) and latency >= 0:
                            detections.append(latency)
                        else:
                            undetected += 1
                episode = []

    return {
        "impostor_episodes": episodes,
        "detected_episodes": len(detections),
        "undetected_episodes": undetected,
        "mean_seconds": sum(detections) / len(detections) if detections else None,
        "median_seconds": (
            (sorted(detections)[(len(detections) - 1) // 2] + sorted(detections)[len(detections) // 2]) / 2
            if detections else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.metrics",
        description="Calcula métricas exploratorias desde etiquetas genuine/impostor en CSV.",
    )
    parser.add_argument("--root", type=Path, default=Path("metrics"))
    parser.add_argument("--score-column", choices=("similarity", "smoothed_similarity"),
                        default="smoothed_similarity")
    args = parser.parse_args(argv)

    if not args.root.exists():
        parser.error(f"No existe el directorio de métricas: {args.root}")
    readings, _ = load_readings(args.root)
    scores, genuine, ignored = _score_and_label(readings, args.score_column)
    try:
        result = calculate_eer(scores, genuine)
        auc = calculate_auc(scores, genuine)
    except ValueError as exc:
        parser.error(str(exc))
    operating = calculate_operating_rates(readings)
    detection = calculate_detection_times(readings)

    print(f"EER empírico: {result['eer'] * 100:.2f}%")
    print(f"Umbral interpolado: {result['threshold']:.6f}")
    print(f"AUC ROC: {auc:.4f}")
    far = operating["far"]
    frr = operating["frr"]
    print(f"FAR al umbral operativo: {far * 100:.2f}%" if far is not None else "FAR: n/a (sin decisiones impostoras)")
    print(f"FRR al umbral operativo: {frr * 100:.2f}%" if frr is not None else "FRR: n/a (sin decisiones genuinas)")
    if operating["operating_threshold"] is not None:
        print(f"Umbral operativo medio guardado: {operating['operating_threshold']:.6f}")
    print(f"Lecturas genuine: {result['genuine_trials']}")
    print(f"Lecturas impostor: {result['impostor_trials']}")
    print(
        f"Detección de impostores: {detection['detected_episodes']}/{detection['impostor_episodes']} episodios; "
        f"{detection['undetected_episodes']} no detectados"
    )
    if detection["mean_seconds"] is not None:
        print(f"Tiempo medio de detección: {detection['mean_seconds']:.2f} s")
        print(f"Tiempo mediano de detección: {detection['median_seconds']:.2f} s")
    else:
        print("Tiempo de detección: n/a (ningún episodio impostor fue detectado)")
    print(f"Filas ignoradas (unknown, mala señal o sin puntuación): {ignored}")
    print("Nota: ventanas consecutivas están correlacionadas; FAR/FRR/AUC/EER y latencias son exploratorias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())