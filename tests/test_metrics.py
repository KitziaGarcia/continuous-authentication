import pytest

from src.metrics import (
    _score_and_label,
    calculate_auc,
    calculate_detection_times,
    calculate_eer,
    calculate_operating_rates,
)


def test_calculate_eer_finds_perfectly_separated_operating_point():
    result = calculate_eer(
        scores=[0.9, 0.8, 0.2, 0.1],
        genuine=[True, True, False, False],
    )

    assert result["eer"] == 0
    assert 0.2 < result["threshold"] < 0.8
    assert result["genuine_trials"] == 2
    assert result["impostor_trials"] == 2


def test_calculate_eer_rejects_missing_ground_truth_class():
    with pytest.raises(ValueError, match="genuine y una impostor"):
        calculate_eer([0.9, 0.8], [True, True])


def test_score_label_filter_excludes_unknown_and_bad_signal():
    readings = [
        {"ground_truth": "genuine", "status": "AUTHENTICATED", "smoothed_similarity": "0.8"},
        {"ground_truth": "impostor", "status": "REJECTED", "smoothed_similarity": "0.2"},
        {"ground_truth": "unknown", "status": "AUTHENTICATED", "smoothed_similarity": "0.9"},
        {"ground_truth": "genuine", "status": "BAD_SIGNAL", "smoothed_similarity": ""},
    ]

    scores, genuine, ignored = _score_and_label(readings, "smoothed_similarity")
    assert scores == [0.8, 0.2]
    assert genuine == [True, False]
    assert ignored == 2


def test_auc_measures_ranking_and_handles_ties():
    assert calculate_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1
    assert calculate_auc([0.5, 0.5], [True, False]) == 0.5


def test_operating_far_frr_use_model_decisions_and_exclude_bad_signal():
    result = calculate_operating_rates([
        {"ground_truth": "genuine", "status": "REJECTED", "threshold": "0.31"},
        {"ground_truth": "genuine", "status": "AUTHENTICATED", "threshold": "0.31"},
        {"ground_truth": "impostor", "status": "AUTHENTICATED", "threshold": "0.31"},
        {"ground_truth": "impostor", "status": "REJECTED", "threshold": "0.31"},
        {"ground_truth": "genuine", "status": "BAD_SIGNAL", "threshold": "0.31"},
    ])

    assert result["far"] == 0.5
    assert result["frr"] == 0.5
    assert result["operating_threshold"] == 0.31
    assert result["false_accepts"] == 1
    assert result["false_rejects"] == 1


def test_detection_latency_is_measured_for_each_impostor_episode():
    result = calculate_detection_times([
        {"session_id": "s1", "time_seconds": "0", "ground_truth": "genuine", "status": "AUTHENTICATED"},
        {"session_id": "s1", "time_seconds": "2.5", "ground_truth": "impostor", "status": "AUTHENTICATED"},
        {"session_id": "s1", "time_seconds": "5", "ground_truth": "impostor", "status": "REJECTED"},
        {"session_id": "s1", "time_seconds": "7.5", "ground_truth": "genuine", "status": "AUTHENTICATED"},
        {"session_id": "s2", "time_seconds": "0", "ground_truth": "impostor", "status": "AUTHENTICATED"},
    ])

    assert result["impostor_episodes"] == 2
    assert result["detected_episodes"] == 1
    assert result["undetected_episodes"] == 1
    assert result["mean_seconds"] == 2.5
    assert result["median_seconds"] == 2.5
