import csv
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from src.api import ScenarioSource, Controller, create_app, quality_codes
from src.checkpoint import save_checkpoint
from src.config import Config
from src.evaluate import make_model_embed_fn
from src.model import ECGEmbeddingNet
from src.quality import check_quality
from src.session import ConnectionLost


@pytest.fixture
def client(tmp_path):
    cfg = Config()
    net = ECGEmbeddingNet(cfg).eval()
    ctl = Controller(make_model_embed_fn(net), cfg, "modelo_test.pt", threshold=0.31, simulate=True, speed=300.0,
                     template_dir=tmp_path / "live")
    with TestClient(create_app(ctl, metrics_dir=tmp_path / "metrics")) as c:
        yield c


def wait_idle(c, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/state").json()
        if s["phase"] == "idle":
            return s
        time.sleep(0.05)
    raise AssertionError("no volvió a idle")


def collect_until(ws, stop_type, limit=500):
    events = []
    for _ in range(limit):
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == stop_type:
            return events
    raise AssertionError(f"no llegó {stop_type}")


def test_health_and_unknown_user(client):
    h = client.get("/health").json()
    assert h["mode"] == "simulate" and h["threshold"] == 0.31
    assert client.get("/users/ana").json() == {"user_id": "ana", "enrolled": False}
    r = client.post("/monitor/start", json={"user_id": "ana"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "NOT_ENROLLED"
    assert client.post("/enroll/start", json={"user_id": "mal id!"}).status_code == 422


def test_loopback_cors_allows_localhost_and_ip_aliases(tmp_path):
    cfg = Config()
    net = ECGEmbeddingNet(cfg).eval()
    ctl = Controller(make_model_embed_fn(net), cfg, "modelo_test.pt", simulate=True, template_dir=tmp_path / "live")
    with TestClient(create_app(ctl, cors_origins=["http://127.0.0.1:5173"])) as c:
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            response = c.options(
                "/health",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin


def test_real_backend_waits_for_ecg_before_marking_sensor_connected(monkeypatch):
    class Source:
        stopped = False

        async def start(self):
            pass

        async def stop(self):
            self.stopped = True

    source = Source()
    waited = []
    monkeypatch.setattr("src.api.PolarSource", lambda: source)

    async def wait_for_ecg(src):
        waited.append(src)
        raise ConnectionLost("ECG stream did not start")

    monkeypatch.setattr("src.api.wait_for_ecg", wait_for_ecg)
    ctl = Controller(lambda _: None, Config(), "modelo_test.pt")
    with pytest.raises(ConnectionLost, match="ECG stream did not start"):
        asyncio.run(ctl.open_source())

    assert waited == [source]
    assert source.stopped
    assert not ctl.state["connected"]
    assert ctl.state["last_error"]["code"] == "CONNECTION_LOST"


def test_enroll_progress_and_result_over_websocket(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/enroll/start", json={"user_id": "ana"}).status_code == 202
        events = collect_until(ws, "enroll_result")
    stages = {e["stage"] for e in events if e["type"] == "enroll_progress"}
    assert stages == {"settling", "recording"}
    result = events[-1]
    assert result["ok"] is True and result["consecutive_failures"] == 0
    assert wait_idle(client)["connected"] is True
    assert client.get("/users/ana").json()["enrolled"] is True


def test_three_failed_enrollments_are_counted_with_codes(client):
    client.post("/sim/scenario", json={"scenario": "bad_signal"})
    for expected in (1, 2, 3):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            client.post("/enroll/start", json={"user_id": "ana"})
            result = collect_until(ws, "enroll_result")[-1]
        assert result["ok"] is False and result["codes"] == ["FLAT_SIGNAL"]
        assert result["consecutive_failures"] == expected and result["messages"]
        wait_idle(client)
    assert client.get("/state").json()["enroll"]["consecutive_failures"] == 3


def test_monitor_emits_decisions_with_streaks(client):
    client.post("/enroll/start", json={"user_id": "ana"})
    wait_idle(client)
    client.post("/sim/scenario", json={"scenario": "bad_signal"})
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/monitor/start", json={"user_id": "ana"}).status_code == 202
        seen = []
        while len(seen) < 25:
            ev = ws.receive_json()
            if ev["type"] == "decision":
                seen.append(ev)
        client.post("/stop")
    keys = {"t", "status", "authenticated", "score", "smoothed", "threshold", "bad_signal_streak",
            "rejection_streak", "hr_bpm", "codes"}
    assert keys <= set(seen[-1])
    assert seen[-1]["status"] == "BAD_SIGNAL" and seen[-1]["authenticated"] is None
    assert seen[-1]["bad_signal_streak"] >= 3 and seen[-1]["codes"] == ["FLAT_SIGNAL"]
    assert client.get("/state").json()["phase"] == "idle"


def test_second_operation_while_busy_is_rejected(client):
    assert client.post("/enroll/start", json={"user_id": "ana"}).status_code == 202
    r = client.post("/enroll/start", json={"user_id": "bea"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "BUSY"
    client.post("/stop")


def test_movement_scenario_produces_movement_code():
    clock = {"t": 0.0}
    src = ScenarioSource(lambda: clock["t"], scenario="movement")
    clock["t"] = 20.0
    import asyncio
    asyncio.run(src.start())
    clock["t"] = 40.0
    raw = src.take(0, src.n_samples())
    q = check_quality(raw, 130, min_peaks=5)
    assert not q.ok and "MOVEMENT_ARTIFACT" in quality_codes(q, raw, 5)


def test_user_id_cannot_escape_the_templates_folder(client):
    for bad in ("../x", "a/b", "a b", "", "x" * 41):
        assert client.post("/enroll/start", json={"user_id": bad}).status_code == 422
        assert client.post("/monitor/start", json={"user_id": bad}).status_code == 422
    assert client.get("/users/..%2Fx").status_code in (404, 422)


def test_exam_routes_keep_answer_key_in_backend(client):
    response = client.get("/exam/questions?level=mixto&difficulty=todas&count=10")
    assert response.status_code == 200
    questions = response.json()["questions"]
    assert len(questions) == 10
    assert all("answer" not in question for question in questions)

    grade = client.post("/exam/grade", json={"answers": [{"question_id": "s01", "answer": "B"}]})
    assert grade.status_code == 200
    assert grade.json()["percentage"] == 100


def test_exam_questions_api_serves_30_per_level_and_difficulty(client):
    for level in ("secundaria", "preparatoria", "universidad"):
        for difficulty in ("facil", "intermedia", "avanzada"):
            response = client.get(
                "/exam/questions",
                params={"level": level, "difficulty": difficulty, "count": 30},
            )
            assert response.status_code == 200
            questions = response.json()["questions"]
            assert len(questions) == 30
            assert all(question["level"] == level for question in questions)
            assert all(question["difficulty"] == difficulty for question in questions)
            assert all("answer" not in question for question in questions)

    assert client.get("/exam/questions?level=secundaria&difficulty=facil&count=12").status_code == 422


@pytest.mark.parametrize(
    ("outcome", "folder", "wearer_label", "ground_truth"),
    [
        ("completed", "examenes_completados", "original", "genuine"),
        ("interrupted", "examenes_interrumpidos", "impostor", "impostor"),
        ("completed", "examenes_completados", "unknown", "unknown"),
    ],
)
def test_exam_metrics_are_saved_as_csv_by_outcome_and_wearer_label(
    client, tmp_path, outcome, folder, wearer_label, ground_truth
):
    reading = {
        "time_seconds": 12.5,
        "question_number": 2,
        "question_id": "s02",
        "question_text": "¿Cuál es la idea principal?",
        "wearer_label": wearer_label,
        "similarity": 0.42,
        "smoothed_similarity": 0.38,
        "threshold": 0.31,
        "status": "AUTHENTICATED",
        "heart_rate_bpm": 71.2,
    }

    response = client.post(
        "/exam/metrics",
        json={"outcome": outcome, "readings": [reading]},
    )

    assert response.status_code == 201
    saved = response.json()
    assert saved["saved"] is True and saved["outcome"] == outcome and saved["rows"] == 1
    csv_path = tmp_path / "metrics" / folder / saved["file"].split("/", 1)[1]
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["session_id"] == saved["session_id"]
    assert rows[0]["outcome"] == outcome
    assert rows[0]["time_seconds"] == "12.5"
    assert rows[0]["question_number"] == "2"
    assert rows[0]["question_id"] == "s02"
    assert rows[0]["question_text"] == reading["question_text"]
    assert rows[0]["wearer_label"] == wearer_label
    assert rows[0]["ground_truth"] == ground_truth
    assert rows[0]["similarity"] == "0.42"
    assert rows[0]["smoothed_similarity"] == "0.38"
    assert rows[0]["threshold"] == "0.31"


def test_exam_metrics_without_readings_still_record_interrupted_session(client, tmp_path):
    response = client.post(
        "/exam/metrics",
        json={"outcome": "interrupted", "readings": []},
    )

    assert response.status_code == 201
    saved = response.json()
    assert saved["rows"] == 1
    csv_path = tmp_path / "metrics" / "examenes_interrumpidos" / saved["file"].split("/", 1)[1]
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["status"] == "NO_READINGS"
    assert row["outcome"] == "interrupted"
    assert row["ground_truth"] == "unknown"


def test_exam_metrics_reject_invalid_wearer_label(client):
    response = client.post(
        "/exam/metrics",
        json={
            "outcome": "completed",
            "readings": [{"time_seconds": 1, "status": "AUTHENTICATED", "wearer_label": "maybe"}],
        },
    )

    assert response.status_code == 422


def test_recheck_uses_the_existing_template_and_returns_a_result(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/enroll/start", json={"user_id": "ana"}).status_code == 202
        result = collect_until(ws, "enroll_result")[-1]
        assert result["ok"] is True
        wait_idle(client)

        assert client.post("/monitor/recheck", json={"user_id": "ana"}).status_code == 202
        events = collect_until(ws, "recheck_result")

    result = events[-1]
    assert result["kind"] == "verified"
    assert result["ok"] is True
    assert result["threshold"] == 0.31
    wait_idle(client)
