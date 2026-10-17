import time

import pytest
from fastapi.testclient import TestClient

from src.api import ScenarioSource, Controller, create_app, quality_codes
from src.checkpoint import save_checkpoint
from src.config import Config
from src.evaluate import make_model_embed_fn
from src.model import ECGEmbeddingNet
from src.quality import check_quality


@pytest.fixture
def client(tmp_path):
    cfg = Config()
    net = ECGEmbeddingNet(cfg).eval()
    ctl = Controller(make_model_embed_fn(net), cfg, "modelo_test.pt", threshold=0.31, simulate=True, speed=300.0,
                     template_dir=tmp_path / "live")
    with TestClient(create_app(ctl)) as c:
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
    response = client.get("/exam/questions?level=mixto&difficulty=todas&count=12")
    assert response.status_code == 200
    questions = response.json()["questions"]
    assert len(questions) == 12
    assert all("answer" not in question for question in questions)

    grade = client.post("/exam/grade", json={"answers": [{"question_id": "s01", "answer": "B"}]})
    assert grade.status_code == 200
    assert grade.json()["percentage"] == 100


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
