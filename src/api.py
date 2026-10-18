"""Backend HTTP + WebSocket para la web: registro de huella y autenticación continua.

Arranque (desde la raíz del repo):
    python -m src.api --simulate        # sin banda: señal sintética que se puede manipular (/sim/scenario)
    python -m src.api                   # con el Polar H10 por Bluetooth (en la misma máquina que el backend)
El contrato completo (rutas, JSON, eventos) está en docs/integracion_web.md.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.checkpoint import load_checkpoint
from src.config import Config
from src.exam import ExamBank
from src.errors import CheckpointError
from src.evaluate import make_model_embed_fn
from src.live import AUTH, BAD_SIGNAL, REJECT, make_template, monitor, score_window
from src.quality import HR_RANGE, MAX_RR_GAP_S, MAX_RR_VARIATION, MAX_SPIKE_RATIO, check_quality
from src.record import POLAR_FS
from src.session import ConnectionLost, PolarSource, wait_for_ecg
from src.synthetic import make_synthetic_signal
from src.try_me import _default_checkpoint
from src.windowing import signal_to_windows

DEFAULT_THRESHOLD = 0.31      # similitud coseno; provisional (se recalibra con el modelo final)
DEFAULT_METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"
SETTLE_S, ENROLL_S = 5.0, 60.0
RECHECK_SETTLE_S, RECHECK_RECORD_S = 3.0, 10.0
USER_ID = re.compile(r"[A-Za-z0-9-]{1,40}")

STATUS = {AUTH: "AUTHENTICATED", REJECT: "REJECTED", BAD_SIGNAL: "BAD_SIGNAL"}

# Códigos de problema de señal -> mensaje en español listo para mostrar al usuario.
SIGNAL_MESSAGES = {
    "FLAT_SIGNAL": "No llega señal: revisa que la banda esté puesta y con los electrodos húmedos.",
    "INVALID_VALUES": "La señal trae valores no válidos: ajusta la banda y repite.",
    "MOVEMENT_ARTIFACT": "Hay movimiento o mal contacto: ponte derecho/a y no te muevas.",
    "FEW_BEATS": "Casi no se detectan latidos: ajusta la banda y humedece los electrodos.",
    "IMPOSSIBLE_HEART_RATE": "Se detecta un ritmo cardiaco imposible: probablemente mal contacto.",
    "IRREGULAR_RHYTHM": "El ritmo es muy irregular: quédate quieto/a y respira normal.",
    "SIGNAL_GAP": "Se perdió la señal unos segundos: la banda se despegó o se movió.",
    "POOR_SIGNAL": "La señal no es buena: ajusta la banda y quédate quieto/a.",
}


def quality_codes(q, raw, min_peaks: int, fs: int = POLAR_FS) -> list:
    """Traduce un QualityReport a códigos estables que el frontend puede usar en un `switch`."""
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw).all():
        return ["INVALID_VALUES"]
    if raw.std() < 1e-6:
        return ["FLAT_SIGNAL"]
    if q.ok:
        return []
    codes = []
    if q.spike_ratio > MAX_SPIKE_RATIO:
        codes.append("MOVEMENT_ARTIFACT")
    if q.n_peaks < min_peaks:
        codes.append("FEW_BEATS")
    else:
        if not HR_RANGE[0] <= q.hr_bpm <= HR_RANGE[1]:
            codes.append("IMPOSSIBLE_HEART_RATE")
        if q.rr_variation > MAX_RR_VARIATION:
            codes.append("IRREGULAR_RHYTHM")
        if len(q.peak_indices) > 1 and np.diff(q.peak_indices).max() / fs > MAX_RR_GAP_S:
            codes.append("SIGNAL_GAP")
    return codes or ["POOR_SIGNAL"]


def _loopback_origin_aliases(origins) -> list[str]:
    """Allow localhost and 127.0.0.1 as equivalent local frontend origins."""
    allowed = set(origins)
    for origin in origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1"}:
            continue
        alias_host = "localhost" if parsed.hostname == "127.0.0.1" else "127.0.0.1"
        port = f":{parsed.port}" if parsed.port is not None else ""
        allowed.add(f"{parsed.scheme}://{alias_host}{port}")
    return sorted(allowed)


# ---------------------------------------------------------------- fuente simulada
class ScenarioSource:
    """Señal sintética con la misma interfaz que PolarSource, pero se puede cambiar lo que "pasa".

    same_person: el mismo "corazón" que en el registro.  other_person: otro latido (invertido), que el modelo
    rechaza.  bad_signal: banda floja (señal plana).  movement: picos de movimiento.
    """

    SCENARIOS = ("same_person", "other_person", "bad_signal", "movement")

    def __init__(self, clock, fs: int = POLAR_FS, seed: int = 1, scenario: str = "same_person"):
        self.clock, self.fs, self._t0 = clock, fs, None
        self._me = make_synthetic_signal(1800, fs, seed=seed, hr_bpm=70) * 400 + 20
        rng = np.random.default_rng(seed)
        noise = np.zeros(len(self._me))
        hits = rng.choice(len(self._me), size=len(self._me) // fs, replace=False)   # ~1 pico por segundo
        noise[hits] = rng.choice([-1, 1], size=len(hits)) * rng.uniform(2000, 4000, size=len(hits))
        self._noise = noise
        self._segments = [(0, scenario)]

    async def start(self) -> None:
        self._t0 = self.clock()

    async def stop(self) -> None:
        pass

    def n_samples(self) -> int:
        return 0 if self._t0 is None else int((self.clock() - self._t0) * self.fs)

    def set_scenario(self, name: str) -> None:
        self._segments.append((self.n_samples(), name))

    def take(self, i0: int, i1: int) -> np.ndarray:
        idx = np.arange(i0, i1)
        base = self._me[idx % len(self._me)]
        out = base.copy()
        for k, (start, name) in enumerate(self._segments):
            end = self._segments[k + 1][0] if k + 1 < len(self._segments) else np.inf
            m = (idx >= start) & (idx < end)
            if name == "other_person":
                out[m] = 40 - base[m]
            elif name == "bad_signal":
                out[m] = 20.0
            elif name == "movement":
                out[m] = base[m] + self._noise[idx[m] % len(self._noise)]
        return out


# ---------------------------------------------------------------- control
class Controller:
    """Una sola sesión a la vez (hay una sola banda). Guarda el estado y reparte eventos a los WebSocket."""

    def __init__(self, embed_fn, cfg: Config, ckpt_name: str, threshold: float = DEFAULT_THRESHOLD,
                 simulate: bool = False, speed: float = 1.0, template_dir="data/live", smooth: int = 6,
                 hop_s: float = 2.5):
        if speed != 1.0 and not simulate:
            raise ValueError("--speed solo se puede usar con --simulate")
        self.embed, self.cfg, self.ckpt_name, self.threshold = embed_fn, cfg, ckpt_name, threshold
        self.simulate, self.speed, self.template_dir = simulate, speed, Path(template_dir)
        self.smooth, self.hop_s = smooth, hop_s
        self.source, self.task = None, None
        self.scenario = "same_person"
        self.failures: dict = {}                                   # user_id -> registros fallidos seguidos
        self.queues: set = set()
        self.state = {
            "phase": "idle", "connected": False, "user_id": None,
            "enroll": {"stage": None, "elapsed_s": 0.0, "total_s": ENROLL_S, "signal_ok": None, "codes": [],
                       "consecutive_failures": 0, "last_failure": None},
            "monitor": None, "last_error": None,
        }

    # ---- tiempo (con --speed N la simulación corre N veces más rápido; con el Polar siempre 1)
    def clock(self) -> float:
        return time.monotonic() * self.speed

    async def sleep(self, dt: float) -> None:
        await asyncio.sleep(dt / self.speed)

    # ---- eventos
    def emit(self, event: dict) -> None:
        for q in list(self.queues):
            q.put_nowait(event)

    def snapshot(self) -> dict:
        return {"type": "state", **self.state}

    def set_state(self, **kw) -> None:
        self.state.update(kw)
        self.emit(self.snapshot())

    def fail(self, code: str, message: str) -> None:
        self.state["last_error"] = {"code": code, "message": message}
        self.emit({"type": "error", "code": code, "message": message})

    # ---- usuarios
    def template_path(self, user_id: str) -> Path:
        return self.template_dir / f"{user_id}.npz"

    def load_template(self, user_id: str) -> np.ndarray:
        p = self.template_path(user_id)
        if not p.exists():
            raise HTTPException(404, {"code": "NOT_ENROLLED", "message": f"{user_id} no tiene huella registrada."})
        with np.load(p, allow_pickle=False) as z:
            if str(z["checkpoint"]) != self.ckpt_name:
                raise HTTPException(409, {"code": "TEMPLATE_MODEL_MISMATCH", "message":
                                          f"La huella es de {z['checkpoint']} y el servidor usa {self.ckpt_name}. "
                                          "Hay que registrar de nuevo."})
            return z["template"]

    # ---- fuente de señal
    async def open_source(self) -> None:
        if self.source is not None:
            return
        self.set_state(phase="connecting")
        src = ScenarioSource(self.clock, scenario=self.scenario) if self.simulate else PolarSource()
        try:
            await src.start()
            if not self.simulate:
                await wait_for_ecg(src)
        except ConnectionLost as e:
            await src.stop()
            self.fail("CONNECTION_LOST", str(e))
            raise ConnectionLost(str(e)) from e
        except RuntimeError as e:
            await src.stop()
            self.fail("NO_DEVICE", str(e))
            raise ConnectionLost(str(e)) from e
        except Exception as e:                    # bleak/Bluetooth: apagado, sin permisos, etc.
            await src.stop()
            self.fail("CONNECTION_FAILED", f"No se pudo conectar con la banda: {e}")
            raise ConnectionLost(str(e)) from e
        self.source = src
        self.state["connected"] = True

    async def close_source(self) -> None:
        if self.source is not None:
            await self.source.stop()
        self.source = None
        self.state["connected"] = False

    def set_scenario(self, name: str) -> None:
        self.scenario = name
        if self.source is not None:
            self.source.set_scenario(name)

    # ---- operaciones
    def start(self, coro_fn, *args) -> None:
        if self.task is not None and not self.task.done():
            raise HTTPException(409, {"code": "BUSY", "message": f"Ya hay una operación en curso ({self.state['phase']})."})
        self.task = asyncio.get_running_loop().create_task(coro_fn(*args))

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        await self.close_source()
        self.state["monitor"] = None
        self.set_state(phase="idle", user_id=None)

    async def _countdown(self, stage: str, total: float, quality: bool, stall_s: float = 5.0) -> None:
        src, e = self.source, self.state["enroll"]
        elapsed, last_n, stalled = 0.0, src.n_samples(), 0.0
        while elapsed < total - 1e-9:
            await self.sleep(1.0)
            elapsed += 1.0
            n = src.n_samples()
            stalled = stalled + 1.0 if n == last_n else 0.0
            last_n = n
            if stalled >= stall_s:
                raise ConnectionLost("Se perdió la conexión con la banda (dejó de llegar señal).")
            if quality and int(elapsed) % 2 == 0 and n >= 10 * POLAR_FS:
                raw = src.take(n - 10 * POLAR_FS, n)
                codes = quality_codes(q := check_quality(raw, POLAR_FS, min_peaks=5), raw, 5)
                e["signal_ok"], e["codes"] = q.ok, codes
            e.update(stage=stage, elapsed_s=min(elapsed, total), total_s=total)
            self.emit({"type": "enroll_progress", "stage": stage, "elapsed_s": e["elapsed_s"], "total_s": total,
                       "signal_ok": e["signal_ok"], "codes": e["codes"]})

    async def run_enroll(self, user_id: str) -> None:
        e = self.state["enroll"]
        e.update(stage=None, elapsed_s=0.0, signal_ok=None, codes=[], last_failure=None,
                 consecutive_failures=self.failures.get(user_id, 0))
        self.state["last_error"] = None
        self.set_state(user_id=user_id)
        try:
            await self.open_source()
            self.set_state(phase="enrolling")
            await self._countdown("settling", SETTLE_S, quality=False)
            i0 = self.source.n_samples()
            await self._countdown("recording", ENROLL_S, quality=True)
            raw = self.source.take(i0, self.source.n_samples())
            q = check_quality(raw, POLAR_FS)
            if not q.ok:
                codes = quality_codes(q, raw, 8)
                self.failures[user_id] = self.failures.get(user_id, 0) + 1
                fail = {"codes": codes, "messages": [SIGNAL_MESSAGES[c] for c in codes]}
                e.update(consecutive_failures=self.failures[user_id], last_failure=fail, stage=None)
                self.emit({"type": "enroll_result", "ok": False, "user_id": user_id,
                           "consecutive_failures": self.failures[user_id], **fail})
                return
            template = make_template(self.embed, signal_to_windows(raw, POLAR_FS, self.cfg))
            self.template_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.template_path(user_id), template=template, checkpoint=self.ckpt_name)
            self.failures[user_id] = 0
            e.update(consecutive_failures=0, last_failure=None, stage=None)
            self.emit({"type": "enroll_result", "ok": True, "user_id": user_id, "consecutive_failures": 0,
                       "codes": [], "messages": []})
        except ConnectionLost as ex:
            e.update(stage=None)
            if self.state["last_error"] is None:
                self.fail("CONNECTION_LOST", str(ex))
            await self.close_source()
        finally:
            self.set_state(phase="idle")

    async def run_monitor(self, user_id: str, template: np.ndarray) -> None:
        self.state["last_error"] = None
        self.set_state(user_id=user_id)
        streaks = {"bad": 0, "rej": 0}
        try:
            await self.open_source()
            self.state["monitor"] = {"t": 0.0, "status": "WARMING_UP", "score": None, "smoothed": None,
                                     "threshold": self.threshold, "bad_signal_streak": 0, "rejection_streak": 0,
                                     "hr_bpm": None, "codes": []}
            self.set_state(phase="monitoring")
            src = self.source

            def on_step(step) -> None:
                n = src.n_samples()
                raw = src.take(max(0, n - 10 * POLAR_FS), n)
                q = check_quality(raw, POLAR_FS, min_peaks=5)
                streaks["bad"] = streaks["bad"] + 1 if step.status == BAD_SIGNAL else 0
                streaks["rej"] = streaks["rej"] + 1 if step.status == REJECT else 0
                ev = {"t": round(step.t, 2), "status": STATUS[step.status],
                      "authenticated": None if step.status == BAD_SIGNAL else step.status == AUTH,
                      "score": step.score, "smoothed": step.smoothed, "threshold": self.threshold,
                      "bad_signal_streak": streaks["bad"], "rejection_streak": streaks["rej"],
                      "hr_bpm": None if np.isnan(q.hr_bpm) else round(float(q.hr_bpm), 1),
                      "codes": quality_codes(q, raw, 5) if step.status == BAD_SIGNAL else []}
                self.state["monitor"] = ev
                self.emit({"type": "decision", **ev})

            await monitor(src, template, self.embed, self.cfg, self.threshold, self.sleep, self.clock,
                          lambda *a, **k: None, on_step, hop_s=self.hop_s, smooth=self.smooth)
        except ConnectionLost as ex:
            if self.state["last_error"] is None:
                self.fail("CONNECTION_LOST", str(ex))
            await self.close_source()
        finally:
            self.set_state(phase="idle")

    async def run_recheck(self, user_id: str, template: np.ndarray) -> None:
        self.state["last_error"] = None
        self.set_state(phase="rechecking", user_id=user_id)
        try:
            await self.open_source()
            self.set_state(phase="rechecking")
            src = self.source
            for elapsed in range(1, int(RECHECK_SETTLE_S + RECHECK_RECORD_S) + 1):
                await self.sleep(1.0)
                stage = "settling" if elapsed <= RECHECK_SETTLE_S else "recording"
                stage_elapsed = elapsed if stage == "settling" else elapsed - RECHECK_SETTLE_S
                total = RECHECK_SETTLE_S if stage == "settling" else RECHECK_RECORD_S
                self.emit({"type": "recheck_progress", "stage": stage, "elapsed_s": stage_elapsed,
                           "total_s": total})
            raw = src.take(src.n_samples() - int(RECHECK_RECORD_S * POLAR_FS), src.n_samples())
            quality = check_quality(raw, POLAR_FS, min_peaks=8)
            codes = quality_codes(quality, raw, 8)
            if not quality.ok:
                self.emit({"type": "recheck_result", "kind": "signal", "ok": False, "score": None,
                           "codes": codes, "messages": [SIGNAL_MESSAGES[code] for code in codes]})
                return
            windows = signal_to_windows(raw, POLAR_FS, self.cfg)
            if not len(windows):
                codes = ["POOR_SIGNAL"]
                self.emit({"type": "recheck_result", "kind": "signal", "ok": False, "score": None,
                           "codes": codes, "messages": [SIGNAL_MESSAGES[code] for code in codes]})
                return
            scores = [score_window(self.embed, template, window) for window in windows]
            score = float(np.mean(scores))
            verified = score >= self.threshold
            self.emit({"type": "recheck_result", "kind": "identity" if not verified else "verified",
                       "ok": verified, "score": score, "threshold": self.threshold, "codes": [],
                       "messages": []})
        except ConnectionLost as exc:
            if self.state["last_error"] is None:
                self.fail("CONNECTION_LOST", str(exc))
            await self.close_source()
        finally:
            self.set_state(phase="idle")


# ---------------------------------------------------------------- API
class UserBody(BaseModel):
    user_id: str = Field(pattern=rf"^{USER_ID.pattern}$")   # anclado: el id forma parte de una ruta de archivo


class ScenarioBody(BaseModel):
    scenario: Literal["same_person", "other_person", "bad_signal", "movement"]

class ExamAnswer(BaseModel):
    question_id: str = Field(min_length=1, max_length=80)
    answer: Literal["A", "B", "C", "D"] | None


class ExamAnswers(BaseModel):
    answers: list[ExamAnswer] = Field(max_length=30)


class ExamMetricReading(BaseModel):
    time_seconds: float = Field(ge=0)
    question_number: int | None = Field(default=None, ge=1, le=30)
    question_id: str = Field(default="", max_length=80)
    question_text: str = Field(default="", max_length=1000)
    wearer_label: Literal["original", "impostor", "unknown"] = "unknown"
    similarity: float | None = Field(default=None, ge=-1, le=1)
    smoothed_similarity: float | None = Field(default=None, ge=-1, le=1)
    threshold: float | None = Field(default=None, ge=-1, le=1)
    status: Literal["AUTHENTICATED", "REJECTED", "BAD_SIGNAL", "NO_READINGS"]
    heart_rate_bpm: float | None = Field(default=None, ge=0, le=300)


class ExamMetrics(BaseModel):
    outcome: Literal["completed", "interrupted"]
    readings: list[ExamMetricReading] = Field(max_length=20000)


def create_app(
    ctl: Controller,
    cors_origins=("*",),
    exam_bank: ExamBank | None = None,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> FastAPI:
    app = FastAPI(title="Autenticación continua con ECG")
    bank = exam_bank or ExamBank()
    metrics_dir = Path(metrics_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_loopback_origin_aliases(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "mode": "simulate" if ctl.simulate else "polar", "model": ctl.ckpt_name,
                "threshold": ctl.threshold}

    @app.get("/state")
    def state():
        return ctl.state

    @app.get("/exam/questions")
    def exam_questions(level: str = "mixto", difficulty: str = "todas", count: int = 20):
        questions = bank.select(level, difficulty, count)
        return {"questions": questions, "count": len(questions)}

    @app.post("/exam/grade")
    def exam_grade(body: ExamAnswers):
        return bank.grade([answer.model_dump() for answer in body.answers])

    @app.post("/exam/metrics", status_code=201)
    def save_exam_metrics(body: ExamMetrics):
        session_id = uuid.uuid4().hex
        saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        directory_name = "examenes_completados" if body.outcome == "completed" else "examenes_interrumpidos"
        output_dir = metrics_dir / directory_name
        filename = f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}.csv"
        output_path = output_dir / filename
        fields = (
            "session_id", "outcome", "saved_at", "time_seconds", "question_number",
            "wearer_label", "ground_truth",
            "question_id", "question_text", "similarity", "smoothed_similarity",
            "threshold", "status", "heart_rate_bpm",
        )
        readings = body.readings or [
            ExamMetricReading(time_seconds=0, status="NO_READINGS")
        ]
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for reading in readings:
                    writer.writerow({
                        "session_id": session_id,
                        "outcome": body.outcome,
                        "saved_at": saved_at,
                        "wearer_label": reading.wearer_label,
                        "ground_truth": {
                            "original": "genuine",
                            "impostor": "impostor",
                            "unknown": "unknown",
                        }[reading.wearer_label],
                        **{
                            **reading.model_dump(),
                        },
                    })
        except OSError as exc:
            logging.getLogger(__name__).exception("No se pudieron guardar las métricas del examen.")
            raise HTTPException(500, "No se pudieron guardar las métricas del examen.") from exc
        return {
            "saved": True,
            "session_id": session_id,
            "outcome": body.outcome,
            "rows": len(readings),
            "file": f"{directory_name}/{filename}",
        }

    @app.get("/users/{user_id}")
    def user(user_id: str):
        if not USER_ID.fullmatch(user_id):
            raise HTTPException(422, {"code": "INVALID_USER_ID", "message": "Solo letras, números y guiones."})
        return {"user_id": user_id, "enrolled": ctl.template_path(user_id).exists()}

    @app.delete("/users/{user_id}")
    def delete_user(user_id: str):
        if not USER_ID.fullmatch(user_id):
            raise HTTPException(422, {"code": "INVALID_USER_ID", "message": "Solo letras, números y guiones."})
        p = ctl.template_path(user_id)
        existed = p.exists()
        p.unlink(missing_ok=True)
        ctl.failures.pop(user_id, None)
        return {"user_id": user_id, "deleted": existed}

    @app.post("/enroll/start", status_code=202)
    async def enroll_start(body: UserBody):
        ctl.start(ctl.run_enroll, body.user_id)
        return {"accepted": True, "user_id": body.user_id}

    @app.post("/monitor/start", status_code=202)
    async def monitor_start(body: UserBody):
        if ctl.task is not None and not ctl.task.done():
            raise HTTPException(409, {"code": "BUSY", "message": f"Ya hay una operación en curso ({ctl.state['phase']})."})
        template = ctl.load_template(body.user_id)
        ctl.start(ctl.run_monitor, body.user_id, template)
        return {"accepted": True, "user_id": body.user_id}

    @app.post("/monitor/recheck", status_code=202)
    async def monitor_recheck(body: UserBody):
        if ctl.task is not None and not ctl.task.done():
            raise HTTPException(409, {"code": "BUSY", "message": f"Ya hay una operación en curso ({ctl.state['phase']})."})
        template = ctl.load_template(body.user_id)
        ctl.start(ctl.run_recheck, body.user_id, template)
        return {"accepted": True, "user_id": body.user_id,
                "settle_s": RECHECK_SETTLE_S, "record_s": RECHECK_RECORD_S}

    @app.post("/stop")
    async def stop():
        await ctl.stop()
        return {"phase": "idle"}

    if ctl.simulate:
        @app.post("/sim/scenario")
        async def sim_scenario(body: ScenarioBody):
            ctl.set_scenario(body.scenario)
            return {"scenario": body.scenario}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        ctl.queues.add(q)
        try:
            await sock.send_json(ctl.snapshot())
            while True:
                await sock.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            ctl.queues.discard(q)

    return app


def main(argv=None) -> int:
    import uvicorn

    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.api", description="Backend de autenticación continua")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default=None, help="por defecto models/modelo_ecg.pt o el último modelo_vN.pt")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--simulate", action="store_true", help="señal sintética, sin banda")
    p.add_argument("--speed", type=float, default=1.0, help="solo con --simulate: acelera el tiempo N veces")
    p.add_argument("--cors-origin", action="append", default=None,
                   help="origen permitido para el navegador (repetible); por defecto cualquiera")
    a = p.parse_args(argv)
    try:
        ckpt = Path(a.checkpoint) if a.checkpoint else _default_checkpoint(cfg.models_dir)
        embed = make_model_embed_fn(load_checkpoint(ckpt, cfg))
        ctl = Controller(embed, cfg, ckpt.name, a.threshold, a.simulate, a.speed)
    except (CheckpointError, ValueError) as e:
        print("ERROR:", e)
        return 1
    print(f"Modelo: {ckpt.name} | umbral: {a.threshold} | modo: {'simulado' if a.simulate else 'Polar H10'}")
    uvicorn.run(create_app(ctl, a.cors_origin or ["*"]), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
