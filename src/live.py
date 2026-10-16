"""Autenticación continua en vivo, sin web: registra tu huella y luego decide cada pocos segundos.

Uso (con la banda puesta y los electrodos húmedos):
    python -m src.live --id kitzia                 # registra 60 s y después vigila hasta Ctrl+C
    python -m src.live --id kitzia --skip-enroll   # reutiliza la huella guardada
Sin sensor:  añade --simulate;   con grabaciones viejas:  --replay a.npy b.npy
"""
from __future__ import annotations

import argparse
import asyncio
import re
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.checkpoint import load_checkpoint
from src.config import Config
from src.data import load_window_dir
from src.errors import CheckpointError, DataError, EvaluationError
from src.evaluate import eer, l2_normalize, make_model_embed_fn, run_protocol, split_enroll_test
from src.quality import check_quality
from src.record import POLAR_FS
from src.session import ConnectionLost, PolarSource, SimClock, SimulatedSource
from src.synthetic import make_synthetic_signal
from src.try_me import _default_checkpoint
from src.windowing import signal_to_windows

AUTH, REJECT, BAD_SIGNAL = "AUTENTICADO", "NO AUTENTICADO", "SEÑAL MALA"


@dataclass
class Step:
    t: float                  # segundos desde que empezó la vigilancia
    status: str               # AUTH, REJECT o BAD_SIGNAL
    score: float | None       # similitud coseno de ESTA ventana con tu huella
    smoothed: float | None    # promedio de las últimas ventanas (es lo que decide)


def make_template(embed_fn, windows: np.ndarray) -> np.ndarray:
    """Huella = promedio de los embeddings de las ventanas de registro, normalizado (como en evaluate)."""
    if len(windows) == 0:
        raise DataError("No hay ventanas válidas para crear la huella")
    return l2_normalize(embed_fn(windows).mean(axis=0, keepdims=True))[0]


def score_window(embed_fn, template: np.ndarray, window: np.ndarray) -> float:
    return float(l2_normalize(embed_fn(window[None]))[0] @ template)


class Decider:
    """Promedia las últimas `smooth` ventanas: una sola ventana mala no te bloquea, pero un impostor sí."""

    def __init__(self, threshold: float, smooth: int = 6):
        self.threshold, self.scores = threshold, deque(maxlen=smooth)

    def reset(self) -> None:
        self.scores.clear()

    def update(self, score: float) -> tuple[float, str]:
        self.scores.append(score)
        mean = float(np.mean(self.scores))
        return mean, AUTH if mean >= self.threshold else REJECT


def calibrate_threshold(embed_fn, polar_dir, cfg: Config) -> float | None:
    """Umbral = punto de EER entre sesiones con las grabaciones Polar que ya tienes.

    OJO: son personas con las que el modelo se entrenó; para alguien nuevo (tú) puede quedar algo
    alto o bajo. Si ves que te rechaza siendo tú o acepta a otros, ajusta con --threshold.
    """
    d = Path(polar_dir)
    if not d.exists() or not list(d.glob("*.npz")):
        return None
    try:
        table = load_window_dir(d, cfg.window_samples)
        keep = [p for p in table.persons() if len(table.sessions_of(p)) >= 2]
        if len(keep) < 2:
            return None
        table = table.subset(np.isin(table.participant_ids, keep))
        enroll, test = split_enroll_test(table, "cross_session")
        t = run_protocol(embed_fn, enroll, test)
        return eer(t.genuine, t.impostor)[1]
    except (DataError, EvaluationError):
        return None


async def _wait(source, sleep, seconds: float, say, label: str, stall_s: float = 5.0) -> None:
    remaining, last_n, stalled = seconds, source.n_samples(), 0.0
    while remaining > 1e-9:
        step = min(1.0, remaining)
        await sleep(step)
        remaining -= step
        n = source.n_samples()
        stalled = stalled + step if n == last_n else 0.0
        last_n = n
        if stalled >= stall_s:
            raise ConnectionLost("Se perdió la conexión con la banda (dejó de llegar señal).")
        say(f"\r  {label}: {remaining:3.0f} s ", end="")
    say("\r" + " " * 40 + "\r", end="")


async def enroll(source, embed_fn, cfg: Config, sleep, say, seconds: float = 60.0, settle_s: float = 5.0):
    """Registro: junta `seconds` de señal buena y devuelve la huella."""
    say("Registro: quédate quieto/a y respira normal.")
    await _wait(source, sleep, settle_s, say, "estabilizando")
    i0 = source.n_samples()
    await _wait(source, sleep, seconds, say, "registrando")
    raw = source.take(i0, source.n_samples())
    q = check_quality(raw, POLAR_FS)
    if not q.ok:
        raise DataError("El registro salió con mala señal: " + " ".join(q.issues) + " Repite el registro.")
    return make_template(embed_fn, signal_to_windows(raw, POLAR_FS, cfg))


async def monitor(source, template, embed_fn, cfg: Config, threshold: float, sleep, clock, say, on_step,
                  hop_s: float = 2.5, smooth: int = 6, duration: float | None = None,
                  max_stall_s: float = 15.0, reset_after: int = 2) -> list:
    """Cada `hop_s` toma los últimos 5 s de señal (con 10 s extra de contexto para el filtro) y decide."""
    win, ctx = cfg.window_samples, 3 * cfg.window_samples      # 5 s de ventana, 15 s de buffer
    decider, steps, bad_run = Decider(threshold, smooth), [], 0
    t0, last_n, stalled = clock(), source.n_samples(), 0.0
    while duration is None or clock() - t0 < duration:
        await sleep(hop_s)
        n = source.n_samples()
        stalled = stalled + hop_s if n == last_n else 0.0
        last_n = n
        if stalled >= max_stall_s:
            raise ConnectionLost("Se perdió la conexión con la banda (dejó de llegar señal).")
        if n < ctx:
            continue                                            # aún no hay suficiente señal
        raw = source.take(n - ctx, n)
        windows = signal_to_windows(raw, POLAR_FS, cfg) if stalled == 0 else []
        if len(windows) == 0 or not check_quality(raw[-2 * win:], POLAR_FS, min_peaks=5).ok:
            step = Step(clock() - t0, BAD_SIGNAL, None, None)   # no se decide con señal mala: ni acepta ni rechaza
            bad_run += 1
            if bad_run >= reset_after:
                # Banda quitada o floja: los puntajes de antes ya no valen. Si alguien más se la pone,
                # no debe heredar tu promedio.
                decider.reset()
        else:
            bad_run = 0
            score = score_window(embed_fn, template, windows[-1])
            smoothed, status = decider.update(score)
            step = Step(clock() - t0, status, score, smoothed)
        steps.append(step)
        on_step(step)
    return steps


def format_step(s: Step, threshold: float) -> str:
    mm, ss = divmod(int(s.t), 60)
    if s.score is None:
        return f"[{mm:02d}:{ss:02d}] sin decisión: la señal no es buena (¿banda floja o te moviste?)"
    return (f"[{mm:02d}:{ss:02d}] similitud {s.score:.2f} | promedio {s.smoothed:.2f} "
            f"(umbral {threshold:.2f})  ->  {s.status}")


def summary(steps: list) -> str:
    decided = [s for s in steps if s.score is not None]
    if not decided:
        return "No hubo ninguna decisión (señal mala todo el tiempo)."
    ok = sum(s.status == AUTH for s in decided)
    return (f"Resumen: {len(decided)} decisiones, {ok} AUTENTICADO ({ok / len(decided) * 100:.0f}%), "
            f"{len(decided) - ok} NO AUTENTICADO, {len(steps) - len(decided)} con señal mala.")


def _template_path(pid: str) -> Path:
    return Path("data/live") / f"{pid}.npz"


def _build_source(a, cfg: Config):
    if a.replay:
        sig = np.concatenate([np.load(f).astype(np.float64) for f in a.replay])
        clock = SimClock()
        return SimulatedSource(clock, signal=sig), clock, clock.sleep, clock
    if a.simulate:
        seed = zlib.crc32(a.id.encode()) % 1000
        me = make_synthetic_signal(a.enroll_seconds + 5 + a.sim_seconds, POLAR_FS, seed=seed) * 400
        other = make_synthetic_signal(a.sim_seconds, POLAR_FS, seed=seed + 1) * 400
        clock = SimClock()
        return SimulatedSource(clock, signal=np.concatenate([me, other])), clock, clock.sleep, clock
    return PolarSource(), time.monotonic, asyncio.sleep, time.monotonic


async def run(a, cfg: Config, embed_fn, threshold: float) -> int:
    source, _, sleep, clock = _build_source(a, cfg)
    say = lambda *x, **k: print(*x, flush=True, **k)
    await source.start()
    try:
        if a.skip_enroll:
            side = np.load(_template_path(a.id), allow_pickle=False)
            if str(side["checkpoint"]) != a.ckpt_name:
                raise DataError(f"La huella guardada es de {side['checkpoint']} y ahora usas {a.ckpt_name}. "
                                "Registra de nuevo.")
            template = side["template"]
            say(f"Huella cargada de {_template_path(a.id)}")
        else:
            template = await enroll(source, embed_fn, cfg, sleep, say, a.enroll_seconds)
            p = _template_path(a.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, template=template, checkpoint=a.ckpt_name)
            say(f"Huella guardada en {p}")
        say(f"\nVigilando (umbral {threshold:.2f}). Ctrl+C para terminar.\n")
        steps: list = []

        def on_step(s: Step) -> None:
            steps.append(s)
            say(format_step(s, threshold))

        offline = a.simulate or bool(a.replay)       # en simulación, quedarse sin señal = se acabó el archivo
        try:
            await monitor(source, template, embed_fn, cfg, threshold, sleep, clock, say, on_step,
                          a.hop, a.smooth, a.sim_seconds * 2 if a.simulate else None)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except ConnectionLost as e:
            if not offline:
                say("\nERROR:", e)
        say("\n" + summary(steps))
        return 0
    finally:
        await source.stop()


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.live", description="Autenticación continua en vivo")
    p.add_argument("--id", required=True)
    p.add_argument("--checkpoint", default=None, help="por defecto modelo_ecg.pt o el último modelo_vN.pt")
    p.add_argument("--polar-dir", default=f"{cfg.data_dir}/polar")
    p.add_argument("--threshold", type=float, default=None, help="por defecto se calcula con las grabaciones Polar")
    p.add_argument("--enroll-seconds", type=float, default=60.0)
    p.add_argument("--hop", type=float, default=2.5, help="cada cuántos segundos se decide")
    p.add_argument("--smooth", type=int, default=6, help="cuántas ventanas se promedian para decidir")
    p.add_argument("--skip-enroll", action="store_true")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--sim-seconds", type=float, default=60.0)
    p.add_argument("--replay", nargs="+", default=None, help="archivos .npy crudos para simular en vivo")
    a = p.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9-]+", a.id):
            raise ValueError("--id solo puede llevar letras, números y guiones")
        ckpt = Path(a.checkpoint) if a.checkpoint else _default_checkpoint(cfg.models_dir)
        a.ckpt_name = ckpt.name
        embed_fn = make_model_embed_fn(load_checkpoint(ckpt, cfg))
        thr = a.threshold if a.threshold is not None else calibrate_threshold(embed_fn, a.polar_dir, cfg)
        if thr is None:
            raise DataError("No pude calcular el umbral (faltan grabaciones Polar); indica --threshold")
        print(f"Modelo: {ckpt.name}   umbral: {thr:.3f}")
        return asyncio.run(run(a, cfg, embed_fn, thr))
    except (DataError, CheckpointError, ConnectionLost, RuntimeError, ValueError, FileNotFoundError) as e:
        print("ERROR:", e)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
