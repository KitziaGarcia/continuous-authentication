"""Evaluación: EER, ROC, FAR/FRR, protocolos de pares, bootstrap por persona y reporte."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import Config
from src.data import PolarSplit, WindowTable
from src.errors import EvaluationError


def _check(genuine: np.ndarray, impostor: np.ndarray) -> None:
    if len(genuine) == 0 or len(impostor) == 0:
        raise EvaluationError(
            f"Se necesitan puntajes genuinos e impostores (hay {len(genuine)} y {len(impostor)})"
        )


def far_frr_at(genuine: np.ndarray, impostor: np.ndarray, threshold: float):
    """Convención: puntaje >= umbral => se acepta.

    FAR (aceptación falsa): impostores aceptados.  FRR (rechazo falso): genuinos rechazados.
    """
    _check(genuine, impostor)
    return float(np.mean(impostor >= threshold)), float(np.mean(genuine < threshold))


def eer(genuine: np.ndarray, impostor: np.ndarray):
    """Punto donde FAR = FRR. Devuelve (eer, umbral). Es el umbral de operación del sistema."""
    _check(genuine, impostor)
    g, i = np.sort(genuine), np.sort(impostor)
    thr = np.unique(np.concatenate([g, i]))
    far = 1 - np.searchsorted(i, thr, side="left") / len(i)   # impostores >= umbral
    frr = np.searchsorted(g, thr, side="left") / len(g)       # genuinos < umbral
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(thr[k])


def auc_score(genuine: np.ndarray, impostor: np.ndarray) -> float:
    _check(genuine, impostor)
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    return float(roc_auc_score(labels, np.r_[genuine, impostor]))


def roc_points(genuine: np.ndarray, impostor: np.ndarray):
    """Puntos de la curva ROC como (FAR, FRR) para graficar."""
    _check(genuine, impostor)
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    fpr, tpr, _ = roc_curve(labels, np.r_[genuine, impostor])
    return fpr, 1 - tpr


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def make_model_embed_fn(model, batch_size: int = 256):
    """Convierte una red en una función ventanas -> embeddings (sin gradientes, modo eval)."""

    def embed(windows: np.ndarray) -> np.ndarray:
        model.eval()   # BatchNorm usa sus estadísticas guardadas, no las del lote
        out = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                out.append(model(torch.from_numpy(windows[i : i + batch_size]).float()).numpy())
        if not out:
            return np.zeros((0, model.cfg.embedding_dim), dtype=np.float32)
        return np.concatenate(out)

    return embed


def raw_embed_fn(windows: np.ndarray) -> np.ndarray:
    """Baseline sin entrenamiento: la propia señal aplanada como "huella".

    Sirve para demostrar qué aporta realmente el entrenamiento de la red (brief, sección 4).
    """
    return l2_normalize(windows.reshape(len(windows), -1).astype(np.float64))


@dataclass
class Trials:
    genuine: np.ndarray           # puntajes ventana-vs-plantilla de la misma persona
    genuine_owner: np.ndarray     # de quién es la plantilla de cada puntaje genuino
    impostor: np.ndarray
    impostor_owner: np.ndarray    # de quién es la plantilla contra la que se probó al impostor

    def n_owners(self) -> int:
        return len(set(self.genuine_owner.tolist()))


def build_trials(enroll_emb, enroll_ids, test_emb, test_ids) -> Trials:
    """Plantilla = promedio de los embeddings de registro de cada persona (brief, sección 3).

    Cada ventana de prueba se compara contra TODAS las plantillas: contra la propia es un
    intento genuino, contra las demás es un intento impostor.
    """
    enroll_ids, test_ids = np.asarray(enroll_ids), np.asarray(test_ids)
    owners = np.array(sorted(set(enroll_ids.tolist())))
    templates = l2_normalize(np.stack([np.asarray(enroll_emb)[enroll_ids == o].mean(axis=0) for o in owners]))
    scores = l2_normalize(np.asarray(test_emb)) @ templates.T            # coseno
    same = test_ids[:, None] == owners[None, :]
    owner_grid = np.broadcast_to(owners[None, :], scores.shape)
    t = Trials(scores[same], owner_grid[same], scores[~same], owner_grid[~same])
    if len(t.genuine) == 0 or len(t.impostor) == 0:
        raise EvaluationError(
            f"Sin pares suficientes: {len(t.genuine)} genuinos y {len(t.impostor)} impostores"
        )
    return t


def first_session_table(table: WindowTable) -> WindowTable:
    mask = np.zeros(len(table), dtype=bool)
    for p in table.persons():
        mask |= (table.participant_ids == p) & (table.session_ids == table.sessions_of(p)[0])
    return table.subset(mask)


def split_enroll_test(table: WindowTable, mode: str):
    """Protocolos del spec.

    cross_session: registro con la 1.ª sesión, prueba con las siguientes (otro día).
    same_session:  1.ª sesión partida a la mitad en el tiempo (primera mitad registra,
                   segunda prueba). Es más fácil; sirve para ver el techo del sistema.
    """
    if mode not in ("cross_session", "same_session"):
        raise EvaluationError(f"Protocolo desconocido: '{mode}'")
    ids, sess = table.participant_ids, table.session_ids
    enroll = np.zeros(len(table), dtype=bool)
    test = np.zeros(len(table), dtype=bool)
    short = []
    for p in table.persons():
        sessions = table.sessions_of(p)
        first = (ids == p) & (sess == sessions[0])
        if mode == "cross_session":
            if len(sessions) < 2:
                short.append(p)
                continue
            enroll |= first
            test |= (ids == p) & (sess != sessions[0])
        else:
            idx = np.flatnonzero(first)
            if len(idx) < 2:
                short.append(p)
                continue
            half = len(idx) // 2
            enroll[idx[:half]] = True
            test[idx[half:]] = True
    if short:
        raise EvaluationError(
            f"Personas sin datos suficientes para el protocolo '{mode}': {short} "
            "(cross_session necesita 2+ sesiones; same_session necesita 2+ ventanas)"
        )
    return table.subset(enroll), table.subset(test)


def run_protocol(embed_fn, enroll_table: WindowTable, test_table: WindowTable) -> Trials:
    return build_trials(
        embed_fn(enroll_table.windows),
        enroll_table.participant_ids,
        embed_fn(test_table.windows),
        test_table.participant_ids,
    )


def bootstrap_eer_ci(trials: Trials, n_boot: int, seed: int, alpha: float = 0.05):
    """Intervalo de confianza de la EER remuestreando PERSONAS (dueños de plantilla).

    Las ventanas de una misma persona están muy correlacionadas; remuestrearlas una a una
    daría intervalos falsamente estrechos. Con menos de 3 personas no hay forma honesta de
    estimar la variabilidad entre personas: se devuelve (nan, nan) y el reporte dice "n/a".
    """
    owners = sorted(set(trials.genuine_owner.tolist()))
    if len(owners) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    g_by = {o: trials.genuine[trials.genuine_owner == o] for o in owners}
    i_by = {o: trials.impostor[trials.impostor_owner == o] for o in owners}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(owners, size=len(owners), replace=True)
        g = np.concatenate([g_by[o] for o in pick])
        i = np.concatenate([i_by[o] for o in pick])
        if len(g) and len(i):
            vals.append(eer(g, i)[0])
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class ReportRow:
    label: str
    eer: float
    ci_lo: float
    ci_hi: float
    auc: float
    far: float
    frr: float
    n_genuine: int
    n_impostor: int
    n_owners: int


def make_row(label: str, trials: Trials, threshold: float, n_boot: int, seed: int) -> ReportRow:
    """FAR/FRR se miden en el umbral recibido (elegido en validación), no en uno propio."""
    e, _ = eer(trials.genuine, trials.impostor)
    lo, hi = bootstrap_eer_ci(trials, n_boot, seed)
    far, frr = far_frr_at(trials.genuine, trials.impostor, threshold)
    return ReportRow(
        label, e, lo, hi, auc_score(trials.genuine, trials.impostor), far, frr,
        len(trials.genuine), len(trials.impostor), trials.n_owners(),
    )


def format_report(rows: list) -> str:
    lines = [
        "| Experimento | EER | IC 95% | AUC | FAR | FRR | pares gen/imp | personas |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = "n/a (<3 personas)" if np.isnan(r.ci_lo) else f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
        lines.append(
            f"| {r.label} | {r.eer:.3f} | {ci} | {r.auc:.3f} | {r.far:.3f} | {r.frr:.3f} "
            f"| {r.n_genuine}/{r.n_impostor} | {r.n_owners} |"
        )
    return "\n".join(lines)


def compare_models(embed_fns: dict, split: PolarSplit, cfg: Config) -> str:
    """Tabla comparativa (p. ej. solo-Polar vs preentrenado+fine-tuning vs señal cruda).

    Mismo reparto de personas, mismas ventanas y mismo protocolo para todos los modelos.
    El umbral de cada modelo sale de su EER en VALIDACIÓN; el hold-out nunca lo decide.
    """
    enroll_val = first_session_table(split.train)
    hold_enroll, hold_test = split_enroll_test(split.holdout, "cross_session")
    # Misma sesión sobre VALIDACIÓN (ventanas que la red no usó para entrenar), no sobre el entrenamiento.
    same_enroll, same_test = split_enroll_test(split.val, "same_session")
    rows = []
    for name, fn in embed_fns.items():
        val = run_protocol(fn, enroll_val, split.val)
        thr = eer(val.genuine, val.impostor)[1]
        n, s = cfg.bootstrap_samples, cfg.seed
        rows.append(make_row(f"{name} · vistos, entre sesiones (val)", val, thr, n, s))
        rows.append(make_row(f"{name} · NO vistos (hold-out), entre sesiones",
                             run_protocol(fn, hold_enroll, hold_test), thr, n, s))
        rows.append(make_row(f"{name} · vistos, misma sesión (val)",
                             run_protocol(fn, same_enroll, same_test), thr, n, s))
    return format_report(rows)


def main(argv=None) -> int:
    from src.checkpoint import load_checkpoint
    from src.data import load_window_dir, split_polar, table_fingerprint

    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--polar-dir", default=f"{cfg.data_dir}/polar")
    c.add_argument("--polar-only", required=True, help="checkpoint entrenado solo con Polar")
    c.add_argument("--finetuned", required=True, help="checkpoint preentrenado + fine-tuning")
    c.add_argument("--out", default="reports/comparison.md")
    a = p.parse_args(argv)

    split = split_polar(load_window_dir(a.polar_dir, cfg.window_samples), cfg.holdout_ids)
    fingerprint = table_fingerprint(split.train)
    # Si el hold-out (como conjunto) o los datos de entrenamiento cambiaron desde que se entrenó el
    # checkpoint, las filas "vistos"/"NO vistos" ya no significarían lo que dicen.
    for ckpt in (a.polar_only, a.finetuned):
        side = json.loads(Path(ckpt).with_suffix(".json").read_text(encoding="utf-8"))
        if set(side["holdout_ids"]) != set(cfg.holdout_ids):
            print(f"ERROR: {Path(ckpt).name} se entrenó con hold-out {side['holdout_ids']} "
                  f"pero la configuración actual es {list(cfg.holdout_ids)}.")
            return 1
        if side["data_fingerprint"] != fingerprint:
            print(f"ERROR: los datos de entrenamiento actuales ({fingerprint}) no son los de "
                  f"{Path(ckpt).name} ({side['data_fingerprint']}). Vuelve a entrenar el modelo.")
            return 1

    fns = {
        "Solo Polar": make_model_embed_fn(load_checkpoint(a.polar_only, cfg)),
        "Preentrenado + fine-tuning": make_model_embed_fn(load_checkpoint(a.finetuned, cfg)),
        "Señal cruda": raw_embed_fn,
    }
    md = compare_models(fns, split, cfg)
    print(md)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
