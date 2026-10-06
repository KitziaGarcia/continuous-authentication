"""Prueba rápida: ¿el modelo reconoce a cada persona a partir de sus grabaciones con el Polar H10?

Registro = la primera grabación (sesión) de cada persona; prueba = las grabaciones posteriores.
Uso:  python -m src.try_me            (usa el último modelo de models/ y data/processed/polar)
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from src.checkpoint import load_checkpoint
from src.config import Config
from src.data import load_window_dir, split_public
from src.errors import CheckpointError, DataError, EvaluationError
from src.plotting import save_figure
from src.evaluate import (
    auc_score, eer, make_model_embed_fn, run_protocol, split_enroll_test, template_scores,
)


@dataclass
class TrialReport:
    owners: np.ndarray            # personas registradas (orden alfabético)
    test_ids: np.ndarray          # de quién es cada ventana de prueba (orden cronológico)
    scores: np.ndarray            # (ventanas de prueba, plantillas): similitud coseno
    threshold: float
    threshold_source: str
    best_threshold: float         # umbral que mejor separa en ESTOS datos (optimista)
    eer: float
    auc: float
    genuine: np.ndarray
    impostor: np.ndarray
    per_person: dict


def run_trial(table, embed_fn, cfg: Config, threshold: float | None = None) -> TrialReport:
    persons = table.persons()
    if len(persons) < 2:
        raise DataError(f"Se necesitan al menos 2 personas grabadas (hay {len(persons)}): "
                        "sin un impostor real no hay forma de saber si el sistema distingue.")
    short = [p for p in persons if len(table.sessions_of(p)) < 2]
    if short:
        counts = ", ".join(
            f"{p} ({n} {'grabación' if n == 1 else 'grabaciones'})"
            for p in persons for n in [len(table.sessions_of(p))]
        )
        raise DataError(
            f"Faltan grabaciones de {short}: cada persona necesita al menos 2 (una para registrarse y otra para "
            "probar), quitándose y volviéndose a poner la banda entre una y otra.\n"
            f"Grabaciones por persona: {counts}.\n"
            "Ojo: las grabaciones de la MISMA persona deben usar el mismo --id (p. ej. siempre --id kitzia); "
            "si la segunda vez usaste otro id (kitzia2), el programa la cuenta como otra persona.")
    enroll, test = split_enroll_test(table, "cross_session")
    owners, scores = template_scores(embed_fn(enroll.windows), enroll.participant_ids, embed_fn(test.windows))
    test_ids = test.participant_ids
    same = test_ids[:, None] == owners[None, :]
    genuine, impostor = scores[same], scores[~same]
    trial_eer, best_thr = eer(genuine, impostor)
    if threshold is None:
        thr, source = best_thr, "el que mejor separa estos mismos datos (optimista)"
    else:
        thr, source = threshold, "indicado por quien llama"
    per_person = {}
    for i, p in enumerate(owners):
        m = test_ids == p
        own = scores[m, i]
        other = np.delete(scores[m], i, axis=1)
        per_person[p] = {
            "n_windows": int(m.sum()),
            "mean_own": float(own.mean()),
            "mean_other": float(other.mean()),
            "accepted_own": float((own >= thr).mean()),
            "accepted_other": float((other >= thr).mean()),
            # Mismas tasas con el umbral que mejor separa ESTOS datos: si el umbral dado viene de otro
            # dataset (ECG-ID) puede quedar fuera de escala para el Polar y aceptar o rechazar a todos.
            "accepted_own_best": float((own >= best_thr).mean()),
            "accepted_other_best": float((other >= best_thr).mean()),
            "looks_like_self": bool(own.mean() > other.mean()),
        }
    return TrialReport(owners, test_ids, scores, thr, source, best_thr, trial_eer,
                       auc_score(genuine, impostor), genuine, impostor, per_person)


def public_threshold(embed_fn, public_dir, cfg: Config) -> float | None:
    """Umbral calculado con personas de ECG-ID (no con los datos del Polar): así la decisión
    "acepta/rechaza" no se afina con las mismas personas que se están probando."""
    d = Path(public_dir)
    if not d.exists() or not list(d.glob("*.npz")):
        return None
    try:
        _, val = split_public(load_window_dir(d, cfg.window_samples), cfg.pretrain_val_fraction, cfg.seed)
        enroll, test = split_enroll_test(val, cfg.pretrain_val_mode)
        t = run_protocol(embed_fn, enroll, test)
        return eer(t.genuine, t.impostor)[1]
    except (DataError, EvaluationError):
        return None


def format_trial(r: TrialReport) -> str:
    lines = ["", "=== Resultado de la prueba ===",
             f"Umbral de decisión: {r.threshold:.3f}  (fuente: {r.threshold_source})", ""]
    differs = abs(r.threshold - r.best_threshold) > 1e-9
    for p, s in r.per_person.items():
        verdict = "SÍ se parece más a sí mismo/a que a los demás" if s["looks_like_self"]             else "NO: se parece más a otra persona (mala señal)"
        lines += [f"{p}  ({s['n_windows']} ventanas de prueba)",
                  f"  similitud media con su propia huella: {s['mean_own']:.2f}   con la de los demás: {s['mean_other']:.2f}",
                  f"  con el umbral {r.threshold:.3f}: lo reconoce {s['accepted_own'] * 100:.0f}% | "
                  f"lo confunde con otra persona {s['accepted_other'] * 100:.0f}%"]
        if differs:
            lines.append(f"  con el mejor umbral para estos datos ({r.best_threshold:.3f}): lo reconoce "
                         f"{s['accepted_own_best'] * 100:.0f}% | lo confunde {s['accepted_other_best'] * 100:.0f}%")
        lines += [f"  -> {verdict}", ""]
    if differs:
        lines += ["NOTA: el umbral original no transfiere bien a estos datos (otro equipo/otra banda cambia la escala",
                  "de las similitudes). Lo importante es si hay un umbral que separe; ese es el 'mejor umbral'.",
                  "Con más grabaciones habría que calibrar el umbral con datos Polar (eso hace el fine-tuning).", ""]
    lines += [f"Separación general (mejor caso, con estos mismos datos): EER {r.eer * 100:.1f}%  |  AUC {r.auc:.3f}",
              "  (EER 0% = perfecto, 50% = azar.  AUC 1.0 = perfecto, 0.5 = azar)"]
    return "\n".join(lines)


def plot_trial(r: TrialReport, out_path) -> Path:
    fig = Figure(figsize=(12, 8))
    top, bottom = fig.subplots(2, 1)
    bins = np.linspace(min(r.genuine.min(), r.impostor.min()), max(r.genuine.max(), r.impostor.max()), 40)
    top.hist(r.impostor, bins=bins, alpha=0.6, color="tab:red", label="contra la huella de OTRA persona")
    top.hist(r.genuine, bins=bins, alpha=0.6, color="tab:green", label="contra su PROPIA huella")
    top.axvline(r.threshold, color="k", ls="--", label=f"umbral {r.threshold:.2f}")
    top.set_title("¿Se separan las dos distribuciones? (verde a la derecha de rojo = funciona)")
    top.set_xlabel("similitud coseno")
    top.legend()
    for i, owner in enumerate(r.owners):
        for p in r.owners:
            m = r.test_ids == p
            own = p == owner
            bottom.plot(r.scores[m, i], color="tab:green" if own else "tab:red", alpha=0.8 if own else 0.5,
                        lw=1.2, label=f"{p} vs huella de {owner}" + ("  (propia)" if own else ""))
    bottom.axhline(r.threshold, color="k", ls="--")
    bottom.set_title("Ventana a ventana: cada línea es una persona contra una huella")
    bottom.set_xlabel("ventana de prueba (en orden)")
    bottom.set_ylabel("similitud")
    bottom.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return save_figure(fig, out_path)


def _default_checkpoint(models_dir) -> Path:
    models_dir = Path(models_dir)
    chosen = models_dir / "modelo_ecg.pt"
    if chosen.exists():
        return chosen
    numbered = [(int(m.group(1)), f) for f in models_dir.glob("modelo_v*.pt")
                if (m := re.search(r"modelo_v(\d+)\.pt$", f.name))]
    if not numbered:
        raise CheckpointError(f"No hay ningún modelo en {models_dir}. Entrena uno: python -m src.train pretrain")
    return max(numbered)[1]


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.try_me", description="Prueba rápida con grabaciones del Polar")
    p.add_argument("--checkpoint", default=None, help="por defecto modelo_ecg.pt o el último modelo_vN.pt")
    p.add_argument("--polar-dir", default=f"{cfg.data_dir}/polar")
    p.add_argument("--public-dir", default=f"{cfg.data_dir}/public")
    p.add_argument("--out", default="reports/try_me.png")
    p.add_argument("--no-open", action="store_true")
    a = p.parse_args(argv)
    try:
        ckpt = Path(a.checkpoint) if a.checkpoint else _default_checkpoint(cfg.models_dir)
        embed = make_model_embed_fn(load_checkpoint(ckpt, cfg))
        table = load_window_dir(a.polar_dir, cfg.window_samples)
        thr = public_threshold(embed, a.public_dir, cfg)
        report = run_trial(table, embed, cfg, threshold=thr)
    except (DataError, CheckpointError, EvaluationError) as e:
        print("ERROR:", e)
        return 1
    print(f"Modelo: {ckpt.name}   personas: {', '.join(report.owners)}")
    if thr is None:
        print("AVISO: no hay datos públicos para fijar el umbral; se usa el que mejor separa estos mismos datos "
              "(resultado optimista).")
    print(format_trial(report))
    png = plot_trial(report, a.out)          # si a.out está abierta en el visor, guarda con otro nombre
    print(f"\nImagen: {png}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
