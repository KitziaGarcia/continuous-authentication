"""Entrenamiento: bucle genérico + (en tareas siguientes) los tres modos y la CLI."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint, promote, save_checkpoint
from src.config import Config, set_seed
from src.data import (
    Augmenter, PolarSplit, WindowDataset, assert_no_holdout_leak, load_window_dir,
    make_label_map, make_loader, split_polar, split_public, table_fingerprint,
)
from src.errors import CheckpointError, LeakageError
from src.evaluate import (
    eer, first_session_table, make_model_embed_fn, run_protocol, split_enroll_test,
)
from src.model import ECGEmbeddingNet


@dataclass
class FitResult:
    best_val_eer: float
    history: list
    epochs_run: int


def make_val_fn(enroll_table, test_table):
    """Devuelve una función red -> EER de validación.

    Se usa la EER y no la exactitud de clasificación porque lo que importa es separar
    personas que la red NO vio, no acertar la etiqueta de las que sí vio.
    """

    def val_fn(model) -> float:
        trials = run_protocol(make_model_embed_fn(model), enroll_table, test_table)
        return eer(trials.genuine, trials.impostor)[0]

    return val_fn


def fit(model, train_loader, val_fn, cfg: Config, epochs: int, lr: float,
        score_initial: bool = False) -> FitResult:
    if epochs < 1:
        raise ValueError("epochs debe ser >= 1")
    # Solo los parámetros entrenables: en la etapa A del fine-tuning las capas tempranas están congeladas.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best, best_state, bad, history = float("inf"), None, 0, []
    if score_initial:
        # El estado de partida también compite: si ninguna época nueva lo mejora (p. ej. la etapa B
        # del fine-tuning empeora lo que logró la etapa A) se conserva en vez de descartarlo.
        best, best_state = val_fn(model), copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model.logits(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        val = val_fn(model)
        history.append({"epoch": epoch, "train_loss": total / n, "val_eer": val})
        if val < best - 1e-9:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)   # nos quedamos con la mejor época, no con la última
    return FitResult(best, history, len(history))


def require_holdout(cfg: Config) -> None:
    """Sin hold-out no hay forma de medir impostores no vistos: mejor negarse a entrenar."""
    if len(cfg.holdout_ids) < cfg.min_holdout_people:
        raise LeakageError(
            f"Hay que definir al menos {cfg.min_holdout_people} participantes hold-out en "
            f"HOLDOUT_IDS (src/config.py); hay {len(cfg.holdout_ids)}."
        )


def _require_pretrain_checkpoint(path) -> None:
    """finetune solo parte de un checkpoint de preentrenamiento (datos públicos, sin Polar)."""
    side = Path(path).with_suffix(".json")
    if not side.exists():
        raise CheckpointError(f"Falta {side.name} junto a {Path(path).name}; no se puede verificar su origen")
    mode = json.loads(side.read_text(encoding="utf-8")).get("mode")
    if mode != "pretrain":
        raise CheckpointError(
            f"{Path(path).name} es de modo '{mode}', no 'pretrain'. Partir de un modelo ya entrenado con "
            "personas Polar podría filtrar al hold-out (o a otro reparto) a través de los pesos."
        )


def _loader(cfg: Config, table, label_map) -> object:
    ds = WindowDataset(table, label_map, Augmenter(cfg, cfg.seed))
    return make_loader(ds, cfg.batch_size, shuffle=True, seed=cfg.seed)


def pretrain(cfg: Config, public_train, public_val, models_dir) -> Path:
    """Etapa 1: la red aprende "cómo mirar un ECG" con muchas personas de un dataset público."""
    set_seed(cfg.seed)
    assert_no_holdout_leak(cfg.holdout_ids, {"public_train": public_train, "public_val": public_val})
    lm = make_label_map(public_train)
    model = ECGEmbeddingNet(cfg, n_classes=len(lm))
    enroll, test = split_enroll_test(public_val, cfg.pretrain_val_mode)
    res = fit(model, _loader(cfg, public_train, lm), make_val_fn(enroll, test), cfg,
              cfg.pretrain_epochs, cfg.pretrain_lr)
    return save_checkpoint(model, cfg, models_dir, "pretrain",
                           {"best_val_eer": res.best_val_eer, "history": res.history},
                           table_fingerprint(public_train))


def finetune(cfg: Config, split: PolarSplit, pretrained_ckpt, models_dir) -> Path:
    """Etapa 2: adaptar lo aprendido a la banda Polar y a nuestras personas.

    A) Cabeza nueva (hay otras personas) y capas tempranas congeladas: solo se ajusta lo
       más específico. B) Todo descongelado con tasa de aprendizaje baja: ajuste fino sin
       borrar lo que el preentrenamiento aprendió.
    """
    require_holdout(cfg)
    assert_no_holdout_leak(cfg.holdout_ids, {"train": split.train, "val": split.val})
    _require_pretrain_checkpoint(pretrained_ckpt)
    set_seed(cfg.seed)
    model = load_checkpoint(pretrained_ckpt, cfg)           # solo encoder, sin cabeza
    lm = make_label_map(split.train)
    model.set_head(len(lm))                                  # cabeza nueva para nuestras personas
    loader = _loader(cfg, split.train, lm)
    val_fn = make_val_fn(first_session_table(split.train), split.val)

    model.freeze_early_layers()
    stage_a = fit(model, loader, val_fn, cfg, cfg.finetune_head_epochs, cfg.finetune_head_lr)
    model.unfreeze_all()
    stage_b = fit(model, loader, val_fn, cfg, cfg.finetune_full_epochs, cfg.finetune_full_lr,
                  score_initial=True)
    return save_checkpoint(
        model, cfg, models_dir, "finetune",
        {"best_val_eer": stage_b.best_val_eer,
         "stage_a": {"best_val_eer": stage_a.best_val_eer, "history": stage_a.history},
         "stage_b": {"best_val_eer": stage_b.best_val_eer, "history": stage_b.history}},
        table_fingerprint(split.train), parent=Path(pretrained_ckpt).name,
    )


def polar_only(cfg: Config, split: PolarSplit, models_dir) -> Path:
    """Alternativa que funciona desde el día uno: entrenar solo con datos del Polar."""
    require_holdout(cfg)
    assert_no_holdout_leak(cfg.holdout_ids, {"train": split.train, "val": split.val})
    set_seed(cfg.seed)
    lm = make_label_map(split.train)
    model = ECGEmbeddingNet(cfg, n_classes=len(lm))
    res = fit(model, _loader(cfg, split.train, lm),
              make_val_fn(first_session_table(split.train), split.val), cfg,
              cfg.polar_only_epochs, cfg.polar_only_lr)
    return save_checkpoint(model, cfg, models_dir, "polar-only",
                           {"best_val_eer": res.best_val_eer, "history": res.history},
                           table_fingerprint(split.train))


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.train", description="Entrenamiento del modelo ECG")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pretrain", "finetune", "polar-only", "promote"):
        s = sub.add_parser(name)
        s.add_argument("--models-dir", default=cfg.models_dir)
        if name == "pretrain":
            s.add_argument("--public-dir", default=f"{cfg.data_dir}/public")
        if name in ("finetune", "polar-only"):
            s.add_argument("--polar-dir", default=f"{cfg.data_dir}/polar")
        if name == "finetune":
            s.add_argument("--from-checkpoint", required=True)
        if name == "promote":
            s.add_argument("--version", type=int, required=True)
    a = p.parse_args(argv)

    if a.cmd == "promote":
        print(promote(a.models_dir, a.version))
    elif a.cmd == "pretrain":
        tr, va = split_public(load_window_dir(a.public_dir, cfg.window_samples),
                              cfg.pretrain_val_fraction, cfg.seed)
        print(pretrain(cfg, tr, va, a.models_dir))
    else:
        require_holdout(cfg)
        split = split_polar(load_window_dir(a.polar_dir, cfg.window_samples), cfg.holdout_ids)
        if a.cmd == "finetune":
            print(finetune(cfg, split, a.from_checkpoint, a.models_dir))
        else:
            print(polar_only(cfg, split, a.models_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
