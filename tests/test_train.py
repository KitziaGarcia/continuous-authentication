import dataclasses

import numpy as np
import pytest
import torch

from src.config import Config
from src.data import Augmenter, WindowDataset, make_label_map, make_loader
from src.evaluate import first_session_table
from src.model import ECGEmbeddingNet
from src.synthetic import make_synthetic_table
from src.train import FitResult, fit, make_val_fn


def _setup(epochs_cfg=None):
    cfg = dataclasses.replace(Config(), batch_size=16, patience=3, **(epochs_cfg or {}))
    t = make_synthetic_table(n_people=4, n_sessions=2, windows_per_session=12, seed=5)
    train = t.subset(t.session_ids == "s1")
    val = t.subset(t.session_ids == "s2")
    lm = make_label_map(train)
    loader = make_loader(WindowDataset(train, lm, Augmenter(cfg, 0)), cfg.batch_size, True, 0)
    net = ECGEmbeddingNet(cfg, n_classes=len(lm))
    return cfg, net, loader, make_val_fn(first_session_table(train), val)


def test_fit_runs_and_returns_history():
    cfg, net, loader, val_fn = _setup()
    torch.manual_seed(0)
    res = fit(net, loader, val_fn, cfg, epochs=3, lr=1e-3)
    assert isinstance(res, FitResult)
    assert res.epochs_run == len(res.history) <= 3
    assert {"epoch", "train_loss", "val_eer"} <= set(res.history[0])
    assert 0 <= res.best_val_eer <= 1


def test_fit_restores_best_weights():
    cfg, net, loader, val_fn = _setup()
    torch.manual_seed(0)
    res = fit(net, loader, val_fn, cfg, epochs=4, lr=1e-3)
    assert val_fn(net) == pytest.approx(res.best_val_eer, abs=1e-9)


def test_fit_early_stops_when_validation_never_improves():
    cfg, net, loader, _ = _setup()
    calls = {"n": 0}

    def flat_val(model):
        calls["n"] += 1
        return 0.3 if calls["n"] == 1 else 0.5      # empeora tras la primera época

    res = fit(net, loader, flat_val, cfg, epochs=20, lr=1e-3)
    assert res.epochs_run == 1 + cfg.patience       # 1 mejora + patience épocas sin mejorar
    assert res.best_val_eer == 0.3


def test_fit_only_updates_trainable_parameters():
    cfg, net, loader, val_fn = _setup()
    net.freeze_early_layers()
    stem_before = net.stem[0].weight.clone()
    head_before = net.head.weight.clone()
    fit(net, loader, val_fn, cfg, epochs=2, lr=1e-2)
    assert torch.equal(stem_before, net.stem[0].weight)
    assert not torch.equal(head_before, net.head.weight)


def test_fit_rejects_zero_epochs():
    cfg, net, loader, val_fn = _setup()
    with pytest.raises(ValueError):
        fit(net, loader, val_fn, cfg, epochs=0, lr=1e-3)


def test_training_loss_decreases_on_separable_synthetic_people():
    cfg, net, loader, val_fn = _setup()
    torch.manual_seed(0)
    res = fit(net, loader, val_fn, cfg, epochs=6, lr=2e-3)
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]


import json
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.data import split_polar, split_public
from src.errors import LeakageError
from src.train import finetune, main, polar_only, pretrain, require_holdout


def _cfg(**kw):
    base = dict(batch_size=16, patience=2, pretrain_epochs=2, finetune_head_epochs=1,
                finetune_full_epochs=2, polar_only_epochs=2, holdout_ids=("p4", "p5"),
                bootstrap_samples=10)
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def _polar_split(cfg):
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=41)
    return split_polar(t, cfg.holdout_ids)


def _public():
    t = make_synthetic_table(n_people=10, n_sessions=2, windows_per_session=6, seed=42, prefix="pub_")
    return split_public(t, 0.2, seed=0)


def _sidecar(p: Path):
    return json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))


def test_require_holdout_needs_at_least_two_people():
    with pytest.raises(LeakageError, match="hold-out"):
        require_holdout(dataclasses.replace(Config(), holdout_ids=()))
    with pytest.raises(LeakageError):
        require_holdout(dataclasses.replace(Config(), holdout_ids=("p1",)))
    require_holdout(dataclasses.replace(Config(), holdout_ids=("p1", "p2")))


def test_polar_only_saves_encoder_checkpoint(tmp_path):
    cfg = _cfg()
    path = polar_only(cfg, _polar_split(cfg), tmp_path)
    side = _sidecar(path)
    assert side["mode"] == "polar-only" and side["parent"] is None
    assert "best_val_eer" in side["metrics"] and side["holdout_ids"] == ["p4", "p5"]
    assert load_checkpoint(path, cfg).head is None


def test_pretrain_then_finetune_records_parent_and_both_stages(tmp_path):
    cfg = _cfg()
    tr, va = _public()
    pre = pretrain(cfg, tr, va, tmp_path)
    assert _sidecar(pre)["mode"] == "pretrain"
    fin = finetune(cfg, _polar_split(cfg), pre, tmp_path)
    side = _sidecar(fin)
    assert side["mode"] == "finetune" and side["parent"] == pre.name
    assert {"stage_a", "stage_b", "best_val_eer"} <= set(side["metrics"])
    assert fin.name == "modelo_v2.pt"


def test_finetune_changes_the_pretrained_weights(tmp_path):
    cfg = _cfg()
    tr, va = _public()
    pre = pretrain(cfg, tr, va, tmp_path)
    fin = finetune(cfg, _polar_split(cfg), pre, tmp_path)
    a = torch.load(pre, weights_only=True)
    b = torch.load(fin, weights_only=True)
    assert any(not torch.equal(a[k], b[k]) for k in a if a[k].dtype.is_floating_point)


def test_training_refuses_when_holdout_person_is_in_public_data(tmp_path):
    cfg = _cfg(holdout_ids=("pub_0", "p5"))
    tr, va = _public()
    with pytest.raises(LeakageError, match="pub_0"):
        pretrain(cfg, tr, va, tmp_path)


def test_polar_modes_refuse_without_holdout(tmp_path):
    cfg = _cfg(holdout_ids=())
    t = make_synthetic_table(n_people=4, n_sessions=2, windows_per_session=6, seed=43)
    with pytest.raises(LeakageError):
        polar_only(cfg, split_polar(t, ()), tmp_path)


def test_training_is_reproducible_with_same_seed(tmp_path):
    cfg = _cfg()
    split = _polar_split(cfg)
    a = polar_only(cfg, split, tmp_path / "a")
    b = polar_only(cfg, split, tmp_path / "b")
    sa, sb = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    # allclose y no equal: los hilos de la CPU pueden sumar en otro orden y cambiar el último decimal
    assert all(torch.allclose(sa[k].float(), sb[k].float(), atol=1e-5) for k in sa)


def test_cli_promote(tmp_path, capsys):
    cfg = _cfg()
    polar_only(cfg, _polar_split(cfg), tmp_path)
    assert main(["promote", "--version", "1", "--models-dir", str(tmp_path)]) == 0
    assert (tmp_path / "modelo_ecg.pt").exists()


# --- Revisión final ---
def test_fit_with_score_initial_keeps_starting_weights_if_no_epoch_improves():
    cfg, net, loader, _ = _setup()
    start = {k: v.clone() for k, v in net.state_dict().items()}
    calls = {"n": 0}

    def val(model):
        calls["n"] += 1
        return 0.1 if calls["n"] == 1 else 0.5      # el estado inicial es el mejor

    res = fit(net, loader, val, cfg, epochs=3, lr=1e-2, score_initial=True)
    assert res.best_val_eer == 0.1
    assert all(torch.equal(start[k], v) for k, v in net.state_dict().items())


def test_finetune_refuses_a_checkpoint_that_is_not_a_pretrain(tmp_path):
    from src.errors import CheckpointError

    cfg = _cfg()
    split = _polar_split(cfg)
    pol = polar_only(cfg, split, tmp_path)
    with pytest.raises(CheckpointError, match="pretrain"):
        finetune(cfg, split, pol, tmp_path)
