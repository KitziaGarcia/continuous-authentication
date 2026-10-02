import dataclasses
import json

import pytest
import torch

from src.checkpoint import (
    REQUIRED_SIDECAR_FIELDS,
    load_checkpoint,
    next_version,
    promote,
    save_checkpoint,
)
from src.config import Config
from src.errors import CheckpointError
from src.model import ECGEmbeddingNet


def _save(tmp_path, model=None, cfg=None, **kw):
    cfg = cfg or Config(holdout_ids=("P8", "P9"))
    model = model or ECGEmbeddingNet(cfg, n_classes=4)
    return save_checkpoint(model, cfg, tmp_path, mode=kw.get("mode", "polar-only"),
                           metrics={"best_val_eer": 0.2}, data_fingerprint="abc123", parent=kw.get("parent"))


def test_versions_increment(tmp_path):
    assert next_version(tmp_path) == 1
    p1, p2 = _save(tmp_path), _save(tmp_path)
    assert p1.name == "modelo_v1.pt" and p2.name == "modelo_v2.pt"
    assert next_version(tmp_path) == 3


def test_sidecar_has_all_required_fields(tmp_path):
    p = _save(tmp_path)
    side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    assert all(f in side for f in REQUIRED_SIDECAR_FIELDS)
    assert side["holdout_ids"] == ["P8", "P9"] and side["version"] == 1
    assert side["config"]["embedding_dim"] == 128


def test_saved_weights_exclude_head(tmp_path):
    p = _save(tmp_path)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    assert not any(k.startswith("head.") for k in sd)


def test_roundtrip_gives_identical_embeddings(tmp_path):
    cfg = Config()
    net = ECGEmbeddingNet(cfg, n_classes=4).eval()
    p = _save(tmp_path, model=net, cfg=cfg)
    loaded = load_checkpoint(p, cfg)
    x = torch.randn(3, 1, 650)
    assert loaded.head is None and not loaded.training
    assert torch.allclose(net(x), loaded(x), atol=1e-6)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(CheckpointError, match="no existe"):
        load_checkpoint(tmp_path / "nada.pt", Config())


# --- Review Focus 5: embedding_dim distinto ---
def test_load_with_different_embedding_dim_raises_readable_error(tmp_path):
    p = _save(tmp_path, cfg=dataclasses.replace(Config(), embedding_dim=64),
              model=ECGEmbeddingNet(dataclasses.replace(Config(), embedding_dim=64)))
    with pytest.raises(CheckpointError, match="embedding_dim"):
        load_checkpoint(p, Config())            # Config() usa 128


def test_promote_copies_weights_and_sidecar(tmp_path):
    _save(tmp_path)
    out = promote(tmp_path, 1)
    assert out.name == "modelo_ecg.pt" and out.exists()
    assert (tmp_path / "modelo_ecg.json").exists()
    assert out.read_bytes() == (tmp_path / "modelo_v1.pt").read_bytes()


def test_promote_unknown_version_raises(tmp_path):
    with pytest.raises(CheckpointError):
        promote(tmp_path, 9)
