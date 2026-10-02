import dataclasses

import pytest
import torch

from src.config import Config
from src.errors import CheckpointError
from src.model import ECGEmbeddingNet


def _x(b=4):
    return torch.randn(b, 1, 650)


def test_forward_shape_and_unit_norm():
    net = ECGEmbeddingNet(Config(), n_classes=5).eval()
    z = net(_x())
    assert z.shape == (4, 128)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)


def test_embedding_dim_is_configurable():
    net = ECGEmbeddingNet(dataclasses.replace(Config(), embedding_dim=64)).eval()
    assert net(_x()).shape == (4, 64)


def test_eval_mode_is_deterministic():
    net = ECGEmbeddingNet(Config()).eval()
    x = _x()
    assert torch.equal(net(x), net(x))


def test_parameter_count_is_small():
    assert 150_000 < ECGEmbeddingNet(Config()).n_parameters() < 450_000


def test_logits_need_a_head():
    net = ECGEmbeddingNet(Config())
    with pytest.raises(RuntimeError, match="cabeza"):
        net.logits(_x())
    net.set_head(7)
    assert net.logits(_x()).shape == (4, 7)


def test_encoder_state_dict_excludes_head():
    net = ECGEmbeddingNet(Config(), n_classes=5)
    assert not any(k.startswith("head.") for k in net.encoder_state_dict())


def test_load_encoder_into_fresh_model_matches_embeddings():
    a = ECGEmbeddingNet(Config(), n_classes=5).eval()
    b = ECGEmbeddingNet(Config()).eval()
    b.load_encoder_state_dict(a.encoder_state_dict())
    x = _x()
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_load_encoder_rejects_foreign_keys():
    net = ECGEmbeddingNet(Config())
    with pytest.raises(CheckpointError):
        net.load_encoder_state_dict({"foreign.weight": torch.zeros(1)})


def test_tiny_overfit_run_drives_loss_down():
    torch.manual_seed(0)
    net = ECGEmbeddingNet(Config(), n_classes=3)
    x, y = torch.randn(12, 1, 650), torch.randint(0, 3, (12,))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    net.train()
    first = crit(net.logits(x), y).item()
    for _ in range(40):
        opt.zero_grad()
        loss = crit(net.logits(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_freeze_early_layers_only_leaves_last_stage_embed_and_head_trainable():
    net = ECGEmbeddingNet(Config(), n_classes=5)
    net.freeze_early_layers()
    trainable = {n.split(".")[0] + "." + n.split(".")[1] if n.startswith("stages") else n.split(".")[0]
                 for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"stages.3", "embed", "head"}
    net.unfreeze_all()
    assert all(p.requires_grad for p in net.parameters())


def test_frozen_layers_keep_batchnorm_stats_in_train_mode():
    net = ECGEmbeddingNet(Config(), n_classes=5)
    net.freeze_early_layers()
    before = net.stem[1].running_mean.clone()
    net.train()
    net(_x(8))
    assert torch.equal(before, net.stem[1].running_mean)       # congelado de verdad
    assert net.stages[3].training                              # la última etapa sí entrena
