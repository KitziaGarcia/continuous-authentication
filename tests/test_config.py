import dataclasses
import json

import pytest
import torch

from src.config import Config, HOLDOUT_IDS, set_seed


def test_window_is_five_seconds_at_130hz():
    cfg = Config()
    assert cfg.fs == 130
    assert cfg.window_samples == 5 * cfg.fs == 650


def test_config_is_json_serializable():
    json.dumps(Config().to_dict())


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().seed = 1


def test_embedding_dim_must_be_64_to_128():
    with pytest.raises(ValueError):
        Config(embedding_dim=256)
    Config(embedding_dim=64)


def test_holdout_default_comes_from_module_constant():
    assert Config().holdout_ids == HOLDOUT_IDS


def test_set_seed_makes_torch_reproducible():
    set_seed(7)
    a = torch.rand(3)
    set_seed(7)
    b = torch.rand(3)
    assert torch.equal(a, b)
