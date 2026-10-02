"""Checkpoints versionados. Se guarda solo el encoder (sin cabeza) + un JSON con el contexto."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import Config
from src.errors import CheckpointError
from src.model import ECGEmbeddingNet

REQUIRED_SIDECAR_FIELDS = (
    "version", "mode", "created_utc", "seed", "holdout_ids",
    "data_fingerprint", "config", "metrics", "parent",
)
_PATTERN = re.compile(r"modelo_v(\d+)\.pt$")


def next_version(models_dir) -> int:
    models_dir = Path(models_dir)
    found = [int(m.group(1)) for f in models_dir.glob("modelo_v*.pt") if (m := _PATTERN.search(f.name))]
    return max(found, default=0) + 1


def save_checkpoint(model, cfg: Config, models_dir, mode: str, metrics: dict,
                    data_fingerprint: str, parent: str | None = None) -> Path:
    """Guarda modelo_vN.pt (pesos del encoder) y modelo_vN.json (cómo se obtuvo).

    El JSON guarda la lista hold-out y la huella de datos: así, meses después, se sabe con
    qué personas se entrenó y cuáles eran impostores no vistos.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(models_dir)
    path = models_dir / f"modelo_v{version}.pt"
    torch.save(model.encoder_state_dict(), path)
    sidecar = {
        "version": version,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "holdout_ids": list(cfg.holdout_ids),
        "data_fingerprint": data_fingerprint,
        "config": cfg.to_dict(),
        "metrics": metrics,
        "parent": parent,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_checkpoint(path, cfg: Config) -> ECGEmbeddingNet:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"El checkpoint {path} no existe")
    model = ECGEmbeddingNet(cfg)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.load_encoder_state_dict(sd)
    except RuntimeError as e:     # tamaños de tensor distintos
        raise CheckpointError(
            f"{path.name} no es compatible con la configuración actual "
            f"(embedding_dim={cfg.embedding_dim}, canales={cfg.stage_channels}). "
            "Revisa el JSON junto al checkpoint para ver con qué configuración se entrenó."
        ) from e
    return model.eval()


def promote(models_dir, version: int) -> Path:
    """Marca una versión como la elegida copiándola a modelo_ecg.pt (copia, no enlace
    simbólico: en Windows los enlaces requieren permisos de administrador)."""
    models_dir = Path(models_dir)
    src = models_dir / f"modelo_v{version}.pt"
    if not src.exists():
        raise CheckpointError(f"No existe {src}")
    dst = models_dir / "modelo_ecg.pt"
    shutil.copyfile(src, dst)
    shutil.copyfile(src.with_suffix(".json"), dst.with_suffix(".json"))
    return dst
