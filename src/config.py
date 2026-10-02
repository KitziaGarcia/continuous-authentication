"""Configuración central. Todo parámetro que afecte resultados vive aquí (reproducibilidad)."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

# IDs de participantes Polar que NUNCA se usan para preentrenar ni entrenar.
# Sirven para evaluar con impostores que la red jamás vio. Editar UNA vez, al
# decidir quiénes serán el hold-out (mínimo 2) y no cambiarlo después, o los
# resultados dejan de ser comparables.  Ejemplo: ("P08", "P09")
HOLDOUT_IDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # --- señal ---
    fs: int = 130                 # Hz del Polar H10
    window_samples: int = 650     # 5 s * 130 Hz
    window_overlap: float = 0.5   # traslape entre ventanas
    band_low_hz: float = 0.5      # pasa-banda del brief (Nyquist = 65 Hz)
    band_high_hz: float = 40.0

    # --- modelo ---
    embedding_dim: int = 128
    stem_channels: int = 16
    stage_channels: tuple[int, ...] = (32, 64, 128, 128)
    kernel_size: int = 5

    # --- entrenamiento ---
    batch_size: int = 64
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    patience: int = 5                      # épocas sin mejorar la EER de validación
    pretrain_epochs: int = 30
    pretrain_lr: float = 1e-3
    finetune_head_epochs: int = 5          # etapa A: capas tempranas congeladas
    finetune_head_lr: float = 1e-3
    finetune_full_epochs: int = 20         # etapa B: todo descongelado, lr bajo
    finetune_full_lr: float = 1e-4
    polar_only_epochs: int = 40
    polar_only_lr: float = 1e-3

    # --- aumento de datos (solo entrenamiento); 0 desactiva cada uno ---
    jitter_std: float = 0.02
    magnitude_warp: float = 0.10
    warp_strength: float = 0.05
    wander_amplitude: float = 0.10

    # --- datos ---
    holdout_ids: tuple[str, ...] = HOLDOUT_IDS
    min_holdout_people: int = 2
    pretrain_val_fraction: float = 0.1
    pretrain_val_mode: str = "same_session"   # ECG-ID tiene pocas sesiones por persona
    data_dir: str = "data/processed"
    models_dir: str = "models"

    # --- evaluación ---
    bootstrap_samples: int = 1000

    def __post_init__(self) -> None:
        if not 64 <= self.embedding_dim <= 128:
            raise ValueError("embedding_dim debe estar entre 64 y 128 (brief, sección 4)")

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Fija todas las semillas para que un entrenamiento se pueda repetir."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
