"""La red: 1D-CNN residual pequeña que convierte una ventana de ECG en una huella (embedding)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.errors import CheckpointError


class ResidualBlock(nn.Module):
    """Dos convoluciones + atajo (skip). El atajo deja pasar la señal original, lo que
    facilita entrenar y evita que la red olvide información útil en capas profundas.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        # El atajo usa conv 1x1 con el mismo stride para que las dimensiones coincidan.
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ECGEmbeddingNet(nn.Module):
    _early_frozen = False

    def __init__(self, cfg: Config, n_classes: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv1d(1, cfg.stem_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(cfg.stem_channels),
            nn.ReLU(),
        )
        chans = [cfg.stem_channels, *cfg.stage_channels]
        self.stages = nn.ModuleList(
            ResidualBlock(chans[i], chans[i + 1], cfg.kernel_size) for i in range(len(cfg.stage_channels))
        )
        self.pool = nn.AdaptiveAvgPool1d(1)       # promedio en el tiempo: no importa en qué punto cae el latido
        self.embed = nn.Linear(chans[-1], cfg.embedding_dim)
        # La cabeza de clasificación SOLO existe para entrenar; no se guarda ni se usa después.
        self.head = nn.Linear(cfg.embedding_dim, n_classes) if n_classes else None

    # --- cálculo ---
    def embed_raw(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        return self.embed(self.pool(h).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizar a longitud 1 hace que el producto punto sea la similitud coseno.
        return F.normalize(self.embed_raw(x), dim=1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("El modelo no tiene cabeza de clasificación; usa set_head(n_clases)")
        # La cabeza ve la capa penúltima SIN normalizar (brief: "penúltima capa como embedding").
        return self.head(self.embed_raw(x))

    def set_head(self, n_classes: int) -> None:
        self.head = nn.Linear(self.cfg.embedding_dim, n_classes)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- guardado: solo el "cómo mirar un ECG", sin la cabeza ---
    def encoder_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("head.")}

    def load_encoder_state_dict(self, sd: dict) -> None:
        res = self.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys if not k.startswith("head.")] + list(res.unexpected_keys)
        if bad:
            raise CheckpointError(f"El checkpoint no coincide con la red. Claves problemáticas: {bad[:5]}")

    # --- fine-tuning ---
    def freeze_early_layers(self) -> None:
        """Etapa A del fine-tuning: las capas tempranas (formas básicas) no cambian."""
        for m in [self.stem, *list(self.stages)[:-1]]:
            for p in m.parameters():
                p.requires_grad = False
        self._early_frozen = True
        self.train(self.training)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self._early_frozen = False
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._early_frozen:
            # BatchNorm actualiza sus estadísticas aunque no tenga gradiente; hay que
            # ponerlo en eval para que "congelado" signifique realmente congelado.
            self.stem.eval()
            for s in list(self.stages)[:-1]:
                s.eval()
        return self
