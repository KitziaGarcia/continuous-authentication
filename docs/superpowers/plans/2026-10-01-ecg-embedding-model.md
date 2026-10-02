# ECG Embedding Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train and evaluate a small residual 1D-CNN that turns a 5 s Polar H10 ECG window into an L2-normalized embedding, with public-dataset pretraining + Polar fine-tuning as the main path and Polar-only training as the day-one fallback.

**Architecture:** Plain Python modules with one responsibility each (`config`, `data`, `windowing`, `model`, `evaluate`, `checkpoint`, `train`). Data lives in `WindowTable` objects (windows + participant/session metadata) and every split is by person. The network is trained as a classifier (cross-entropy) and the penultimate layer is saved as the embedding; verification is cosine similarity against a mean-embedding template.

**Tech Stack:** Python 3.10+ (3.11 or 3.12 recommended, see Task 0), PyTorch (CPU), NumPy, SciPy, scikit-learn, Matplotlib, wfdb (ECG-ID download), pytest, nbformat/Jupyter.

**Spec:** `docs/superpowers/specs/2026-10-01-ecg-embedding-model-design.md` (parent brief: `docs/brief_continuous_ecg_authentication.md`)

## Global Constraints

- Window: **650 samples = 5 s at 130 Hz**, z-scored per window, tensor shape `(1, 650)`.
- Embedding: **64–128 dims**, L2-normalized; default 128. Score = **cosine similarity** against a template (mean of enrollment embeddings).
- Network: small residual 1D-CNN, stem 16 ch, 4 stride-2 residual stages with channels **32 → 64 → 128 → 128**, global average pooling, Linear embedding, classification head used **only in training** (never saved).
- Training runs on **CPU** in minutes to about an hour. Loss: **cross-entropy with label smoothing only** (no triplet/ArcFace in v1).
- **Splits are always by person, never by window.** The hold-out list lives in `src/config.py` (`HOLDOUT_IDS`), needs **at least 2 Polar participants**, and the code must **refuse to train** if a held-out `participant_id` shows up in any training or validation data.
- **Threshold is chosen at the EER on validation data only**; hold-out people are never used to choose it.
- **Confidence intervals: bootstrap resampling by person**, not by window. Every reported number prints its genuine/impostor pair counts.
- **Early stopping on validation EER**, not accuracy.
- **Reproducibility:** fixed seeds, one `Config`, one command per stage. Checkpoints are `models/modelo_vN.pt` + JSON sidecar; `models/modelo_ecg.pt` is the chosen version.
- **Code comments and docstrings are in Spanish**, explaining the *why* (brief, section 9). Identifiers and test names are in English.
- **The user commits manually.** Every task ends with a *Checkpoint* step that proposes a commit message. Executors must **not** run `git commit` unless the user asks.
- Out of scope: BLE acquisition, the full preprocessing module (only a minimal `windowing.py` here), decision logic, web app, SVM baseline.

## Review Focus

Inputs the spec implies but its tests would not otherwise exercise, most likely first. Each one is pinned by a test in the task named in brackets.

1. **A Polar participant with only one session** → clear error naming the person and asking for a second session, not a crash or silent train/val leak. [Task 3]
2. **Hold-out id that does not exist in the data (typo, e.g. `"P7"` vs `"P07"`)** → error. Silently training with fewer hold-outs than intended would invalidate the evaluation. [Task 3]
3. **Recording with lead-off / flat / NaN segments or shorter than one window** → flat windows are dropped, NaNs raise `DataError`, a too-short recording yields zero windows and no crash. [Task 2]
4. **Fewer than 3 people in a bootstrap** → the confidence interval is reported as `n/a`, never a fake narrow interval. [Task 8]
5. **Loading a checkpoint trained with a different `embedding_dim`** → `CheckpointError` with a readable message, not a raw PyTorch shape error. [Task 9]

---

## File Structure

```
requirements.txt
pytest.ini
.gitignore
src/
  __init__.py
  errors.py        custom exceptions (DataError, LeakageError, EvaluationError, CheckpointError)
  config.py        Config dataclass, HOLDOUT_IDS, set_seed
  data.py          WindowTable, npz I/O, splits by person, leakage guards, augmentation, torch Dataset
  windowing.py     minimal bandpass / resample / segment / z-score (stand-in until the preprocessing spec)
  synthetic.py     fake ECG people for tests and notebooks
  model.py         ResidualBlock, ECGEmbeddingNet
  evaluate.py      EER/ROC/AUC/FAR/FRR, trials, protocols, bootstrap, report, compare CLI
  checkpoint.py    versioned save/load/promote + JSON sidecar
  train.py         fit loop, pretrain / finetune / polar-only drivers, CLI
  public_data.py   ECG-ID download and conversion to 130 Hz windows
tests/
  conftest.py
  test_config.py test_windowing.py test_synthetic.py test_data.py test_model.py
  test_evaluate.py test_checkpoint.py test_train.py test_public_data.py
notebooks/
  build_notebooks.py                 generates the two notebooks below (cells live in code, easy to diff)
  02_model_and_training.ipynb
  03_finetuning_explained.ipynb
data/raw/  data/processed/polar/  data/processed/public/   (git-ignored)
models/    reports/    (git-ignored except .gitkeep)
README.md
```

The spec lists augmentation and loaders inside `data.py` (kept) and does not list `windowing.py`, `synthetic.py`, `checkpoint.py`, `public_data.py`, `errors.py`. These are small helpers split out so no file does two jobs and so `train.py` and `evaluate.py` do not import each other.

**Data file contract (used by every task).** One `.npz` per recording in `data/processed/<source>/`, with keys: `windows` float32 `(N, 650)` or `(N, 1, 650)`, and scalar strings `participant_id`, `session_id`, `activity`, `source`. **`session_id` values must sort chronologically** (use `2026-10-01_s1`, `2026-10-03_s2`, ...) because "first session = enrollment, last session = validation". Windows inside a file must be in time order.

---

### Task 0: Environment, scaffolding and `Config`

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `src/__init__.py`, `src/errors.py`, `src/config.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `src.errors.{DataError, LeakageError, EvaluationError, CheckpointError}`; `src.config.Config` (frozen dataclass, fields below), `src.config.HOLDOUT_IDS: tuple[str, ...]`, `src.config.set_seed(seed: int) -> None`, `Config.to_dict() -> dict`.

- [ ] **Step 1: Create the virtual environment and check PyTorch installs**

The machine's default Python is 3.14, which may not have PyTorch wheels yet. Prefer 3.12 or 3.11 if available (`py -0` lists installed versions).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python --version
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

Expected: prints a torch version. If `py -3.12` does not exist, install Python 3.12 from python.org first. If the torch install fails on every available version, stop and tell the user (Colab fallback is the alternative).

- [ ] **Step 2: Write `requirements.txt`, `pytest.ini`, `.gitignore`**

`requirements.txt`:
```
numpy
scipy
scikit-learn
matplotlib
pytest
wfdb
nbformat
nbconvert
ipykernel
```
(torch is installed separately in Step 1 to get the CPU wheel.)

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
data/raw/
data/processed/
models/*.pt
models/*.json
reports/*.md
```

Then:
```powershell
pip install -r requirements.txt
```

- [ ] **Step 3: Write the failing test**

`tests/conftest.py` (empty file is fine; it marks the tests root):
```python
# Raíz de pruebas: pytest.ini ya añade la raíz del repo al path.
```

`tests/test_config.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src'`.

- [ ] **Step 5: Write minimal implementation**

`src/__init__.py`: empty file.

`src/errors.py`:
```python
"""Excepciones propias: mensajes claros en vez de errores crípticos de numpy/torch."""


class DataError(ValueError):
    """Los datos no cumplen el contrato (forma, NaN, sesiones insuficientes...)."""


class LeakageError(RuntimeError):
    """Una persona reservada (hold-out) apareció donde no debía: invalida la evaluación."""


class EvaluationError(ValueError):
    """No hay suficientes datos o pares para evaluar de forma válida."""


class CheckpointError(Exception):
    """El checkpoint no existe o no es compatible con la configuración actual.

    Hereda de Exception y NO de RuntimeError a propósito: load_checkpoint captura los
    RuntimeError de PyTorch (tamaños distintos) y no debe capturar este error por accidente.
    """
```

`src/config.py`:
```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 7: Checkpoint**

Propose commit (do not run unless asked): `chore: scaffolding, Config and custom errors`

---

### Task 1: `WindowTable`, `.npz` I/O and synthetic people

**Files:**
- Create: `src/data.py`, `src/synthetic.py`
- Test: `tests/test_data.py`, `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `Config`, `DataError`.
- Produces:
  - `WindowTable(windows: ndarray (N,1,650) float32, participant_ids: ndarray[str], session_ids: ndarray[str], activities: ndarray[str], source: str)` with `__len__`, `validate(window_samples=650)`, `subset(mask: ndarray[bool]) -> WindowTable`, `persons() -> list[str]`, `sessions_of(pid) -> list[str]`.
  - `concat_tables(tables: list[WindowTable]) -> WindowTable`
  - `save_window_file(path, windows, participant_id, session_id, activity, source) -> None`
  - `load_window_dir(directory, window_samples=650) -> WindowTable`
  - `table_fingerprint(table) -> str` (12 hex chars)
  - `make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=40, seed=0, prefix="p", window_samples=650, fs=130) -> WindowTable` (ids `f"{prefix}{i}"`, sessions `"s1".."sN"`).

- [ ] **Step 1: Write the failing tests**

`tests/test_synthetic.py`:
```python
import numpy as np

from src.synthetic import make_synthetic_table


def test_shape_and_metadata():
    t = make_synthetic_table(n_people=3, n_sessions=2, windows_per_session=5, seed=1)
    assert t.windows.shape == (3 * 2 * 5, 1, 650)
    assert t.windows.dtype == np.float32
    assert t.persons() == ["p0", "p1", "p2"]
    assert t.sessions_of("p0") == ["s1", "s2"]


def test_windows_are_zscored():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=4, seed=2)
    assert np.allclose(t.windows.mean(axis=2), 0, atol=1e-4)
    assert np.allclose(t.windows.std(axis=2), 1, atol=1e-3)


def test_same_seed_same_data_different_seed_different_data():
    a = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=5)
    b = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=5)
    c = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, seed=6)
    assert np.array_equal(a.windows, b.windows)
    assert not np.array_equal(a.windows, c.windows)


def test_prefix_changes_ids():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="pub_")
    assert t.persons() == ["pub_0", "pub_1"]
```

`tests/test_data.py` (this task's part; later tasks append):
```python
import numpy as np
import pytest

from src.data import (
    WindowTable,
    concat_tables,
    load_window_dir,
    save_window_file,
    table_fingerprint,
)
from src.errors import DataError
from src.synthetic import make_synthetic_table


def test_validate_rejects_wrong_window_length():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2)
    bad = WindowTable(t.windows[:, :, :100], t.participant_ids, t.session_ids, t.activities, "synthetic")
    with pytest.raises(DataError, match="650"):
        bad.validate()


def test_validate_rejects_nan():
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2)
    t.windows[0, 0, 0] = np.nan
    with pytest.raises(DataError, match="NaN"):
        t.validate()


def test_subset_and_persons():
    t = make_synthetic_table(n_people=3, n_sessions=2, windows_per_session=2)
    only_p1 = t.subset(t.participant_ids == "p1")
    assert only_p1.persons() == ["p1"]
    assert len(only_p1) == 4


def test_concat_tables():
    a = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=2, prefix="a")
    b = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3, prefix="b")
    c = concat_tables([a, b])
    assert len(c) == 5 and c.persons() == ["a0", "b0"]


def test_save_and_load_roundtrip(tmp_path):
    t = make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=3, seed=3)
    save_window_file(tmp_path / "p0__s1.npz", t.windows, "p0", "s1", "rest", "synthetic")
    back = load_window_dir(tmp_path)
    assert back.persons() == ["p0"] and back.sessions_of("p0") == ["s1"]
    assert back.source == "synthetic"
    assert np.allclose(back.windows, t.windows)


def test_load_empty_dir_raises(tmp_path):
    with pytest.raises(DataError, match="npz"):
        load_window_dir(tmp_path)


def test_fingerprint_changes_when_data_changes():
    a = make_synthetic_table(n_people=2, n_sessions=2, windows_per_session=3)
    b = make_synthetic_table(n_people=2, n_sessions=2, windows_per_session=4)
    assert table_fingerprint(a) == table_fingerprint(a)
    assert table_fingerprint(a) != table_fingerprint(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic.py tests/test_data.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.synthetic'`.

- [ ] **Step 3: Write `src/data.py` (table + I/O part)**

```python
"""Datos: tabla de ventanas con metadatos, lectura/escritura, particiones por persona y aumento."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.errors import DataError


@dataclass
class WindowTable:
    """Ventanas de ECG (N, 1, 650) + quién, cuándo y haciendo qué.

    Guardamos los metadatos junto a las ventanas para poder particionar SIEMPRE por
    persona/sesión y no por ventana (ventanas vecinas se parecen mucho: mezclarlas
    entre train y test inflaría los resultados).
    """

    windows: np.ndarray
    participant_ids: np.ndarray
    session_ids: np.ndarray
    activities: np.ndarray
    source: str

    def __len__(self) -> int:
        return len(self.windows)

    def validate(self, window_samples: int = 650) -> None:
        w = self.windows
        if w.ndim != 3 or w.shape[1] != 1 or w.shape[2] != window_samples:
            raise DataError(
                f"Se esperaban ventanas de forma (N, 1, {window_samples}) y llegó {w.shape}"
            )
        for name in ("participant_ids", "session_ids", "activities"):
            if len(getattr(self, name)) != len(w):
                raise DataError(f"'{name}' no tiene la misma longitud que las ventanas")
        if not np.isfinite(w).all():
            raise DataError("Hay NaN o inf en las ventanas (¿pérdida de señal sin limpiar?)")

    def subset(self, mask: np.ndarray) -> "WindowTable":
        return WindowTable(
            self.windows[mask],
            self.participant_ids[mask],
            self.session_ids[mask],
            self.activities[mask],
            self.source,
        )

    def persons(self) -> list[str]:
        return sorted(set(self.participant_ids.tolist()))

    def sessions_of(self, pid: str) -> list[str]:
        # Orden lexicográfico = cronológico (contrato: los session_id se ordenan por fecha).
        return sorted(set(self.session_ids[self.participant_ids == pid].tolist()))


def concat_tables(tables: list[WindowTable]) -> WindowTable:
    if not tables:
        raise DataError("No hay tablas que concatenar")
    sources = {t.source for t in tables}
    return WindowTable(
        np.concatenate([t.windows for t in tables]),
        np.concatenate([t.participant_ids for t in tables]),
        np.concatenate([t.session_ids for t in tables]),
        np.concatenate([t.activities for t in tables]),
        sources.pop() if len(sources) == 1 else "mixed",
    )


def save_window_file(path, windows, participant_id, session_id, activity, source) -> None:
    """Guarda UNA grabación ya procesada (ver contrato de datos en el plan)."""
    w = np.asarray(windows, dtype=np.float32)
    if w.ndim == 2:
        w = w[:, None, :]
    np.savez_compressed(
        path,
        windows=w,
        participant_id=participant_id,
        session_id=session_id,
        activity=activity,
        source=source,
    )


def load_window_dir(directory, window_samples: int = 650) -> WindowTable:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise DataError(f"No hay archivos .npz en {directory}")
    tables = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            w = z["windows"].astype(np.float32)
            if w.ndim == 2:
                w = w[:, None, :]
            n = len(w)
            tables.append(
                WindowTable(
                    w,
                    np.full(n, str(z["participant_id"])),
                    np.full(n, str(z["session_id"])),
                    np.full(n, str(z["activity"])),
                    str(z["source"]),
                )
            )
    table = concat_tables(tables)
    table.validate(window_samples)
    return table


def table_fingerprint(table: WindowTable) -> str:
    """Huella corta de qué datos se usaron; va en el JSON del checkpoint."""
    h = hashlib.sha256()
    pairs = sorted(set(zip(table.participant_ids.tolist(), table.session_ids.tolist())))
    for pid, sid in pairs:
        n = int(((table.participant_ids == pid) & (table.session_ids == sid)).sum())
        h.update(f"{pid}|{sid}|{n};".encode())
    return h.hexdigest()[:12]
```

- [ ] **Step 4: Write `src/synthetic.py`**

```python
"""ECG sintético: personas con forma de latido distinta y variación entre sesiones.

No es ECG real; sirve para probar el código y para que los notebooks corran sin datos.
Cada persona tiene su propia frecuencia cardiaca y amplitudes de las ondas P, QRS y T.
"""
from __future__ import annotations

import numpy as np

from src.data import WindowTable


def _gauss(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _one_window(rng, params: dict, window_samples: int, fs: int) -> np.ndarray:
    t = np.arange(window_samples) / fs
    period = 60.0 / (params["hr"] * (1 + rng.normal(0, 0.02)))
    first = -rng.uniform(0, period)             # fase aleatoria: la ventana no empieza en el pico R
    x = np.zeros(window_samples)
    for beat in np.arange(first, t[-1] + period, period):
        x += _gauss(t, beat - 0.16, 0.025, params["p_amp"])               # onda P
        x += _gauss(t, beat, params["qrs_w"], 1.0)                        # pico R
        x += _gauss(t, beat + 0.03, params["qrs_w"], -0.15)               # onda S
        x += _gauss(t, beat + params["t_off"], 0.05, params["t_amp"])     # onda T
    x += rng.normal(0, 0.02, window_samples)
    return (x - x.mean()) / (x.std() + 1e-8)


def make_synthetic_table(
    n_people: int = 6,
    n_sessions: int = 3,
    windows_per_session: int = 40,
    seed: int = 0,
    prefix: str = "p",
    window_samples: int = 650,
    fs: int = 130,
) -> WindowTable:
    rng = np.random.default_rng(seed)
    wins, pids, sids = [], [], []
    for i in range(n_people):
        base = {
            "hr": rng.uniform(55, 95),
            "p_amp": rng.uniform(0.10, 0.30),
            "qrs_w": rng.uniform(0.008, 0.020),
            "t_amp": rng.uniform(0.20, 0.50),
            "t_off": rng.uniform(0.22, 0.38),
        }
        for s in range(n_sessions):
            # Variación entre sesiones: colocación de la banda y estado fisiológico distintos.
            sess = {k: v * (1 + rng.normal(0, 0.05)) for k, v in base.items()}
            sess["hr"] = base["hr"] + rng.normal(0, 3)
            for _ in range(windows_per_session):
                wins.append(_one_window(rng, sess, window_samples, fs))
                pids.append(f"{prefix}{i}")
                sids.append(f"s{s + 1}")
    n = len(wins)
    table = WindowTable(
        np.asarray(wins, dtype=np.float32)[:, None, :],
        np.array(pids),
        np.array(sids),
        np.full(n, "rest"),
        "synthetic",
    )
    table.validate(window_samples)
    return table
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_synthetic.py tests/test_data.py -v`
Expected: 11 passed.

- [ ] **Step 6: Checkpoint**

Propose commit: `feat: WindowTable, npz I/O and synthetic ECG generator`

---

### Task 2: `windowing.py` (minimal preprocessing stand-in)

**Files:**
- Create: `src/windowing.py`
- Test: `tests/test_windowing.py`

**Interfaces:**
- Consumes: `Config`, `DataError`.
- Produces:
  - `bandpass(sig, fs, low, high, order=4) -> ndarray`
  - `resample_to_fs(sig, fs_in, fs_out) -> ndarray`
  - `segment(sig, window_samples, overlap) -> ndarray (N, window_samples)` (zero rows if the signal is shorter)
  - `zscore_windows(w, eps=1e-8) -> ndarray`
  - `signal_to_windows(sig, fs_in, cfg) -> ndarray (N, 1, 650) float32` (band-pass at `fs_in`, resample to `cfg.fs`, segment, drop flat windows, z-score; raises `DataError` on NaN/inf).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from src.config import Config
from src.errors import DataError
from src.windowing import (
    bandpass,
    resample_to_fs,
    segment,
    signal_to_windows,
    zscore_windows,
)


def test_bandpass_removes_dc_and_keeps_10hz():
    fs = 130
    t = np.arange(0, 10, 1 / fs)
    sig = 5.0 + np.sin(2 * np.pi * 10 * t)
    out = bandpass(sig, fs, 0.5, 40.0)
    assert abs(out.mean()) < 0.05
    assert out.std() == pytest.approx(np.sin(2 * np.pi * 10 * t).std(), rel=0.1)


def test_resample_500_to_130_length():
    sig = np.random.default_rng(0).normal(size=5000)  # 10 s a 500 Hz
    out = resample_to_fs(sig, 500, 130)
    assert len(out) == 1300


def test_resample_same_rate_is_identity():
    sig = np.arange(10.0)
    assert np.array_equal(resample_to_fs(sig, 130, 130), sig)


def test_segment_overlap_50_percent():
    sig = np.arange(650 * 3)
    w = segment(sig, 650, 0.5)
    assert w.shape == (5, 650)            # pasos de 325: 0, 325, 650, 975, 1300
    assert w[1, 0] == 325


def test_segment_short_signal_gives_zero_windows():
    assert segment(np.zeros(100), 650, 0.5).shape == (0, 650)


def test_zscore_windows():
    w = np.random.default_rng(0).normal(5, 3, size=(4, 650))
    z = zscore_windows(w)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_signal_to_windows_shape_dtype():
    cfg = Config()
    sig = np.random.default_rng(0).normal(size=130 * 20)
    w = signal_to_windows(sig, 130, cfg)
    assert w.ndim == 3 and w.shape[1:] == (1, 650)
    assert w.dtype == np.float32


# --- Review Focus 3: grabaciones problemáticas ---
def test_flat_segment_windows_are_dropped_not_nan():
    cfg = Config()
    rng = np.random.default_rng(1)
    good = rng.normal(size=130 * 10)
    flat = np.zeros(130 * 10)                 # banda desconectada: línea plana
    w = signal_to_windows(np.concatenate([good, flat]), 130, cfg)
    assert np.isfinite(w).all()
    assert len(w) < len(segment(np.zeros(130 * 20), 650, 0.5))


def test_nan_in_recording_raises():
    cfg = Config()
    sig = np.random.default_rng(2).normal(size=130 * 10)
    sig[100] = np.nan
    with pytest.raises(DataError, match="NaN"):
        signal_to_windows(sig, 130, cfg)


def test_recording_shorter_than_one_window_gives_zero_windows():
    cfg = Config()
    sig = np.random.default_rng(3).normal(size=130 * 2)   # 2 s < 5 s
    w = signal_to_windows(sig, 130, cfg)
    assert w.shape == (0, 1, 650)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_windowing.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.windowing'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Preprocesamiento mínimo: filtro, remuestreo, ventanas y z-score.

Es un reemplazo sencillo hasta que exista el módulo completo de preprocesamiento (con
detección de picos R y calidad de señal). Sirve para ECG-ID y para grabaciones del Polar.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.config import Config
from src.errors import DataError


def bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Pasa-banda de fase cero. Quita deriva de línea base (<0.5 Hz) y ruido muscular (>40 Hz)."""
    if high >= fs / 2:
        raise DataError(f"La frecuencia de corte {high} Hz excede Nyquist ({fs / 2} Hz)")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def resample_to_fs(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(sig, ratio.numerator, ratio.denominator)


def segment(sig: np.ndarray, window_samples: int, overlap: float) -> np.ndarray:
    step = max(1, int(round(window_samples * (1 - overlap))))
    starts = range(0, len(sig) - window_samples + 1, step)
    if len(starts) == 0:
        return np.zeros((0, window_samples))
    return np.stack([sig[s : s + window_samples] for s in starts])


def zscore_windows(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Media 0 y desviación 1 por ventana: la red ve la FORMA, no la amplitud absoluta."""
    return (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + eps)


def signal_to_windows(sig: np.ndarray, fs_in: float, cfg: Config) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    if not np.isfinite(sig).all():
        raise DataError("La grabación contiene NaN o inf; hay que limpiarla o recortarla antes")
    # Se filtra a la frecuencia ORIGINAL y luego se remuestrea (si no, el filtro anti-alias
    # de remuestrear descartaría banda útil o dejaría pasar ruido).
    raw = resample_to_fs(sig, fs_in, cfg.fs)
    filt = resample_to_fs(bandpass(sig, fs_in, cfg.band_low_hz, cfg.band_high_hz), fs_in, cfg.fs)
    w = segment(filt, cfg.window_samples, cfg.window_overlap)
    if len(w):
        # Ventanas planas = banda desconectada; z-score las convertiría en ruido. Se detectan en la
        # señal CRUDA porque el filtro deja "ecos" que hacen que una zona plana no salga con std 0.
        # (La calidad de señal más fina corresponde al módulo de preprocesamiento futuro.)
        raw_w = segment(raw, cfg.window_samples, cfg.window_overlap)
        w = w[raw_w.std(axis=1) > 1e-6]
    w = zscore_windows(w) if len(w) else w
    return w[:, None, :].astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_windowing.py -v`
Expected: 10 passed.

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: minimal windowing (bandpass, resample, segment, z-score)`

---

### Task 3: Splits by person and leakage guards

**Files:**
- Modify: `src/data.py` (append)
- Test: `tests/test_data.py` (append)

**Interfaces:**
- Consumes: `WindowTable`, `DataError`, `LeakageError`.
- Produces:
  - `PolarSplit(train: WindowTable, val: WindowTable, holdout: WindowTable)`
  - `split_polar(table, holdout_ids) -> PolarSplit` (hold-out = all windows of those people; for every other person, **last session = validation**, earlier sessions = train)
  - `split_public(table, val_fraction, seed) -> tuple[WindowTable, WindowTable]` (by person)
  - `assert_no_holdout_leak(holdout_ids, tables: dict[str, WindowTable]) -> None`
  - `assert_disjoint_people(tables: dict[str, WindowTable]) -> None`

- [ ] **Step 1: Write the failing tests (append to `tests/test_data.py`)**

```python
from src.data import (
    PolarSplit,
    assert_disjoint_people,
    assert_no_holdout_leak,
    split_polar,
    split_public,
)
from src.errors import LeakageError


def _polar(n_people=5, n_sessions=3):
    return make_synthetic_table(n_people=n_people, n_sessions=n_sessions, windows_per_session=4, seed=11)


def test_split_polar_holdout_people_are_complete_and_separate():
    t = _polar()
    s = split_polar(t, holdout_ids=("p3", "p4"))
    assert isinstance(s, PolarSplit)
    assert s.holdout.persons() == ["p3", "p4"]
    assert set(s.train.persons()) == {"p0", "p1", "p2"}
    assert set(s.val.persons()) == {"p0", "p1", "p2"}
    assert len(s.holdout) == 2 * 3 * 4          # todas las sesiones de los hold-out


def test_split_polar_last_session_is_validation():
    s = split_polar(_polar(), holdout_ids=("p3", "p4"))
    assert set(s.val.session_ids.tolist()) == {"s3"}
    assert set(s.train.session_ids.tolist()) == {"s1", "s2"}


def test_split_polar_never_loses_or_duplicates_windows():
    t = _polar()
    s = split_polar(t, holdout_ids=("p3", "p4"))
    assert len(s.train) + len(s.val) + len(s.holdout) == len(t)


# --- Review Focus 1: una sola sesión ---
def test_person_with_single_session_raises_and_names_person():
    t = _polar()
    keep = ~((t.participant_ids == "p1") & (t.session_ids != "s1"))
    t = t.subset(keep)
    with pytest.raises(DataError, match="p1"):
        split_polar(t, holdout_ids=("p3", "p4"))


# --- Review Focus 2: hold-out inexistente (typo) ---
def test_unknown_holdout_id_raises():
    with pytest.raises(DataError, match="P7"):
        split_polar(_polar(), holdout_ids=("p3", "P7"))


def test_assert_no_holdout_leak_detects_leak():
    t = _polar()
    with pytest.raises(LeakageError, match="p4"):
        assert_no_holdout_leak(("p4",), {"train": t})


def test_assert_no_holdout_leak_passes_when_clean():
    s = split_polar(_polar(), holdout_ids=("p3", "p4"))
    assert_no_holdout_leak(("p3", "p4"), {"train": s.train, "val": s.val})


def test_assert_disjoint_people():
    a = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="x")
    b = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="x")
    with pytest.raises(LeakageError):
        assert_disjoint_people({"a": a, "b": b})
    c = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=2, prefix="y")
    assert_disjoint_people({"a": a, "c": c})


def test_split_public_is_by_person_and_deterministic():
    t = make_synthetic_table(n_people=10, n_sessions=1, windows_per_session=3, prefix="pub_")
    tr, va = split_public(t, val_fraction=0.2, seed=0)
    assert set(tr.persons()).isdisjoint(va.persons())
    assert len(va.persons()) == 2
    tr2, va2 = split_public(t, val_fraction=0.2, seed=0)
    assert va.persons() == va2.persons()


def test_split_public_needs_enough_people():
    t = make_synthetic_table(n_people=2, n_sessions=1, windows_per_session=3, prefix="pub_")
    with pytest.raises(DataError):
        split_public(t, val_fraction=0.5, seed=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL, `ImportError: cannot import name 'PolarSplit'`.

- [ ] **Step 3: Append the implementation to `src/data.py`**

Also add `from src.errors import DataError, LeakageError` (replace the existing errors import).

```python
@dataclass
class PolarSplit:
    train: WindowTable
    val: WindowTable
    holdout: WindowTable


def assert_no_holdout_leak(holdout_ids, tables: dict) -> None:
    """Falla si una persona reservada aparece en cualquier tabla de entrenamiento/validación."""
    for name, t in tables.items():
        bad = set(t.persons()) & set(holdout_ids)
        if bad:
            raise LeakageError(
                f"Participantes hold-out {sorted(bad)} aparecen en '{name}'. "
                "Eso invalida la evaluación con impostores no vistos."
            )


def assert_disjoint_people(tables: dict) -> None:
    """Ninguna persona puede estar en dos tablas (p. ej. público vs Polar)."""
    seen: dict[str, str] = {}
    for name, t in tables.items():
        for p in t.persons():
            if p in seen:
                raise LeakageError(f"La persona '{p}' está en '{seen[p]}' y en '{name}'")
            seen[p] = name


def split_polar(table: WindowTable, holdout_ids) -> PolarSplit:
    """Hold-out por persona; para el resto, la ÚLTIMA sesión es validación.

    Validar con una sesión posterior (otro día) mide lo que importa: que la huella
    sobreviva a cambios de colocación de la banda y de estado fisiológico.
    """
    present = set(table.persons())
    missing = [h for h in holdout_ids if h not in present]
    if missing:
        raise DataError(
            f"Los participantes hold-out {missing} no existen en los datos "
            f"(personas disponibles: {sorted(present)}). ¿Error de escritura en HOLDOUT_IDS?"
        )
    is_hold = np.isin(table.participant_ids, list(holdout_ids))
    holdout, rest = table.subset(is_hold), table.subset(~is_hold)

    single = [p for p in rest.persons() if len(rest.sessions_of(p)) < 2]
    if single:
        raise DataError(
            f"Las personas {single} tienen una sola sesión. Se necesitan al menos 2 sesiones "
            "en días distintos por persona (la última se usa como validación)."
        )
    train_mask = np.zeros(len(rest), dtype=bool)
    for p in rest.persons():
        last = rest.sessions_of(p)[-1]
        train_mask |= (rest.participant_ids == p) & (rest.session_ids != last)
    split = PolarSplit(rest.subset(train_mask), rest.subset(~train_mask), holdout)
    assert_no_holdout_leak(holdout_ids, {"train": split.train, "val": split.val})
    return split


def split_public(table: WindowTable, val_fraction: float, seed: int):
    """Separa personas (no ventanas) del dataset público para validar el preentrenamiento."""
    persons = table.persons()
    n_val = max(1, int(round(len(persons) * val_fraction)))
    if len(persons) - n_val < 2:
        raise DataError("Muy pocas personas públicas para separar entrenamiento y validación")
    rng = np.random.default_rng(seed)
    val_people = rng.choice(persons, size=n_val, replace=False).tolist()
    mask = np.isin(table.participant_ids, val_people)
    return table.subset(~mask), table.subset(mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: all passed (7 earlier + 10 new = 17).

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: person-level splits and hold-out leakage guards`

---

### Task 4: Augmentation and torch `Dataset`

**Files:**
- Modify: `src/data.py` (append)
- Test: `tests/test_data.py` (append)

**Interfaces:**
- Consumes: `Config` (`jitter_std`, `magnitude_warp`, `warp_strength`, `wander_amplitude`, `fs`), `WindowTable`.
- Produces:
  - `Augmenter(cfg, seed=0)`; `__call__(window: ndarray (1,650)) -> ndarray (1,650) float32`, output re-z-scored.
  - `make_label_map(table) -> dict[str, int]` (sorted people → 0..K-1)
  - `WindowDataset(table, label_map, augmenter=None)`: items `(torch.FloatTensor (1,650), int)`
  - `make_loader(dataset, batch_size, shuffle, seed) -> DataLoader`

The spec's "amplitude scaling" is implemented as **magnitude warping** (a smooth random gain curve over time, as in the brief). A single global gain would be undone by the final z-score and have no effect.

- [ ] **Step 1: Write the failing tests (append)**

```python
import dataclasses

import torch

from src.config import Config
from src.data import Augmenter, WindowDataset, make_label_map, make_loader


def _window():
    return make_synthetic_table(n_people=1, n_sessions=1, windows_per_session=1, seed=4).windows[0]


def test_augmenter_output_shape_dtype_and_zscore():
    out = Augmenter(Config(), seed=0)(_window())
    assert out.shape == (1, 650) and out.dtype == np.float32
    assert abs(out.mean()) < 1e-3 and out.std() == pytest.approx(1.0, abs=1e-2)


def test_augmenter_with_all_strengths_zero_is_identity():
    cfg = dataclasses.replace(
        Config(), jitter_std=0, magnitude_warp=0, warp_strength=0, wander_amplitude=0
    )
    w = _window()
    assert np.allclose(Augmenter(cfg, seed=0)(w), w, atol=1e-4)


def test_augmenter_changes_the_window_and_is_seeded():
    w = _window()
    a = Augmenter(Config(), seed=1)(w)
    b = Augmenter(Config(), seed=1)(w)
    c = Augmenter(Config(), seed=2)(w)
    assert not np.allclose(a, w, atol=1e-3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_label_map_is_sorted_and_contiguous():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=2)
    assert make_label_map(t) == {"p0": 0, "p1": 1, "p2": 2}


def test_dataset_items_and_loader_batches():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=4)
    lm = make_label_map(t)
    ds = WindowDataset(t, lm, augmenter=Augmenter(Config(), seed=0))
    x, y = ds[0]
    assert x.shape == (1, 650) and x.dtype == torch.float32 and y == 0
    xb, yb = next(iter(make_loader(ds, batch_size=5, shuffle=True, seed=0)))
    assert xb.shape == (5, 1, 650) and yb.shape == (5,)


def test_loader_shuffle_is_reproducible():
    t = make_synthetic_table(n_people=3, n_sessions=1, windows_per_session=4)
    ds = WindowDataset(t, make_label_map(t))
    a = next(iter(make_loader(ds, 12, True, seed=3)))[1]
    b = next(iter(make_loader(ds, 12, True, seed=3)))[1]
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL, `ImportError: cannot import name 'Augmenter'`.

- [ ] **Step 3: Append the implementation to `src/data.py`**

Add at the top of the file: `import torch` and `from torch.utils.data import DataLoader, Dataset`, `from src.config import Config`.

```python
class Augmenter:
    """Aumento de datos SOLO para entrenamiento (nunca en validación ni en evaluación).

    La idea: con pocas personas la red tiende a memorizar. Variar la señal de forma
    realista (otra colocación de banda, respiración, ruido) le enseña a fijarse en la
    forma del latido y no en detalles accidentales.
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _smooth_curve(self, n: int, strength: float, knots: int = 4) -> np.ndarray:
        """Curva suave alrededor de 1: pocos puntos aleatorios interpolados linealmente."""
        pos = np.linspace(0, n - 1, knots)
        vals = 1 + self.rng.uniform(-strength, strength, knots)
        return np.interp(np.arange(n), pos, vals)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = window[0].astype(np.float64)
        n = x.shape[0]
        idx = np.arange(n)
        if c.warp_strength > 0:
            # Deformación temporal: el ritmo "se acelera o frena" ligeramente.
            speed = self._smooth_curve(n, c.warp_strength)
            t = np.cumsum(speed)
            t = (t - t[0]) / (t[-1] - t[0]) * (n - 1)
            x = np.interp(t, idx, x)
        if c.magnitude_warp > 0:
            # Ganancia que varía suavemente (la ganancia global no serviría: el z-score la anula).
            x = x * self._smooth_curve(n, c.magnitude_warp)
        if c.wander_amplitude > 0:
            # Deriva de línea base por respiración / movimiento (0.1 a 0.5 Hz).
            f = self.rng.uniform(0.1, 0.5)
            phase = self.rng.uniform(0, 2 * np.pi)
            x = x + c.wander_amplitude * np.sin(2 * np.pi * f * idx / c.fs + phase)
        if c.jitter_std > 0:
            x = x + self.rng.normal(0, c.jitter_std, n)
        x = (x - x.mean()) / (x.std() + 1e-8)   # se conserva el contrato: ventana z-scored
        return x[None, :].astype(np.float32)


def make_label_map(table: WindowTable) -> dict:
    return {p: i for i, p in enumerate(table.persons())}


class WindowDataset(Dataset):
    def __init__(self, table: WindowTable, label_map: dict, augmenter: Augmenter | None = None):
        self.table = table
        self.augmenter = augmenter
        self.labels = np.array([label_map[p] for p in table.participant_ids], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int):
        w = self.table.windows[i]
        if self.augmenter is not None:
            w = self.augmenter(w)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), int(self.labels[i])


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)   # el orden de los lotes también es reproducible
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: all passed (17 earlier + 6 new = 23).

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: training-only augmentation and torch Dataset/DataLoader`

---

### Task 5: The network (`model.py`)

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Config` (`stem_channels`, `stage_channels`, `kernel_size`, `embedding_dim`), `CheckpointError`.
- Produces `ECGEmbeddingNet(cfg, n_classes=None)`:
  - `forward(x: (B,1,650)) -> (B, embedding_dim)` L2-normalized
  - `embed_raw(x) -> (B, embedding_dim)` not normalized (what the head sees)
  - `logits(x) -> (B, n_classes)`; raises `RuntimeError` if no head
  - `set_head(n_classes)`, `encoder_state_dict()` (no `head.*`), `load_encoder_state_dict(sd)` (raises `CheckpointError` on anything other than missing `head.*`)
  - `freeze_early_layers()` (stem + all stages except the last; their BatchNorm stays in eval mode), `unfreeze_all()`
  - `n_parameters() -> int`

Note on size: with the spec's layout (`k=5`, channels 32→64→128→128, 1×1 conv shortcuts) the network has about 370k parameters. The spec estimated "roughly 200–300k"; the count is still tiny for CPU training, so the test accepts 150k–450k and the number is printed in the notebook.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.model'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: 11 passed. If `test_parameter_count_is_small` fails, print `ECGEmbeddingNet(Config()).n_parameters()`; adjust the assertion bounds only if the count is genuinely reasonable (under ~500k) and tell the user.

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: residual 1D-CNN embedding network`

---

### Task 6: Metrics (EER, ROC, AUC, FAR/FRR)

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `EvaluationError`.
- Produces:
  - `eer(genuine: ndarray, impostor: ndarray) -> tuple[float, float]` → `(eer, threshold)`
  - `far_frr_at(genuine, impostor, threshold) -> tuple[float, float]`
  - `auc_score(genuine, impostor) -> float`
  - `roc_points(genuine, impostor) -> tuple[ndarray, ndarray]` → `(far, frr)`
  - Convention: **score ≥ threshold ⇒ accepted**. FAR = fraction of impostor scores ≥ threshold; FRR = fraction of genuine scores < threshold.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from src.errors import EvaluationError
from src.evaluate import auc_score, eer, far_frr_at, roc_points


def test_perfect_separation_gives_zero_eer():
    e, thr = eer(np.full(50, 0.9), np.full(50, 0.1))
    assert e == 0.0
    far, frr = far_frr_at(np.full(50, 0.9), np.full(50, 0.1), thr)
    assert far == 0.0 and frr == 0.0


def test_random_scores_give_eer_near_half():
    rng = np.random.default_rng(0)
    e, _ = eer(rng.uniform(size=5000), rng.uniform(size=5000))
    assert e == pytest.approx(0.5, abs=0.05)


def test_known_overlap_eer():
    # genuinos 5..14, impostores 0..9 (se traslapan en 5..9); cuenta hecha a mano:
    genuine = np.arange(5.0, 15.0)
    impostor = np.arange(0.0, 10.0)
    e, thr = eer(genuine, impostor)
    # umbral 7: FAR = impostores>=7 -> 3/10 ; FRR = genuinos<7 -> 2/10
    # umbral 8: FAR = 2/10 ; FRR = 3/10  => en ambos |FAR-FRR| = 0.1 y EER = (0.3+0.2)/2 = 0.25
    assert e == pytest.approx(0.25)
    assert thr in (7.0, 8.0)


def test_far_frr_at_threshold_convention():
    genuine = np.array([0.2, 0.6, 0.8])
    impostor = np.array([0.1, 0.5, 0.7])
    far, frr = far_frr_at(genuine, impostor, 0.5)
    assert far == pytest.approx(2 / 3)    # 0.5 y 0.7 aceptados (>=)
    assert frr == pytest.approx(1 / 3)    # solo 0.2 rechazado (<)


def test_auc_perfect_and_random():
    assert auc_score(np.full(10, 0.9), np.full(10, 0.1)) == 1.0
    rng = np.random.default_rng(1)
    assert auc_score(rng.uniform(size=3000), rng.uniform(size=3000)) == pytest.approx(0.5, abs=0.05)


def test_roc_points_shapes_and_range():
    far, frr = roc_points(np.array([0.9, 0.8, 0.4]), np.array([0.1, 0.5, 0.3]))
    assert len(far) == len(frr)
    assert far.min() >= 0 and far.max() <= 1 and frr.min() >= 0 and frr.max() <= 1


def test_empty_inputs_raise():
    with pytest.raises(EvaluationError):
        eer(np.array([]), np.array([0.1]))
    with pytest.raises(EvaluationError):
        eer(np.array([0.1]), np.array([]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Evaluación: EER, ROC, FAR/FRR, protocolos de pares, bootstrap por persona y reporte."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: 7 passed. (`test_known_overlap_eer` was computed by hand in the test comment; if it fails, recompute from the FAR/FRR definitions before touching the expected value, and do not just loosen the tolerance.)

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: EER, ROC, AUC and FAR/FRR metrics`

---

### Task 7: Templates, trials and enrollment/test protocols

**Files:**
- Modify: `src/evaluate.py` (append)
- Test: `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: `WindowTable`, `ECGEmbeddingNet`, `eer`.
- Produces:
  - `l2_normalize(x: ndarray (N,D)) -> ndarray`
  - `make_model_embed_fn(model, batch_size=256) -> Callable[[ndarray (N,1,650)], ndarray (N,D)]`
  - `raw_embed_fn(windows: ndarray) -> ndarray` (flatten + L2; the "raw signal" comparison)
  - `Trials(genuine, genuine_owner, impostor, impostor_owner)` with `.n_owners() -> int`
  - `build_trials(enroll_emb, enroll_ids, test_emb, test_ids) -> Trials`
  - `first_session_table(table) -> WindowTable`
  - `split_enroll_test(table, mode: "cross_session"|"same_session") -> tuple[WindowTable, WindowTable]`
  - `run_protocol(embed_fn, enroll_table, test_table) -> Trials`

- [ ] **Step 1: Write the failing tests (append)**

```python
import torch

from src.config import Config
from src.data import WindowTable
from src.evaluate import (
    Trials,
    build_trials,
    first_session_table,
    l2_normalize,
    make_model_embed_fn,
    raw_embed_fn,
    run_protocol,
    split_enroll_test,
)
from src.model import ECGEmbeddingNet
from src.synthetic import make_synthetic_table


def test_l2_normalize_rows_have_unit_norm():
    x = np.random.default_rng(0).normal(size=(5, 8))
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1)


def test_build_trials_templates_and_labels():
    # Dos personas con embeddings 2-D perfectamente separados.
    enroll = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
    enroll_ids = np.array(["a", "a", "b", "b"])
    test = np.array([[1, 0.05], [0.05, 1]], dtype=float)
    test_ids = np.array(["a", "b"])
    t = build_trials(enroll, enroll_ids, test, test_ids)
    assert isinstance(t, Trials)
    assert len(t.genuine) == 2 and len(t.impostor) == 2
    assert t.genuine.min() > t.impostor.max()
    assert set(t.genuine_owner.tolist()) == {"a", "b"}
    assert t.n_owners() == 2


def test_unenrolled_test_person_counts_only_as_impostor():
    enroll = np.array([[1.0, 0.0]])
    t = build_trials(enroll, np.array(["a"]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array(["a", "z"]))
    assert len(t.genuine) == 1 and len(t.impostor) == 1


def test_build_trials_without_impostors_raises():
    with pytest.raises(EvaluationError):
        build_trials(np.array([[1.0, 0.0]]), np.array(["a"]), np.array([[1.0, 0.0]]), np.array(["a"]))


def _table(n_people=3, n_sessions=3, w=6):
    return make_synthetic_table(n_people=n_people, n_sessions=n_sessions, windows_per_session=w, seed=21)


def test_first_session_table_keeps_only_first_session_per_person():
    ft = first_session_table(_table())
    assert set(ft.session_ids.tolist()) == {"s1"}
    assert len(ft) == 3 * 6


def test_split_enroll_test_cross_session():
    enroll, test = split_enroll_test(_table(), "cross_session")
    assert set(enroll.session_ids.tolist()) == {"s1"}
    assert "s1" not in set(test.session_ids.tolist())
    assert len(enroll) + len(test) == len(_table())


def test_split_enroll_test_same_session_halves_first_session():
    enroll, test = split_enroll_test(_table(), "same_session")
    assert set(enroll.session_ids.tolist()) == {"s1"} == set(test.session_ids.tolist())
    assert len(enroll) == len(test) == 3 * 3     # mitad y mitad de las 6 ventanas de s1


def test_split_enroll_test_rejects_insufficient_data():
    t = _table(n_sessions=1)
    with pytest.raises(EvaluationError, match="cross_session"):
        split_enroll_test(t, "cross_session")


def test_split_enroll_test_unknown_mode():
    with pytest.raises(EvaluationError):
        split_enroll_test(_table(), "otra")


def test_run_protocol_with_model_and_raw_embeddings():
    t = _table()
    enroll, test = split_enroll_test(t, "cross_session")
    net = ECGEmbeddingNet(Config())
    trials = run_protocol(make_model_embed_fn(net), enroll, test)
    assert len(trials.genuine) > 0 and len(trials.impostor) > 0
    raw = run_protocol(raw_embed_fn, enroll, test)
    assert len(raw.genuine) == len(trials.genuine)


def test_model_embed_fn_leaves_no_gradients_and_returns_unit_vectors():
    net = ECGEmbeddingNet(Config())
    x = np.random.default_rng(0).normal(size=(5, 1, 650)).astype(np.float32)
    z = make_model_embed_fn(net)(x)
    assert z.shape == (5, 128) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    assert all(p.grad is None for p in net.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL, `ImportError: cannot import name 'Trials'`.

- [ ] **Step 3: Append the implementation to `src/evaluate.py`**

Add imports at the top: `from dataclasses import dataclass`, `import torch`, `from src.data import WindowTable`.

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: all passed (7 earlier + 11 new = 18).

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: templates, trials and enrollment/test protocols`

---

### Task 8: Bootstrap by person, report rows and model comparison

**Files:**
- Modify: `src/evaluate.py` (append)
- Test: `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: `Trials`, `eer`, `far_frr_at`, `auc_score`, `PolarSplit`, `run_protocol`, `split_enroll_test`, `first_session_table`, `Config.bootstrap_samples`, `Config.seed`.
- Produces:
  - `bootstrap_eer_ci(trials, n_boot, seed, alpha=0.05) -> tuple[float, float]`; `(nan, nan)` when fewer than 3 template owners.
  - `ReportRow(label, eer, ci_lo, ci_hi, auc, far, frr, n_genuine, n_impostor, n_owners)`
  - `make_row(label, trials, threshold, n_boot, seed) -> ReportRow`
  - `format_report(rows: list[ReportRow]) -> str` (Markdown table)
  - `compare_models(embed_fns: dict[str, Callable], split: PolarSplit, cfg: Config) -> str` (Markdown). For each model: **seen/cross-session (validation)**, **unseen/hold-out cross-session**, **seen/same-session**; the threshold of each model comes from its **validation** EER.

- [ ] **Step 1: Write the failing tests (append)**

```python
import dataclasses

from src.data import split_polar
from src.evaluate import (
    ReportRow,
    bootstrap_eer_ci,
    compare_models,
    format_report,
    make_row,
)


def _trials(n_owners=4, sep=True, per=30, seed=0):
    rng = np.random.default_rng(seed)
    g, go, i, io = [], [], [], []
    for k in range(n_owners):
        g.append(rng.normal(0.8 if sep else 0.5, 0.05, per)); go += [f"o{k}"] * per
        i.append(rng.normal(0.2 if sep else 0.5, 0.05, per)); io += [f"o{k}"] * per
    return Trials(np.concatenate(g), np.array(go), np.concatenate(i), np.array(io))


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    t = _trials()
    point, _ = eer(t.genuine, t.impostor)
    lo, hi = bootstrap_eer_ci(t, n_boot=200, seed=0)
    assert lo - 1e-9 <= point <= hi + 1e-9
    assert (lo, hi) == bootstrap_eer_ci(t, n_boot=200, seed=0)


def test_bootstrap_ci_wider_when_scores_overlap():
    lo1, hi1 = bootstrap_eer_ci(_trials(sep=True), 200, 0)
    lo2, hi2 = bootstrap_eer_ci(_trials(sep=False), 200, 0)
    assert (hi2 - lo2) > (hi1 - lo1)


# --- Review Focus 4: menos de 3 personas ---
def test_bootstrap_ci_is_nan_with_fewer_than_three_people():
    lo, hi = bootstrap_eer_ci(_trials(n_owners=2), n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_make_row_fields_and_counts():
    t = _trials()
    thr = eer(t.genuine, t.impostor)[1]
    row = make_row("demo", t, thr, n_boot=50, seed=0)
    assert isinstance(row, ReportRow)
    assert row.n_genuine == len(t.genuine) and row.n_impostor == len(t.impostor)
    assert row.n_owners == 4 and 0 <= row.eer <= 1 and 0 <= row.auc <= 1


def test_format_report_prints_counts_and_na_ci():
    rows = [
        make_row("tres personas", _trials(), 0.5, 50, 0),
        make_row("dos personas", _trials(n_owners=2), 0.5, 50, 0),
    ]
    md = format_report(rows)
    assert md.startswith("|") and "tres personas" in md and "dos personas" in md
    assert "n/a" in md                       # IC no disponible con <3 personas
    assert f"{rows[0].n_genuine}/{rows[0].n_impostor}" in md


def test_compare_models_produces_all_sections_for_each_model():
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=31)
    split = split_polar(t, ("p4", "p5"))
    cfg = dataclasses.replace(Config(), bootstrap_samples=20)
    fns = {"red": make_model_embed_fn(ECGEmbeddingNet(cfg)), "cruda": raw_embed_fn}
    md = compare_models(fns, split, cfg)
    for model in ("red", "cruda"):
        assert f"{model} · vistos, entre sesiones (val)" in md
        assert f"{model} · NO vistos (hold-out), entre sesiones" in md
        assert f"{model} · vistos, misma sesión" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL, `ImportError: cannot import name 'ReportRow'`.

- [ ] **Step 3: Append the implementation to `src/evaluate.py`**

Add imports at the top: `from src.config import Config`, `from src.data import PolarSplit, WindowTable` (extend the existing `WindowTable` import).

```python
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
    same_enroll, same_test = split_enroll_test(split.train, "same_session")
    rows = []
    for name, fn in embed_fns.items():
        val = run_protocol(fn, enroll_val, split.val)
        thr = eer(val.genuine, val.impostor)[1]
        n, s = cfg.bootstrap_samples, cfg.seed
        rows.append(make_row(f"{name} · vistos, entre sesiones (val)", val, thr, n, s))
        rows.append(make_row(f"{name} · NO vistos (hold-out), entre sesiones",
                             run_protocol(fn, hold_enroll, hold_test), thr, n, s))
        rows.append(make_row(f"{name} · vistos, misma sesión",
                             run_protocol(fn, same_enroll, same_test), thr, n, s))
    return format_report(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: all passed (18 earlier + 6 new = 24).

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: person-level bootstrap CI, report rows and model comparison`

---

### Task 9: Versioned checkpoints with JSON sidecar

**Files:**
- Create: `src/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `ECGEmbeddingNet`, `Config`, `CheckpointError`.
- Produces:
  - `REQUIRED_SIDECAR_FIELDS: tuple[str, ...]` = `("version","mode","created_utc","seed","holdout_ids","data_fingerprint","config","metrics","parent")`
  - `next_version(models_dir) -> int`
  - `save_checkpoint(model, cfg, models_dir, mode, metrics, data_fingerprint, parent=None) -> Path` (writes `modelo_vN.pt` with **encoder weights only** + `modelo_vN.json`)
  - `load_checkpoint(path, cfg) -> ECGEmbeddingNet` (eval mode, **no head**; raises `CheckpointError` if missing or incompatible, e.g. different `embedding_dim`)
  - `promote(models_dir, version) -> Path` (copies `modelo_vN.pt`/`.json` to `modelo_ecg.pt`/`.json`; copy, not symlink, because Windows symlinks need admin rights)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checkpoint.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_checkpoint.py -v`
Expected: 8 passed.

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: versioned checkpoints with JSON sidecar and promote`

---

### Task 10: Training loop (`fit`)

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `ECGEmbeddingNet`, `Config`, `eer`, `run_protocol`, `make_model_embed_fn`.
- Produces:
  - `FitResult(best_val_eer: float, history: list[dict], epochs_run: int)`; each history item is `{"epoch", "train_loss", "val_eer"}`.
  - `fit(model, train_loader, val_fn, cfg, epochs, lr) -> FitResult` — AdamW over the **trainable** parameters, cosine LR, cross-entropy with label smoothing, **early stopping on `val_fn(model)` (an EER)**, restores the best weights at the end. Raises `ValueError` if `epochs < 1`.
  - `make_val_fn(enroll_table, test_table) -> Callable[[model], float]` — returns validation EER.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Entrenamiento: bucle genérico + (en tareas siguientes) los tres modos y la CLI."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.config import Config
from src.evaluate import eer, make_model_embed_fn, run_protocol


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


def fit(model, train_loader, val_fn, cfg: Config, epochs: int, lr: float) -> FitResult:
    if epochs < 1:
        raise ValueError("epochs debe ser >= 1")
    # Solo los parámetros entrenables: en la etapa A del fine-tuning las capas tempranas están congeladas.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best, best_state, bad, history = float("inf"), None, 0, []
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: 6 passed. If `test_training_loss_decreases_on_separable_synthetic_people` is flaky, raise `epochs` to 8 or `lr` to 3e-3 in that test; do not weaken the assertion.

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: training loop with early stopping on validation EER`

---

### Task 11: Three training modes and CLI

**Files:**
- Modify: `src/train.py` (append)
- Test: `tests/test_train.py` (append)

**Interfaces:**
- Consumes: `fit`, `make_val_fn`, `save_checkpoint`, `load_checkpoint`, `promote`, `split_polar`, `split_public`, `PolarSplit`, `WindowDataset`, `Augmenter`, `make_label_map`, `make_loader`, `table_fingerprint`, `assert_no_holdout_leak`, `load_window_dir`, `first_session_table`, `split_enroll_test`.
- Produces:
  - `pretrain(cfg, public_train, public_val, models_dir) -> Path`
  - `finetune(cfg, split: PolarSplit, pretrained_ckpt, models_dir) -> Path` (loads encoder, **new head** sized for Polar train people, stage A frozen early layers, stage B everything at lower lr)
  - `polar_only(cfg, split: PolarSplit, models_dir) -> Path`
  - `require_holdout(cfg) -> None` (raises `LeakageError` if `len(cfg.holdout_ids) < cfg.min_holdout_people`)
  - `main(argv=None) -> int` with subcommands `pretrain --public-dir`, `finetune --polar-dir --from-checkpoint`, `polar-only --polar-dir`, `promote --version`. Sidecar `metrics` contains `best_val_eer` and `history` (finetune: `stage_a` and `stage_b`).

- [ ] **Step 1: Write the failing tests (append)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL, `ImportError: cannot import name 'finetune'`.

- [ ] **Step 3: Append the implementation to `src/train.py`**

Add these imports at the top of the file:
```python
import argparse
from pathlib import Path

from src.checkpoint import load_checkpoint, promote, save_checkpoint
from src.config import set_seed
from src.data import (
    Augmenter, PolarSplit, WindowDataset, assert_no_holdout_leak, load_window_dir,
    make_label_map, make_loader, split_polar, split_public, table_fingerprint,
)
from src.errors import LeakageError
from src.evaluate import first_session_table, split_enroll_test
from src.model import ECGEmbeddingNet
```

Then append:
```python
def require_holdout(cfg: Config) -> None:
    """Sin hold-out no hay forma de medir impostores no vistos: mejor negarse a entrenar."""
    if len(cfg.holdout_ids) < cfg.min_holdout_people:
        raise LeakageError(
            f"Hay que definir al menos {cfg.min_holdout_people} participantes hold-out en "
            f"HOLDOUT_IDS (src/config.py); hay {len(cfg.holdout_ids)}."
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
    set_seed(cfg.seed)
    model = load_checkpoint(pretrained_ckpt, cfg)           # solo encoder, sin cabeza
    lm = make_label_map(split.train)
    model.set_head(len(lm))                                  # cabeza nueva para nuestras personas
    loader = _loader(cfg, split.train, lm)
    val_fn = make_val_fn(first_session_table(split.train), split.val)

    model.freeze_early_layers()
    stage_a = fit(model, loader, val_fn, cfg, cfg.finetune_head_epochs, cfg.finetune_head_lr)
    model.unfreeze_all()
    stage_b = fit(model, loader, val_fn, cfg, cfg.finetune_full_epochs, cfg.finetune_full_lr)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: all passed (6 earlier + 8 new = 14). The whole file should run in well under a minute.

- [ ] **Step 5: Run the full suite so far**

Run: `pytest -v`
Expected: all tests from Tasks 0–11 pass.

- [ ] **Step 6: Checkpoint**

Propose commit: `feat: pretrain / finetune / polar-only training modes and CLI`

---

### Task 12: Public dataset (ECG-ID) download and conversion

**Files:**
- Create: `src/public_data.py`
- Test: `tests/test_public_data.py`

**Interfaces:**
- Consumes: `signal_to_windows`, `save_window_file`, `load_window_dir`, `Config`.
- Produces:
  - `iter_ecgid_records(root) -> Iterator[tuple[str, str, ndarray, float]]` → `(person_dir_name, record_name, signal, fs)` (uses `wfdb`, imported lazily; channel 0 = raw lead I)
  - `build_public_windows(root, out_dir, cfg) -> int` → number of `.npz` written; ids are `pub_<Person_NN>`, session ids `rec_<NN zero-padded>`, activity `rest`, source `public`
  - `download_ecgid(dest) -> None`
  - CLI: `python -m src.public_data download --dest data/raw/public/ecgid` and `python -m src.public_data build --raw data/raw/public/ecgid --out data/processed/public`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

import src.public_data as pd_
from src.config import Config
from src.data import load_window_dir


def _fake_records():
    rng = np.random.default_rng(0)
    fs = 500.0
    t = np.arange(0, 20, 1 / fs)
    for person in ("Person_01", "Person_02"):
        for rec in ("rec_1", "rec_2", "rec_10"):
            beat = np.sin(2 * np.pi * 1.2 * t) ** 21           # picos tipo R a ~72 lpm
            yield person, rec, beat + rng.normal(0, 0.02, len(t)), fs


def test_build_public_windows_writes_loadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: _fake_records())
    n = pd_.build_public_windows("ignored", tmp_path, Config())
    assert n == 6
    t = load_window_dir(tmp_path)
    assert t.persons() == ["pub_Person_01", "pub_Person_02"]
    assert t.source == "public"
    assert t.windows.shape[1:] == (1, 650)
    # 20 s a 130 Hz con ventanas de 5 s y 50 % de traslape -> 7 ventanas por registro
    assert len(t) == 6 * 7


def test_session_ids_are_zero_padded_so_they_sort_chronologically(tmp_path, monkeypatch):
    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: _fake_records())
    pd_.build_public_windows("ignored", tmp_path, Config())
    t = load_window_dir(tmp_path)
    assert t.sessions_of("pub_Person_01") == ["rec_01", "rec_02", "rec_10"]


def test_records_too_short_are_skipped(tmp_path, monkeypatch):
    def short():
        yield "Person_01", "rec_1", np.random.default_rng(0).normal(size=500 * 2), 500.0

    monkeypatch.setattr(pd_, "iter_ecgid_records", lambda root: short())
    assert pd_.build_public_windows("ignored", tmp_path, Config()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_public_data.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.public_data'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Dataset público ECG-ID (PhysioNet) -> ventanas de 5 s a 130 Hz, para preentrenar.

Se usa SOLO para preentrenar: son otras derivaciones y otro hardware que la banda Polar, por eso
después se hace fine-tuning con datos del Polar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.data import save_window_file
from src.windowing import signal_to_windows


def download_ecgid(dest) -> None:
    import wfdb  # import perezoso: solo hace falta para descargar/leer

    wfdb.dl_database("ecgiddb", dl_dir=str(dest))


def iter_ecgid_records(root):
    """Recorre Person_XX/rec_N. Canal 0 = ECG I crudo (el canal 1 ya viene filtrado por ellos)."""
    import wfdb

    for hea in sorted(Path(root).rglob("rec_*.hea")):
        rec = wfdb.rdrecord(str(hea.with_suffix("")))
        yield hea.parent.name, hea.stem, rec.p_signal[:, 0], float(rec.fs)


def build_public_windows(root, out_dir, cfg: Config) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for person, rec, sig, fs in iter_ecgid_records(root):
        windows = signal_to_windows(sig, fs, cfg)
        if len(windows) == 0:
            continue
        # Relleno con ceros para que "rec_10" ordene después de "rec_2" (orden cronológico).
        session = f"rec_{int(rec.split('_')[1]):02d}"
        pid = f"pub_{person}"      # prefijo: nunca choca con un participante Polar
        save_window_file(out_dir / f"{pid}__{session}.npz", windows, pid, session, "rest", "public")
        written += 1
    return written


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.public_data")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", default="data/raw/public/ecgid")
    b = sub.add_parser("build")
    b.add_argument("--raw", default="data/raw/public/ecgid")
    b.add_argument("--out", default=f"{cfg.data_dir}/public")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download_ecgid(a.dest)
    else:
        print(f"{build_public_windows(a.raw, a.out, cfg)} archivos escritos en {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_public_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Manual check with the real dataset (needs internet, about 500 MB or less)**

This step confirms the license and the real data, which the spec left open.

1. Open `https://physionet.org/content/ecgiddb/` and note the **license** and the **sampling rate** in the dataset page. Record both in `README.md` (Task 16). If the license does not allow your use (academic project), stop and tell the user.
2. Run:
```powershell
python -m src.public_data download --dest data/raw/public/ecgid
python -m src.public_data build --raw data/raw/public/ecgid --out data/processed/public
python -c "from src.data import load_window_dir; t = load_window_dir('data/processed/public'); print(len(t), 'ventanas,', len(t.persons()), 'personas')"
```
Expected: about 90 people and a few thousand windows. If the numbers are far off (for example 0 people), inspect the folder layout under `data/raw/public/ecgid` and adjust the `rec_*.hea` glob in `iter_ecgid_records`; report what you found.

- [ ] **Step 6: Checkpoint**

Propose commit: `feat: ECG-ID download and conversion to 130 Hz windows`

---

### Task 13: Comparison experiment CLI (Polar-only vs pretrained + fine-tuned)

**Files:**
- Modify: `src/evaluate.py` (append)
- Test: `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: `compare_models`, `load_checkpoint`, `make_model_embed_fn`, `raw_embed_fn`, `split_polar`, `load_window_dir`, `Config`.
- Produces: `main(argv=None) -> int` with `compare --polar-dir --polar-only CKPT --finetuned CKPT [--out reports/comparison.md]`. Prints the Markdown table, writes it to `--out`. It rebuilds the split from `Config.holdout_ids`, so it uses exactly the split the models were trained on; it refuses (exits non-zero with a message) if `holdout_ids` in either checkpoint's JSON differs from the current config.

- [ ] **Step 1: Write the failing test (append)**

```python
from src.checkpoint import save_checkpoint
from src.data import save_window_file
from src.evaluate import main as evaluate_main


def _write_polar_dir(tmp_path):
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=51)
    d = tmp_path / "polar"
    d.mkdir()
    for p in t.persons():
        for s in t.sessions_of(p):
            m = (t.participant_ids == p) & (t.session_ids == s)
            save_window_file(d / f"{p}__{s}.npz", t.windows[m], p, s, "rest", "synthetic")
    return d


def test_compare_cli_writes_report(tmp_path, monkeypatch):
    import src.evaluate as ev

    cfg = dataclasses.replace(Config(), holdout_ids=("p4", "p5"), bootstrap_samples=10)
    monkeypatch.setattr(ev, "Config", lambda: cfg)
    polar = _write_polar_dir(tmp_path)
    net = ECGEmbeddingNet(cfg, n_classes=3)
    a = save_checkpoint(net, cfg, tmp_path / "m", "polar-only", {}, "x")
    b = save_checkpoint(net, cfg, tmp_path / "m", "finetune", {}, "x", parent="modelo_v0.pt")
    out = tmp_path / "rep.md"
    rc = evaluate_main(["compare", "--polar-dir", str(polar), "--polar-only", str(a),
                        "--finetuned", str(b), "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Solo Polar" in md and "Preentrenado + fine-tuning" in md and "Señal cruda" in md


def test_compare_cli_refuses_when_holdout_differs_from_checkpoint(tmp_path, monkeypatch):
    import src.evaluate as ev

    trained = dataclasses.replace(Config(), holdout_ids=("p0", "p1"))
    current = dataclasses.replace(Config(), holdout_ids=("p4", "p5"), bootstrap_samples=10)
    monkeypatch.setattr(ev, "Config", lambda: current)
    polar = _write_polar_dir(tmp_path)
    net = ECGEmbeddingNet(trained, n_classes=3)
    a = save_checkpoint(net, trained, tmp_path / "m", "polar-only", {}, "x")
    b = save_checkpoint(net, trained, tmp_path / "m", "finetune", {}, "x")
    rc = evaluate_main(["compare", "--polar-dir", str(polar), "--polar-only", str(a),
                        "--finetuned", str(b), "--out", str(tmp_path / "r.md")])
    assert rc != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v -k compare_cli`
Expected: FAIL, `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Append the implementation to `src/evaluate.py`**

Add imports at the top: `import argparse`, `import json`, `from pathlib import Path`. The `Config` name must stay importable at module level (the tests monkeypatch `src.evaluate.Config`); imports of `load_window_dir`, `split_polar` and `load_checkpoint` go inside `main` to avoid a circular import (`checkpoint` imports `model`, not `evaluate`, so a top-level import of `load_checkpoint` would also work; keep it local for symmetry).

```python
def main(argv=None) -> int:
    from src.checkpoint import load_checkpoint
    from src.data import load_window_dir, split_polar

    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--polar-dir", default=f"{cfg.data_dir}/polar")
    c.add_argument("--polar-only", required=True, help="checkpoint entrenado solo con Polar")
    c.add_argument("--finetuned", required=True, help="checkpoint preentrenado + fine-tuning")
    c.add_argument("--out", default="reports/comparison.md")
    a = p.parse_args(argv)

    # Si el hold-out del checkpoint no es el actual, la comparación no sería sobre personas "no vistas".
    for ckpt in (a.polar_only, a.finetuned):
        side = json.loads(Path(ckpt).with_suffix(".json").read_text(encoding="utf-8"))
        if tuple(side["holdout_ids"]) != tuple(cfg.holdout_ids):
            print(f"ERROR: {Path(ckpt).name} se entrenó con hold-out {side['holdout_ids']} "
                  f"pero la configuración actual es {list(cfg.holdout_ids)}.")
            return 1

    split = split_polar(load_window_dir(a.polar_dir, cfg.window_samples), cfg.holdout_ids)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: all passed (24 earlier + 2 new = 26).

- [ ] **Step 5: Checkpoint**

Propose commit: `feat: Polar-only vs fine-tuned comparison CLI`

---

### Task 14: Notebook 02 (model and training)

**Files:**
- Create: `notebooks/build_notebooks.py` (generator), `notebooks/02_model_and_training.ipynb` (generated)

**Interfaces:**
- Consumes: everything above. Runs on **real Polar data if `data/processed/polar/*.npz` exists, otherwise on synthetic data**, and says which one at the top.
- Produces: an executable notebook with: what goes in (windows of three people), the architecture and parameter count, a Polar-only training run with loss curves, and the 2D projection (PCA and t-SNE) of embeddings of the validation and hold-out people.

The notebooks are written in Spanish (the team will defend in Spanish and the brief asks for Spanish explanations). Each cell is defined in `build_notebooks.py` so changes are readable in git diffs.

- [ ] **Step 1: Write the generator with notebook 02**

`notebooks/build_notebooks.py`:
```python
"""Genera los notebooks (las celdas viven aquí para poder revisar cambios con git diff).

Uso:  python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = '''import sys, tempfile, dataclasses
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import Config, set_seed
from src.data import load_window_dir, split_polar
from src.synthetic import make_synthetic_table

set_seed(42)
polar_dir = ROOT / "data" / "processed" / "polar"
if list(polar_dir.glob("*.npz")):
    USING = "datos REALES del Polar H10"
    cfg = Config()
    table = load_window_dir(polar_dir)
    models_dir = ROOT / cfg.models_dir
else:
    USING = "datos SINTÉTICOS (aún no hay grabaciones en data/processed/polar)"
    cfg = dataclasses.replace(Config(), holdout_ids=("p4", "p5"), batch_size=32, patience=3,
                              polar_only_epochs=8, bootstrap_samples=200)
    table = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=40, seed=0)
    models_dir = Path(tempfile.mkdtemp())      # no ensuciamos models/ con pruebas
split = split_polar(table, cfg.holdout_ids)
print("Usando:", USING)
print({"train": len(split.train), "val": len(split.val), "hold-out": len(split.holdout)}, "ventanas")'''


def nb02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 02 · El modelo y su entrenamiento\n\n"
            "**Qué entra:** una ventana de 5 s de ECG (650 muestras a 130 Hz), ya filtrada y con z-score.\n\n"
            "**Qué sale:** un vector de 128 números (la *huella*) de longitud 1. Dos ventanas de la misma "
            "persona deben dar vectores parecidos (coseno alto); de personas distintas, vectores distintos.\n\n"
            "Este notebook muestra cada pieza funcionando. El fine-tuning se explica en el notebook 03."
        ),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell(
            "## 1. Qué entra: ventanas de ECG\n"
            "Cada persona tiene una forma de latido distinta (en datos reales, la forma depende de la anatomía, "
            "la posición del corazón y la colocación de la banda). Aquí vemos una ventana de tres personas."
        ),
        nbf.v4.new_code_cell(
            '''fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
people = split.train.persons()[:3]
for ax, p in zip(axes, people):
    w = split.train.windows[split.train.participant_ids == p][0, 0]
    ax.plot(np.arange(len(w)) / cfg.fs, w)
    ax.set_ylabel(p)
axes[-1].set_xlabel("segundos")
fig.suptitle("Una ventana (z-score) de tres personas distintas")
plt.show()'''
        ),
        nbf.v4.new_markdown_cell(
            "## 2. La arquitectura\n"
            "Un *stem* (convolución inicial), cuatro etapas residuales que van reduciendo la longitud a la mitad y "
            "aumentando los canales (32→64→128→128), un promedio en el tiempo y una capa lineal que produce el "
            "embedding. La *cabeza de clasificación* solo existe para entrenar y se descarta."
        ),
        nbf.v4.new_code_cell(
            '''from src.model import ECGEmbeddingNet
net = ECGEmbeddingNet(cfg, n_classes=len(split.train.persons()))
print(f"Parámetros: {net.n_parameters():,}")
x = torch.zeros(1, 1, cfg.window_samples)
h = net.stem(x); print("stem      ->", tuple(h.shape))
for i, s in enumerate(net.stages):
    h = s(h); print(f"etapa {i + 1}   ->", tuple(h.shape))
print("embedding ->", tuple(net.eval()(x).shape), "| norma:", float(net(x).norm()))'''
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Entrenamiento (solo Polar, el modo de respaldo)\n"
            "Se entrena como clasificador de las personas de entrenamiento. La parada temprana mira la **EER de "
            "validación** (sesión posterior de esas mismas personas), no la exactitud."
        ),
        nbf.v4.new_code_cell(
            '''from src.train import polar_only
import json
ckpt = polar_only(cfg, split, models_dir)
side = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
hist = side["metrics"]["history"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist]); ax[0].set_title("Pérdida de entrenamiento")
ax[1].plot([h["epoch"] for h in hist], [h["val_eer"] for h in hist]); ax[1].set_title("EER de validación")
for a in ax: a.set_xlabel("época")
plt.show()
print("Checkpoint:", ckpt.name, "| mejor EER de validación:", round(side["metrics"]["best_val_eer"], 3))'''
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Cómo se ve la huella en 2D\n"
            "Cada punto es una ventana proyectada de 128 a 2 dimensiones. Si la red funciona, las ventanas de una "
            "persona forman un grupo. **Los puntos de hold-out son personas que la red nunca vio.**"
        ),
        nbf.v4.new_code_cell(
            '''from src.checkpoint import load_checkpoint
from src.evaluate import make_model_embed_fn
from src.data import concat_tables
trained = load_checkpoint(ckpt, cfg)
shown = concat_tables([split.val, split.holdout])
emb = make_model_embed_fn(trained)(shown.windows)
ids = shown.participant_ids
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for title, proj, a in (("PCA", PCA(2).fit_transform(emb), ax[0]),
                       ("t-SNE", TSNE(2, perplexity=min(30, len(emb) // 4), random_state=0).fit_transform(emb), ax[1])):
    for p in sorted(set(ids)):
        m = ids == p
        a.scatter(proj[m, 0], proj[m, 1], s=12, marker="x" if p in cfg.holdout_ids else "o",
                  label=p + (" (hold-out)" if p in cfg.holdout_ids else ""))
    a.set_title(title)
ax[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()'''
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Distribución de puntajes genuinos vs impostores (hold-out)\n"
            "Puntaje = similitud coseno con la plantilla (registro con la 1.ª sesión). Mientras menos se traslapen "
            "las dos distribuciones, mejor separa el sistema a personas no vistas."
        ),
        nbf.v4.new_code_cell(
            '''from src.evaluate import split_enroll_test, run_protocol, eer
enroll, test = split_enroll_test(split.holdout, "cross_session")
trials = run_protocol(make_model_embed_fn(trained), enroll, test)
e, thr = eer(trials.genuine, trials.impostor)
plt.hist(trials.impostor, bins=30, alpha=0.6, label="impostores")
plt.hist(trials.genuine, bins=30, alpha=0.6, label="genuinos")
plt.axvline(thr, color="k", ls="--", label=f"umbral EER={e:.2f}")
plt.legend(); plt.xlabel("similitud coseno"); plt.title("Hold-out (personas no vistas)"); plt.show()
print(f"Pares: {len(trials.genuine)} genuinos / {len(trials.impostor)} impostores de {trials.n_owners()} personas")'''
        ),
    ]
    return nb


NOTEBOOKS = {"02_model_and_training.ipynb": nb02}

if __name__ == "__main__":
    for name, build in NOTEBOOKS.items():
        nbf.write(build(), HERE / name)
        print("escrito", name)
```

- [ ] **Step 2: Generate and execute the notebook**

```powershell
python notebooks/build_notebooks.py
jupyter nbconvert --to notebook --execute notebooks/02_model_and_training.ipynb --output-dir "$env:TEMP\nb_check"
```
Expected: `[NbConvertApp] Writing ... bytes` with no exception. If a cell fails, fix the cell text in `build_notebooks.py` (not the `.ipynb`), regenerate and re-run. A kernel error `No module named 'torch'` means Jupyter is not using the venv: run `python -m ipykernel install --user --name ecg-auth` and pass `--ExecutePreprocessor.kernel_name=ecg-auth`.

- [ ] **Step 3: Visually check the output**

Open the executed notebook from `%TEMP%\nb_check` and confirm: the first output says whether it used real or synthetic data; the three-person plot shows distinct waveforms; the stage shapes end at `(1, 128)`; the loss curve goes down; the PCA/t-SNE plots render with hold-out people shown as `x`; the histogram shows both distributions and the pair counts line. Report anything that looks wrong rather than papering over it.

- [ ] **Step 4: Checkpoint**

Propose commit: `docs: notebook 02 (model and training) and notebook generator`

---

### Task 15: Notebook 03 (fine-tuning explained + Polar-only vs fine-tuned)

**Files:**
- Modify: `notebooks/build_notebooks.py` (add `nb03` and register it)
- Create: `notebooks/03_finetuning_explained.ipynb` (generated)

**Interfaces:**
- Consumes: `pretrain`, `finetune`, `polar_only`, `compare_models`, `make_model_embed_fn`, `raw_embed_fn`, `split_public`, `load_window_dir`.
- Produces: a plain-language explanation of fine-tuning (analogy + diagram), the experiment table (EER with confidence intervals and pair counts for Polar-only vs pretrained + fine-tuned vs raw signal), and 2D projections of hold-out embeddings before and after fine-tuning. Uses real ECG-ID + Polar data when both folders exist; otherwise synthetic stand-ins (a 20-person "public" set and the 6-person "Polar" set) and says so.

- [ ] **Step 1: Add `nb03` to `notebooks/build_notebooks.py`**

Insert before the `NOTEBOOKS = ...` line:
```python
SETUP03 = SETUP.replace(
    'split = split_polar(table, cfg.holdout_ids)',
    '''split = split_polar(table, cfg.holdout_ids)
from src.data import split_public
public_dir = ROOT / "data" / "processed" / "public"
if USING.startswith("datos REALES"):
    # Con Polar real NO se mezcla un público sintético: el resultado no significaría nada.
    if not list(public_dir.glob("*.npz")):
        raise FileNotFoundError(
            "Faltan los datos públicos. Ejecuta: python -m src.public_data download && "
            "python -m src.public_data build"
        )
    public = load_window_dir(public_dir)
    PUBLIC_USING = "ECG-ID real"
else:
    cfg = dataclasses.replace(cfg, pretrain_epochs=8, finetune_head_epochs=3, finetune_full_epochs=8)
    public = make_synthetic_table(n_people=20, n_sessions=2, windows_per_session=20, seed=7, prefix="pub_")
    PUBLIC_USING = "público SINTÉTICO (20 personas)"
public_train, public_val = split_public(public, cfg.pretrain_val_fraction, cfg.seed)
print("Preentrenamiento con:", PUBLIC_USING)''',
)


def nb03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 03 · Fine-tuning explicado y experimento Polar-solo vs preentrenado\n\n"
            "## ¿Qué es el fine-tuning, en palabras sencillas?\n\n"
            "Imagina que quieres reconocer a tus compañeros por su forma de caminar, pero solo los has visto "
            "caminar unos días. Es mucho más fácil si **antes** viste caminar a cientos de personas: ya sabes "
            "qué detalles importan (el balanceo de los brazos, el largo del paso) aunque esas personas no sean "
            "tus compañeros. Después solo te falta *afinar* lo que sabes con las pocas personas que sí te "
            "importan.\n\n"
            "Con la red pasa lo mismo, en dos etapas:\n\n"
            "1. **Preentrenamiento** con un dataset público (ECG-ID, muchas personas). La red aprende *cómo "
            "mirar* un ECG: dónde están el QRS, la onda T, cómo varían de persona a persona. Esas personas no "
            "son las nuestras; no importa.\n"
            "2. **Fine-tuning** con las grabaciones del Polar. Primero cambiamos la *cabeza* (la parte final que "
            "solo sirve para entrenar; ahora hay otras personas y otro número de clases) y **congelamos** las "
            "capas tempranas para que no se desordenen (etapa A). Luego **descongelamos todo** y entrenamos con "
            "una tasa de aprendizaje mucho más baja, para ajustar sin borrar lo aprendido (etapa B).\n\n"
            "**¿Por qué no solo entrenar con el Polar?** Con menos de 10 personas la red tiende a memorizar a "
            "esas personas en vez de aprender a distinguir en general. Este notebook mide si el preentrenamiento "
            "realmente ayuda."
        ),
        nbf.v4.new_code_cell(SETUP03),
        nbf.v4.new_markdown_cell("### Diagrama del proceso"),
        nbf.v4.new_code_cell(
            '''fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
steps = [("1. Preentrenar\\n(dataset público,\\nmuchas personas)", "#cfe8ff"),
         ("2A. Cabeza nueva\\n+ capas tempranas\\ncongeladas", "#ffe9b3"),
         ("2B. Todo descongelado\\n(tasa de aprendizaje\\nbaja)", "#ffd0d0"),
         ("3. Red final\\n(huella de 128\\nnúmeros)", "#d5f5d5")]
for i, (txt, color) in enumerate(steps):
    ax.text(i * 2.7 + 1.1, 0.5, txt, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", fc=color, ec="gray"))
    if i < len(steps) - 1:
        ax.annotate("", xy=((i + 1) * 2.7 + 0.1, 0.5), xytext=(i * 2.7 + 2.1, 0.5),
                    arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-0.3, 10.8); ax.set_ylim(0, 1); plt.show()'''
        ),
        nbf.v4.new_markdown_cell("## Entrenamos los tres modelos con la MISMA partición de personas"),
        nbf.v4.new_code_cell(
            '''from src.train import pretrain, finetune, polar_only
import json
pre = pretrain(cfg, public_train, public_val, models_dir)
fin = finetune(cfg, split, pre, models_dir)
pol = polar_only(cfg, split, models_dir)
for name, p in (("preentrenado", pre), ("fine-tuned", fin), ("solo Polar", pol)):
    side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"{name:14s} {p.name}  mejor EER de validación = {side['metrics']['best_val_eer']:.3f}")'''
        ),
        nbf.v4.new_markdown_cell(
            "## El experimento: EER de cada enfoque\n"
            "El umbral de cada modelo se elige con su EER de **validación**; las filas *NO vistos (hold-out)* son "
            "personas que ni el preentrenamiento ni el fine-tuning vieron. Cada fila muestra cuántos pares "
            "genuinos/impostores y cuántas personas respaldan el número; con pocas personas el intervalo de "
            "confianza puede aparecer como *n/a* a propósito."
        ),
        nbf.v4.new_code_cell(
            '''from src.checkpoint import load_checkpoint
from src.evaluate import compare_models, make_model_embed_fn, raw_embed_fn
from IPython.display import Markdown, display
fns = {"Solo Polar": make_model_embed_fn(load_checkpoint(pol, cfg)),
       "Preentrenado + fine-tuning": make_model_embed_fn(load_checkpoint(fin, cfg)),
       "Señal cruda": raw_embed_fn}
display(Markdown(compare_models(fns, split, cfg)))'''
        ),
        nbf.v4.new_markdown_cell(
            "## Las huellas de personas no vistas: antes y después del fine-tuning\n"
            "Mismas ventanas de hold-out proyectadas con PCA. Si el fine-tuning ayuda, los grupos por persona se "
            "ven más separados a la derecha que a la izquierda."
        ),
        nbf.v4.new_code_cell(
            '''fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, ckpt) in zip(axes, (("Solo preentrenado", pre), ("Preentrenado + fine-tuning", fin), ("Solo Polar", pol))):
    emb = make_model_embed_fn(load_checkpoint(ckpt, cfg))(split.holdout.windows)
    proj = PCA(2).fit_transform(emb)
    for p in split.holdout.persons():
        m = split.holdout.participant_ids == p
        ax.scatter(proj[m, 0], proj[m, 1], s=12, label=p)
    ax.set_title(title)
axes[0].legend()
plt.tight_layout(); plt.show()'''
        ),
        nbf.v4.new_markdown_cell(
            "## Cómo leer los resultados\n"
            "- Si *Preentrenado + fine-tuning* tiene EER menor que *Solo Polar* en **NO vistos**, el preentrenamiento "
            "ayuda y vale la pena su complejidad.\n"
            "- Si son parecidos, la ventaja no está demostrada con estos datos; con más participantes puede cambiar. "
            "Reporta ambos resultados con honestidad.\n"
            "- *Señal cruda* es la referencia sin entrenamiento: cualquier mejora sobre ella es lo que aporta la red.\n"
            "- Con 2–3 personas de hold-out hay muy pocos pares impostores: los números son una **prueba de concepto**."
        ),
    ]
    return nb
```
Then change the registry line to:
```python
NOTEBOOKS = {"02_model_and_training.ipynb": nb02, "03_finetuning_explained.ipynb": nb03}
```

- [ ] **Step 2: Generate and execute both notebooks**

```powershell
python notebooks/build_notebooks.py
jupyter nbconvert --to notebook --execute notebooks/03_finetuning_explained.ipynb --output-dir "$env:TEMP\nb_check"
```
Expected: no exception. Execution should take a few minutes at most on synthetic data.

- [ ] **Step 3: Visually check the output**

Confirm: the diagram renders four boxes with arrows; the training cell prints three checkpoints; the comparison table has 9 rows (3 models × 3 scenarios) with pair counts; the hold-out PCA shows three panels. Report the printed EER values to the user as-is; on synthetic data they say nothing about real performance.

- [ ] **Step 4: Checkpoint**

Propose commit: `docs: notebook 03 (fine-tuning explained and Polar-only vs fine-tuned)`

---

### Task 16: README and final verification

**Files:**
- Create: `README.md`, `models/.gitkeep`, `reports/.gitkeep`

**Interfaces:**
- Consumes: all commands defined above.
- Produces: a README that a teammate can follow to go from zero to a trained model.

- [ ] **Step 1: Write `README.md`**

```markdown
# Continuous ECG authentication: embedding model

Neural network that turns a 5 s Polar H10 ECG window (130 Hz) into a 128-d embedding; cosine
similarity against a template tells whether two windows come from the same person, including
people never seen in training. See `docs/superpowers/specs/2026-10-01-ecg-embedding-model-design.md`.

## Setup
    py -3.12 -m venv .venv
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    pytest

## Data contract
One `.npz` per recording in `data/processed/polar/` (and `data/processed/public/`):
`windows` float32 `(N, 650)` (time-ordered), `participant_id`, `session_id`, `activity`, `source`.
`session_id` must sort chronologically (e.g. `2026-10-01_s1`). Each person needs at least 2 sessions
on different days. Use `src/windowing.py::signal_to_windows` to cut a recording into windows.

## Hold-out
Edit `HOLDOUT_IDS` in `src/config.py` once (at least 2 Polar participants) and do not change it later.
Training refuses to start without it, and refuses if a hold-out person leaks into training data.

## Commands
    # Day one, Polar data only
    python -m src.train polar-only

    # Main path: public pretraining, then fine-tuning
    python -m src.public_data download
    python -m src.public_data build
    python -m src.train pretrain
    python -m src.train finetune --from-checkpoint models/modelo_v1.pt

    # Experiment: Polar-only vs pretrained + fine-tuned (EER with CIs and pair counts)
    python -m src.evaluate compare --polar-only models/modelo_v2.pt --finetuned models/modelo_v3.pt

    # Pick the version the app will use
    python -m src.train promote --version 3

Notebooks: `python notebooks/build_notebooks.py` regenerates them; open `notebooks/02_*.ipynb` and `03_*.ipynb`.

## Public dataset
ECG-ID (PhysioNet). License: <fill in from https://physionet.org/content/ecgiddb/ when checked in Task 12>.
Used only for pretraining; different lead/hardware than the Polar chest strap.

## Caveat
With fewer than 10 participants the hold-out evaluation rests on few impostor pairs. Reports print
pair counts and person-level bootstrap intervals; treat early numbers as a proof of concept.
```
Replace the license placeholder with the value noted in Task 12 Step 5 before finishing; if Task 12's manual check was skipped, say so in the README instead of leaving a placeholder.

- [ ] **Step 2: Create `models/.gitkeep` and `reports/.gitkeep`** (empty files).

- [ ] **Step 3: Run the full test suite from a clean shell**

Run: `pytest -v`
Expected: all tests pass, in a few minutes at most. Paste the summary line to the user.

- [ ] **Step 4: Spec coverage walk-through**

Confirm each is satisfied and note where: window contract (Tasks 1–2), person-level splits and hold-out guard (3, 11), model (5), two-stage training and fallback (11), augmentation train-only (4, 11), early stop on EER (10), versioned checkpoints with sidecar (9), metrics, bootstrap by person, pair counts (6–8), Polar-only vs fine-tuned experiment (8, 13, notebook 03), raw vs embedding (7–8), notebooks (14–15), error handling (all), tests with synthetic data (all). List any gap to the user instead of hiding it.

- [ ] **Step 5: Checkpoint**

Propose commit: `docs: README and final verification`. Then tell the user: the code is ready for real data; the next steps outside this plan are recording Polar sessions into the data contract and choosing `HOLDOUT_IDS`.
