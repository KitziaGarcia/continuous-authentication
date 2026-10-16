# ECG Embedding Model — Design Spec

Date: 2026-10-01
Status: Draft for review
Parent document: `docs/brief_continuous_ecg_authentication.md`

## 1. Purpose and scope

Build the neural network that converts a 5 s ECG window (Polar H10, 130 Hz) into an L2-normalized embedding (a "fingerprint") so that cosine similarity tells whether two windows come from the same person, including people never seen in training.

### In scope
- `src/model.py`: the network.
- `src/train.py`: pretraining, fine-tuning and Polar-only training.
- `src/data.py`: datasets, splits by person, augmentation, public-dataset loader.
- `src/config.py`: all parameters, seeds, hold-out list.
- `src/evaluate.py`: EER, ROC/AUC, FAR/FRR, bootstrap CIs, report tables.
- `tests/` with synthetic signals; two notebooks.

### Out of scope (separate specs)
Acquisition (BLE), full preprocessing module, continuous decision logic (smoothing/locking), web app, SVM baseline, ArcFace/triplet loss.

### Success criteria
- The network trains on CPU in minutes to about an hour.
- EER and ROC are reported on unseen people and across sessions, with confidence intervals and the number of pairs behind each number.
- The experiment "Polar-only vs. pretrained + fine-tuned" is run and reported.
- Two students can explain every component and decision.

## 2. Context and assumptions

- Polar H10 recordings are the primary data. The team already owns the sensor.
- Initially fewer than 10 participants; the number is expected to grow.
- Hardware: 16 GB RAM, 6-core CPU, no GPU (Colab as a fallback).
- Stack: Python, PyTorch, NumPy/SciPy, scikit-learn, Matplotlib.
- The preprocessing module is a dependency with a thin contract (Section 3); its own spec comes later.

## 3. Data contract

- **Window:** 650 samples (5 s at 130 Hz), z-scored per window, tensor shape `(1, 650)`.
- **Metadata per window:** `participant_id`, `session_id`, `activity`, `source` (`polar` or `public`).
- **Sources:**
  - Polar H10 recordings (primary).
  - A public multi-person dataset (ECG-ID to start), resampled to 130 Hz, used only for pretraining.

### Splits (always by person, never by window)

Three independent groups of people, plus the public dataset. Planned size: **20 Polar participants** (3 sessions each, on different days).

| Group | People (of 20) | Used for | Rule |
|---|---|---|---|
| Public-dataset people (ECG-ID) | – | Pretraining | Never mixed with Polar people |
| Polar **train** | 14 | Fine-tuning / Polar-only training | Sessions 1–2 train the network; session 3 is a cross-day check on these same people |
| Polar **validation** | 2 | Early stopping on EER and threshold selection | Unseen people: never used to update weights |
| Polar **held-out (test)** | 4 | Final test with unseen users and impostors | Used once, at the end. Never used in pretraining, fine-tuning, early stopping or threshold selection |

- **Sessions for validation and test people:** session 1 is enrollment (builds the template); sessions 2 and 3 are authentication attempts, both genuine and impostor (cross-session by construction).
- **Why a separate validation group of unseen people:** early stopping and the threshold depend on EER, and the real system must work for people the network never saw. Validating only on later sessions of training people would give optimistic numbers.
- The validation and hold-out lists live in `config.py`, chosen at random once, **after** data collection and **before** any Polar training, and never changed afterwards. `data.py` raises an error if a validation or held-out `participant_id` appears in any training split.
- **Scaling rule if fewer people are recruited:** keep at least 2 hold-out people and 1 validation person; give the rest to training (e.g. 10 people → 6 train / 1 validation / 3 test).
- **Optional upgrade:** person-level 5-fold cross-validation (5 folds of 4 people) so all 20 people serve once as unseen; costs 5 trainings (minutes each on CPU) and gives tighter confidence intervals.
- Honest caveat: with 4 test people there are only 12 (template, impostor) person combinations, so the held-out EER rests on few impostor pairs and carries wide confidence intervals. Results are labeled a proof of concept; the protocol is written so the numbers become more meaningful as the dataset grows.

## 4. Model

### 4.1 Tipo de red
Red neuronal convolucional 1D (1D-CNN) con bloques residuales (estilo ResNet), usada como **extractor de embeddings** (aprendizaje de métricas), no como clasificador final. Entrada: señal ECG cruda filtrada de un solo canal; salida: vector de 128 dimensiones con norma L2 = 1.

### 4.2 Arquitectura capa por capa

Dimensiones para una ventana de entrada `(1, 650)` (canales, muestras). Todas las convoluciones usan `bias=False` porque BatchNorm ya aporta el desplazamiento.

| # | Bloque | Operación | Kernel / stride | Canales (in → out) | Salida (C × T) |
|---|---|---|---|---|---|
| 0 | Entrada | Ventana z-score | – | 1 | 1 × 650 |
| 1 | Stem | Conv1d + BatchNorm + ReLU | k=7, s=1, pad=3 | 1 → 16 | 16 × 650 |
| 2 | Etapa residual 1 | Bloque residual | k=5, s=2 | 16 → 32 | 32 × 325 |
| 3 | Etapa residual 2 | Bloque residual | k=5, s=2 | 32 → 64 | 64 × 163 |
| 4 | Etapa residual 3 | Bloque residual | k=5, s=2 | 64 → 128 | 128 × 82 |
| 5 | Etapa residual 4 | Bloque residual | k=5, s=2 | 128 → 128 | 128 × 41 |
| 6 | Pooling global | AdaptiveAvgPool1d(1) sobre el tiempo | – | 128 | 128 |
| 7 | Embedding | Linear (capa densa) | – | 128 → 128 | 128 |
| 8 | Normalización | L2 (`F.normalize`) | – | 128 | 128 (norma 1) |
| 9 | Cabeza (solo entrenamiento) | Linear sobre el embedding **sin normalizar** | – | 128 → N personas | N |

### 4.3 Bloque residual (detalle)
Cada etapa es el mismo bloque:

```
x ─┬─ Conv1d(k=5, stride=2, pad=2) → BatchNorm → ReLU
   │   → Conv1d(k=5, stride=1, pad=2) → BatchNorm ──────────┐
   └─ atajo: Conv1d(k=1, stride=2) → BatchNorm ───────────────┴─ suma → ReLU
```

- El **stride 2** de la primera convolución reduce a la mitad la longitud temporal en cada etapa (650 → 325 → 163 → 82 → 41) y duplica o mantiene los canales.
- El atajo usa una convolución 1x1 con el mismo stride para que las dimensiones coincidan en la suma, en las cuatro etapas.
- La ReLU se aplica después de la suma.

### 4.4 Hiperparámetros de la arquitectura
- Activación: ReLU. Normalización: BatchNorm1d. Sin dropout.
- Campo receptivo: crece con la profundidad; la última etapa "ve" segmentos de varios latidos.
- Dimensión del embedding: 128 (configurable entre 64 y 128 en `config.py`).
- Anchos de canal configurables: `stem_channels=16`, `stage_channels=(32, 64, 128, 128)`, `kernel_size=5`.

### 4.5 Tamaño
- **371,024 parámetros** en el codificador (hasta el embedding, sin cabeza). La cabeza añade `128 × N + N` parámetros (N = personas de entrenamiento) y se descarta tras entrenar. Valor medido con `ECGEmbeddingNet.n_parameters()`.
- Distribución aproximada: las etapas 3 y 4 concentran ~84 % de los parámetros (36 % y 49 %).

### 4.6 Uso e inferencia
- Pooling global promedio: hace la huella independiente de en qué punto de la ventana cae cada latido y de la longitud exacta.
- La normalización L2 hace que el producto punto sea igual a la similitud coseno.
- La cabeza recibe el embedding **sin normalizar** (la "penúltima capa" del brief); la huella usada para verificar es la versión normalizada.
- Solo se guarda el codificador (`encoder_state_dict`); la cabeza se descarta. La red (pesos) permanece separada de la base de plantillas y de la lógica de decisión, como en el brief.

## 5. Training

### Stage 1: Pretraining (public dataset)
Plain classifier over the public identities: cross-entropy with label smoothing. Goal: learn general ECG shape features (R peak, T wave, interval differences between people).

### Stage 2: Fine-tuning (Polar training people)
- Replace the pretraining head with a new head sized for the Polar training people.
- Step A: freeze the early layers; train only the last stage and the head for a few epochs.
- Step B: unfreeze everything and train with a much lower learning rate.
- Purpose: adapt to the Polar strap and the team's people without erasing what pretraining learned.

### Fallback: Polar-only
Same `train.py` with pretraining skipped, selected by one config flag. Works from day one, before any public data is downloaded.

### Details
- **Augmentation (training only), valores en `config.py`:**

  | Técnica | Parámetro | Valor |
  |---|---|---|
  | Jitter (ruido gaussiano) | `jitter_std` | 0.02 |
  | Magnitude warping (escala suave variable en el tiempo) | `magnitude_warp` | 0.10 |
  | Time warping (deformación suave del eje temporal) | `warp_strength` | 0.05 |
  | Deriva lenta de línea base | `wander_amplitude` | 0.10 |

  Un valor de 0 desactiva cada técnica.
- **Optimizer:** AdamW (`weight_decay=1e-4`) con scheduler `CosineAnnealingLR` (T_max = número de épocas de la etapa).
- **Pérdida:** entropía cruzada con `label_smoothing=0.1`.
- **Batch size:** 64.
- **Hiperparámetros por etapa:**

  | Etapa | Épocas | Learning rate | Qué se entrena |
  |---|---|---|---|
  | Preentrenamiento (público) | 30 | 1e-3 | Toda la red + cabeza |
  | Fine-tuning A | 5 | 1e-3 | Última etapa residual, capa de embedding y cabeza nueva (197,504 parámetros del codificador entrenables, 173,520 congelados) |
  | Fine-tuning B | 20 | 1e-4 | Toda la red |
  | Solo Polar | 40 | 1e-3 | Toda la red + cabeza |

- **Congelamiento:** en la etapa A, las capas congeladas (stem y etapas 1–3) también tienen su BatchNorm en modo evaluación, para que sus estadísticas no cambien.
- **Early stopping on validation EER**, no en accuracy de clasificación (paciencia = 5 épocas sin mejora), porque el objetivo es separar personas que la red no ha visto.
- **Checkpoints:** `models/modelo_vN.pt` plus a JSON sidecar with config, seed, hold-out list, data version and metrics. `modelo_ecg.pt` points to the chosen version. Solo se guardan los pesos del codificador.
- **Reproducibility:** semilla fija (`seed=42`) para `random`, NumPy y PyTorch; one config file, one command per stage.
- **Preprocesamiento de entrada (referencia):** pasa-banda 0.5–40 Hz, ventanas de 650 muestras con 50 % de traslape, z-score por ventana.
- **Loss:** configurable, only cross-entropy implemented in v1. Triplet and ArcFace are deliberately deferred (YAGNI) and can be added if EER is not good enough.

## 6. Evaluation (`evaluate.py`)

- **Score:** cosine similarity between a window's embedding and a person's template (mean of that person's enrollment embeddings).
- **Pairs:**
  - Genuine: template from session 1, windows from a later session of the same person (cross-session).
  - Impostor: a person's windows scored against another person's template.
- **Metrics:** EER, ROC, AUC, FAR and FRR at the chosen threshold.
- **Threshold:** chosen at the EER on validation data only; held-out people are never used to choose it.
- **Confidence intervals:** bootstrap resampling by person (not by window, since windows from one person are correlated).
- **Report tables:**
  - same-session vs. cross-session;
  - seen people (validation sessions) vs. unseen people (held-out);
  - **Polar-only vs. pretrained + fine-tuned** (same split, windows and protocol; EER with CIs for both);
  - raw-signal similarity vs. embedding similarity.
- Every number is printed with the count of genuine and impostor pairs it rests on.

## 7. Testing (synthetic signals, no real data required)

- **Model:** output shape `(B, 128)`; norms equal 1.0; deterministic in eval mode; a tiny overfit run on a small batch drives loss down.
- **Data:** window shape `(1, 650)`; a leakage test fails if any held-out `participant_id` appears in train or validation.
- **Evaluation:** perfectly separated scores give EER ≈ 0; random scores give EER ≈ 0.5.
- **Checkpoint:** save/load gives identical embeddings; the sidecar JSON contains all required fields.

## 8. Error handling (fail loudly, clear message)

- Wrong window length or sampling rate, or NaN values in the data.
- A held-out person found in the training data.
- Missing or incompatible checkpoint.
- Too few windows or sessions per person to evaluate.

## 9. File layout

```
src/
  config.py        all parameters, seeds, hold-out list
  data.py          datasets, splits by person, augmentation, public-data loader
  model.py         network
  train.py         pretrain / finetune / polar-only modes
  evaluate.py      metrics, bootstrap, report tables
tests/             synthetic tests from Section 7
notebooks/
  02_model_and_training.ipynb    architecture, loss curves, 2D embeddings (PCA/t-SNE)
  03_finetuning_explained.ipynb  plain-language fine-tuning + Polar-only vs. fine-tuned
models/            modelo_vN.pt + sidecar JSON
```

Code comments are in Spanish explaining the why of each decision (per the brief). The fine-tuning notebook explains the step in plain language with a diagram and with embeddings projected in 2D before and after fine-tuning.

## 10. Risks and open items

- **Few identities:** with fewer than 10 people the embedding may generalize poorly. Mitigation: public pretraining, augmentation, honest reporting.
- **Domain gap:** public datasets use different leads and hardware than the Polar chest strap. Mitigation: fine-tuning; the Polar-only vs. fine-tuned experiment measures whether pretraining actually helps.
- **Thin held-out evaluation:** 2–3 people give few impostor pairs. Mitigation: pair counts and person-level bootstrap CIs in every report.
- **Public dataset choice:** ECG-ID is the starting point; confirm its license, sampling rate and resampling to 130 Hz during the first implementation task.

## 11. Next step

After spec approval, write the task-by-task implementation plan in a separate document (`docs/superpowers/plans/`) using the writing-plans skill.
