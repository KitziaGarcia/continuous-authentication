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

| Group | Used for | Rule |
|---|---|---|
| Public-dataset people | Pretraining | Never mixed with Polar people |
| Polar **train** people | Fine-tuning / Polar-only training | Later sessions held out as validation |
| Polar **held-out** people (2–3, fixed) | Unseen-impostor test | Never used in pretraining or fine-tuning |

- The hold-out list lives in `config.py`. `data.py` raises an error if a held-out `participant_id` appears in any training or validation split.
- Honest caveat: with fewer than 10 people and 2–3 held out, only about 5–7 Polar people remain for training, and held-out EER rests on very few impostor pairs. Results are labeled a proof of concept until more participants are added; the protocol is written so the numbers become meaningful as the dataset grows.

## 4. Model

Small residual 1D-CNN.

```
input (1, 650)
 → stem: Conv1d(k=7) 16 ch, BatchNorm, ReLU
 → 4 residual stages, each downsampling by 2: channels 32 → 64 → 128 → 128
   (each stage: two Conv1d k=5 + BatchNorm + ReLU, with a skip connection)
 → global average pooling over time
 → Linear → embedding (128-d) → L2 normalization
 → [training only] Linear(128, n_people) classification head
```

- Roughly 200–300k parameters.
- Embedding size (64 or 128) and channel widths are set in the config.
- The classification head is discarded after training. Only the part up to the L2-normalized embedding is saved and used for enrollment and authentication.
- The network (weights) stays separate from the template database and from the decision logic, as in the brief.

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
- **Augmentation (training only):** jitter, amplitude scaling, small time warping, slow baseline-wander drift; strengths configurable.
- **Optimizer:** AdamW with cosine learning-rate schedule.
- **Early stopping on validation EER**, not classification accuracy, because the goal is separation of people the network has not seen.
- **Checkpoints:** `models/modelo_vN.pt` plus a JSON sidecar with config, seed, hold-out list, data version and metrics. `modelo_ecg.pt` points to the chosen version.
- **Reproducibility:** fixed seeds, one config file, one command per stage.
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
