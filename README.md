# Continuous ECG authentication: embedding model

Neural network that turns a 5 s Polar H10 ECG window (130 Hz) into a 128-d embedding; cosine
similarity against a template tells whether two windows come from the same person, including
people never seen in training. See `docs/superpowers/specs/2026-10-01-ecg-embedding-model-design.md`
and the task-by-task plan in `docs/superpowers/plans/2026-10-01-ecg-embedding-model.md`.

## Setup
    py -3.12 -m venv .venv        # any Python 3.10+ works; 3.14 was verified with torch 2.14 CPU
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    pytest

## Data contract
One `.npz` per recording in `data/processed/polar/` (and `data/processed/public/`):
`windows` float32 `(N, 650)` (time-ordered), `participant_id`, `session_id`, `activity`, `source`.
`session_id` must sort chronologically (e.g. `2026-10-01_s1`). Each person needs at least 2 sessions
on different days. Use `src/windowing.py::signal_to_windows` to cut a recording into windows and
`src/data.py::save_window_file` to write the `.npz`.

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

Notebooks (in Spanish): `python notebooks/build_notebooks.py` regenerates them; open
`notebooks/02_model_and_training.ipynb` and `notebooks/03_finetuning_explained.ipynb`. They run on
synthetic data until real Polar recordings exist in `data/processed/polar/`.

## Data collection protocol (Polar H10)
Step-by-step guide for the team and volunteers (in Spanish): [`docs/guia_recoleccion.md`](docs/guia_recoleccion.md).
Each person gets a fixed anonymous ID (`P01`, `P02`, ...) and records **3 sessions on different days**
(~5.5 min each: rest, reading, problem solving, talking):

    python -m src.session --id P03                      # one guided sitting; the date and file names are automatic
    python -m src.session --id P03 --redo reposo        # repeat one activity (the old file moves to data/descartadas/)
    python -m src.status                                # who recorded which days, who is due next

Files are written as `data/processed/polar/P03__2026-10-06__reposo.npz` (one per activity; all activities of a
sitting share the same session id = the date). Each activity is checked automatically (`src/quality.py`: flat
signal, spikes, irregular rhythm, lost beats); bad ones are set aside, never deleted. Personal data is git-ignored
and `tests/test_repo_hygiene.py` guards it.

## Quick trial (older, ad-hoc recorder) with the Polar H10 (2 people)
Record each person at least twice (enrollment + test; take the strap off and put it back on in between,
ideally on different days), then check whether the model tells you apart:

    python -m src.record --id kitzia --seconds 60      # also: --simulate to test without the strap
    python -m src.try_me                               # uses models/modelo_ecg.pt or the latest modelo_vN.pt

`record` saves the raw signal in `data/raw/polar/`, the windows in `data/processed/polar/`, and opens a PNG
(`reports/recordings/`) with the whole signal and a 10 s zoom so you can confirm a real heartbeat was captured.
`try_me` enrolls each person with their first recording, scores their later recordings against everyone's
fingerprint, and prints plain-language results plus a picture. With only 2 people there is no one left to test a
fine-tuned model on, so the trial uses the ECG-ID-pretrained model as is.

## Public dataset
ECG-ID (PhysioNet, https://physionet.org/content/ecgiddb/): 90 people, 310 recordings, 500 Hz.
License: **Open Data Commons Attribution License v1.0** (checked on the dataset page on 2026-10-01;
attribute the dataset when publishing results). Used only for pretraining: it is a different lead and
hardware than the Polar chest strap, which is why fine-tuning on Polar data follows.

## Caveat
With fewer than 10 participants the hold-out evaluation rests on few impostor pairs. Reports print
pair counts and person-level bootstrap intervals (shown as `n/a` with fewer than 3 people); treat
early numbers as a proof of concept.
