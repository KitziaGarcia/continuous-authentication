# Project: Continuous Biometric Authentication via ECG

This document is the brief for our degree project. Use it as a guide to develop the complete system. We are two Computer Engineering students who will **supervise** the development and later **explain and defend it**, so everything must be understandable, modular, and well documented. We prefer a clear and solid solution over a "super complex" one that we cannot explain.

---

## 1. Context

- Team's previous work: an embedded system (ESP32-S3 + custom analog circuit with INA128P and 0.5–45 Hz filters) that identified 5 people with a small 1D-CNN (closed-set classification with softmax, input of 2500 samples at 500 Hz, TensorFlow Lite). This project does **not** reuse that hardware or that classification approach.
- There is now **no embedded-system constraint**: processing runs on a laptop.
- Acquisition hardware: **Polar H10** (chest strap, single lead, ECG at **130 Hz**, over Bluetooth Low Energy).
- Computing equipment: laptop with 16 GB of RAM, 6-core CPU, no dedicated GPU. Google Colab available as a fallback if training turns out to be slow.

## 2. Objective

A web-based exam application where the student authenticates with their ECG and **the system keeps continuously verifying** that it is the same person throughout the entire exam. If it detects that the person changed (for example, someone else put on the strap mid-exam) or the signal is lost, the exam is locked.

Rationale: the initial authentication can be forged or "handed off" to another person afterward; continuous verification closes that gap. Applicable to any access to sensitive information.

## 3. Key design decision: verification with embeddings, not classification

**Do not use closed-set classification with softmax as the final system.** Problems:
- Softmax always assigns one of the known classes, even to an impostor it has never seen.
- Registering a new user would require retraining the network.

**Use embedding learning (metric learning):**
- The network converts an ECG window into a vector (embedding) of 64–128 dimensions.
- It is trained so that vectors from the same person end up close together and vectors from different people end up far apart.
- Comparison is done with **cosine similarity** between vectors.
- It must work with people the network **never saw** during training.

### The three phases of the system

| Phase | Is the network trained? | What happens | When |
|---|---|---|---|
| Training | Yes | The network learns to convert ECG into useful vectors. Saved to `models/modelo_ecg.pt` | During development |
| Enrollment | No (frozen network) | The user records ~1–2 min, their vectors are computed and the average is stored as a template in the database | Each new user |
| Continuous authentication | No (frozen network) | Each new window passes through the network and its vector is compared with the template | During the exam |

Important separation to maintain in the code:
- **The network (weights in `.pt`)** = the knowledge of "how to look at" an ECG. It does not change when registering or deleting users.
- **The template database** = the "memory" of each registered person.
- **The decision logic** = threshold and locking rules.

## 4. The winning combo

| Component | Choice |
|---|---|
| Acquisition | Polar H10 via BLE from a Python backend (`bleak`) |
| Preprocessing | 0.5–40 Hz band-pass, R-peak detection with NeuroKit2, ~5 s overlapping windows, z-score |
| Model | Small residual 1D-CNN → 64–128 dim embedding (PyTorch) |
| Training | Classifier over many people, then use the penultimate layer as the embedding |
| Verification | Cosine similarity against the user's template + threshold chosen at the EER |
| Continuous logic | Moving average of scores, signal quality index, locking rule with tolerance |
| Baseline | Fiducial features + SVM |
| Web | FastAPI backend with WebSocket to the exam frontend, showing live authentication status |

### Details of each component

**Acquisition (Polar H10)**
- Single-lead ECG at 130 Hz. The H10's ECG stream is enabled through its PMD (Polar Measurement Data) service over BLE; research the protocol and commands needed to start the stream from Python with `bleak`.
- Save raw recordings with metadata (participant, session, date, activity).
- Detect and report disconnections.

**Preprocessing**
- ~0.5–40 Hz band-pass filter (Nyquist = 65 Hz).
- R-peak detection with NeuroKit2 (also serves for the signal quality index).
- ~5 s windows (~650 samples) with overlap (for example 50%).
- Per-window z-score normalization.
- Main representation: **filtered 1D signal**. Do not use Gramian Angular Field or spectrograms as the main approach (heavier and with no clear advantage at 130 Hz with a single lead).

**Model**
- 1D-CNN with residual blocks, 4–6 convolutional layers, ending in an L2-normalized embedding of 64–128 dimensions.
- Main training strategy: train as a classifier over all training people and use the penultimate layer as the embedding. Alternatives to evaluate if time permits: triplet loss or ArcFace.
- Must train on CPU in a reasonable time (minutes to a few hours).
- Save versioned checkpoints (`modelo_v1.pt`, `modelo_v2.pt`, ...).

**Verification and continuous logic**
- User template = average of the embeddings from their enrollment.
- Score per window = cosine similarity with the template.
- Threshold chosen at the EER point on the validation set.
- **Temporal smoothing:** moving average of scores or an "N of the last M windows below the threshold" rule before locking, so as not to lock because of a single bad window (cough, movement).
- **Signal quality index:** if there are no plausible R peaks or the heart rate is impossible, mark the window as "unreliable", not as "impostor".
- **Signal loss / disconnection:** pause or lock the exam.
- Parameters (threshold, N, M, window size, overlap) configurable in a single configuration file.
- Optional extension: **adaptive template update** (gradually blend high-confidence embeddings into the template) without retraining the network.

**Baseline**
- Fiducial features (amplitudes and intervals: QRS, QT, PR, RR, etc.) + SVM with scikit-learn, to compare against the network.
- Additional experiment: verification with direct similarity on the raw signal vs. on the embeddings, to demonstrate what the training contributes.

**Web**
- FastAPI backend that receives the Polar stream, processes windows, and runs the verification.
- WebSocket to the frontend to show live: status (authenticated / doubtful / locked), current score, and signal plot.
- Page flow: enrollment → exam start → continuous monitoring → lock if it fails.
- It can be a simple test exam; what matters is the authentication mechanism.

## 5. Data

The bottleneck of the project. Requirements:
- Ideally recruit **20–30 participants**, with informed consent.
- **Multiple sessions per person on different days** (variability between sessions, due to strap placement and physiological state, is the real challenge).
- Include realistic exam conditions: seated at rest, reading, solving problems, talking.
- **Splits by person, not by window:** the test set must include people the network never saw in training, in order to evaluate unknown impostors.
- Also evaluate **across sessions** (enroll on one day, test on another day) and report it honestly.
- Optional: pretrain with public databases (for example PhysioNet's ECG-ID or Heartprint) resampled to 130 Hz and then fine-tune with Polar data. Keep in mind that these are different leads from the chest strap.
- Optional data augmentation only in training (jitter, scaling, magnitude/time warping).

## 6. Evaluation and metrics

- **EER** (Equal Error Rate) and **ROC** curve.
- **FAR** and **FRR** at the chosen threshold.
- **Time to detection:** experiment where person A enrolls and starts the exam and halfway through hands the strap to person B; measure how many seconds the system takes to lock.
- **False lock rate** for the legitimate user during a full exam.
- Comparisons: network vs. SVM baseline vs. raw signal; same session vs. across sessions; with and without temporal smoothing.

## 7. Development stack

- **Python** with a virtual environment (`venv` or conda).
- **PyTorch** (model and training).
- **NeuroKit2, SciPy, NumPy** (signal processing).
- **scikit-learn** (metrics, SVM baseline).
- **Matplotlib** (plots).
- **bleak** (BLE with the Polar H10).
- **FastAPI + WebSockets** (server and live communication).
- **SQLite** (user templates).
- **VS Code + Jupyter notebooks** for exploration and explanation.
- **Git + GitHub** for teamwork.

## 8. Proposed project structure

```
ecg-auth/
├── data/
│   ├── raw/              ← Polar recordings as-is
│   └── processed/        ← already filtered and normalized windows
├── notebooks/            ← to explore and understand each stage
├── src/
│   ├── acquisition.py    ← connection to the Polar H10
│   ├── preprocessing.py  ← filters, R peaks, windows
│   ├── model.py          ← network definition
│   ├── train.py          ← trains and saves the .pt
│   ├── evaluate.py       ← EER, FAR, FRR, ROC curves
│   └── decision.py       ← continuous authentication logic
├── models/
│   └── modelo_ecg.pt     ← the trained network
├── app/                  ← FastAPI server + exam page
└── templates.db          ← vectors of registered users
```

Usage flow:
1. `python src/train.py` → trains and saves `models/modelo_ecg.pt`.
2. `python app/...` (server) → on startup it loads the `.pt` once and uses it for enrollment and authentication, without retraining.

## 9. How we want to work

- **Build one module at a time**, in this suggested order: acquisition → preprocessing → model and training → evaluation → continuous logic → web.
- **Each module with a notebook** that shows it working with plots: what goes in, what comes out, and why (raw vs. filtered signal, detected R peaks, windows, embeddings projected in 2D with PCA or t-SNE, distributions of genuine vs. impostor scores, etc.).
- Code commented in Spanish explaining the **why** of each decision, not just the what.
- One README per module or one general README that explains the architecture in plain language.
- Basic tests for preprocessing and decision logic (for example, with synthetic signals).
- Before important design decisions, explain the options and reasoning to us.
- Prioritize clarity and reproducibility (fixed seeds, centralized configuration, scripts that can be rerun).
