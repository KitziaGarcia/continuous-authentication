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
with torch.no_grad():
    z = net.eval()(x)
print("embedding ->", tuple(z.shape), "| norma:", float(z.norm()))'''
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
            '''from src.evaluate import split_enroll_test, run_protocol, eer, far_frr_at, first_session_table
embed = make_model_embed_fn(trained)
# El umbral se elige SOLO con validación (sesión posterior de las personas de entrenamiento);
# el hold-out jamás decide el umbral, solo se mide con él.
val_trials = run_protocol(embed, first_session_table(split.train), split.val)
val_eer, thr = eer(val_trials.genuine, val_trials.impostor)
enroll, test = split_enroll_test(split.holdout, "cross_session")
trials = run_protocol(embed, enroll, test)
hold_eer, _ = eer(trials.genuine, trials.impostor)
far, frr = far_frr_at(trials.genuine, trials.impostor, thr)
plt.hist(trials.impostor, bins=30, alpha=0.6, label="impostores")
plt.hist(trials.genuine, bins=30, alpha=0.6, label="genuinos")
plt.axvline(thr, color="k", ls="--", label=f"umbral (EER de validación={val_eer:.2f})")
plt.legend(); plt.xlabel("similitud coseno"); plt.title("Hold-out (personas no vistas)"); plt.show()
print(f"EER en hold-out: {hold_eer:.3f} | con el umbral de validación: FAR={far:.3f}, FRR={frr:.3f}")
print(f"Pares: {len(trials.genuine)} genuinos / {len(trials.impostor)} impostores de {trials.n_owners()} personas")'''
        ),
    ]
    return nb


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


NOTEBOOKS = {"02_model_and_training.ipynb": nb02, "03_finetuning_explained.ipynb": nb03}

if __name__ == "__main__":
    for name, build in NOTEBOOKS.items():
        nbf.write(build(), HERE / name)
        print("escrito", name)
