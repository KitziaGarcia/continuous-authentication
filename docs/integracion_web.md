# Guía para conectar la web con el modelo de autenticación continua con ECG

**Para quién es:** quien haga la web (y el asistente de IA que use, Claude o Gemini, que **no** tiene acceso al
resto del repo). Este documento resume lo necesario para integrar el modelo sin leer todo el código.
Si algo aquí contradice al código, manda el código; avisa al equipo del modelo.

---

## 1. Qué hace el sistema

1. Una persona se pone una banda de pecho **Polar H10** (mide ECG a 130 Hz).
2. **Registro:** se graban ~60 s de su ECG y se guarda su "huella" (un vector de 128 números).
3. **Vigilancia:** cada ~2.5 s se toman sus últimos 5 s de ECG, se convierten en un vector de 128 números y se
   compara contra la huella (similitud coseno). Si el promedio de las últimas 6 comparaciones es ≥ umbral,
   la persona sigue **autenticada**; si no, **no autenticada**.

El modelo es una red neuronal pequeña (1D-CNN residual, ~371 mil parámetros, ~1.5 MB) en **PyTorch**.
Corre sin problema en CPU: una decisión toma ~5 ms (medido en un CPU de PC).

> **Estado: prototipo de investigación (tesis).** El error medido entre días con personas ya vistas es de
> ~7 % (EER). **No es un sistema de seguridad real.** No lo presentes como tal en la interfaz.

---

## 2. Arquitectura recomendada

```
Polar H10 --BLE--> [Backend Python: lee muestras, corre el modelo] --HTTP/WebSocket--> [Web: muestra el estado]
```

- **El modelo y todo el preprocesamiento viven en Python** (backend). La web **no** corre la red neuronal.
  El filtrado usa `scipy` (filtro de fase cero) y las normalizaciones deben ser idénticas a las del
  entrenamiento; reimplementarlo en JavaScript es arriesgado y **no se ha probado**.
- **Lectura de la banda (dos opciones):**
  - **A (recomendada, ya funciona):** el backend Python lee la banda por Bluetooth con `bleak` +
    `polar-python` (clase `PolarSource` en `src/session.py`). Backend y banda deben estar en la misma
    máquina (con Bluetooth), y la web solo muestra resultados.
  - **B (no probada):** el navegador lee la banda con Web Bluetooth y manda las muestras al backend. Solo
    Chrome/Edge, requiere HTTPS o `localhost`, y el protocolo ECG del Polar (servicio PMD) no está
    implementado en JS en este proyecto.
- Sugerencia: FastAPI + WebSocket para empujar el estado a la web. Aún no está instalado ni hay servidor en el
  repo; es trabajo nuevo.

---

## 3. Contrato de la señal (lo más importante: no cambiar nada de esto)

| Parámetro | Valor |
|---|---|
| Frecuencia de muestreo | **130 Hz** (lo que entrega el H10 en modo ECG) |
| Muestras | Enteros en µV, **crudos**, sin normalizar (14 bits) |
| Ventana de entrada al modelo | **650 muestras = 5 s** |
| Filtro | Pasa-banda 0.5–40 Hz, Butterworth orden 4, **fase cero** (`sosfiltfilt`) |
| Normalización | **z-score por ventana** (media 0, desviación 1) |
| Forma del tensor | `(N, 1, 650)`, `float32` |
| Salida | `(N, 128)`, vectores **ya normalizados a longitud 1** |
| Similitud | Producto punto con la huella = **similitud coseno** |

Todo el preprocesamiento está en `src/windowing.py::signal_to_windows(sig, fs, cfg)`. **Úsala; no la
reescribas.** Si cambia cualquier parámetro, el modelo deja de funcionar bien sin dar error.

Trucos que importan:
- **El filtro de fase cero necesita contexto.** Por eso, para decidir cada vez se usan las **últimas 1950
  muestras (15 s)**, se filtran juntas y se toma la **última ventana de 650**. No filtres solo 5 s.
- Una **ventana plana** (banda desconectada) se descarta antes del z-score; si no, quedaría convertida en ruido.
- Las primeras ~5 s tras conectar la banda son inestables: se descartan en el registro.
- Si la banda manda datos por lotes (BLE), recorta ~1 s en los bordes de cada tramo grabado.

---

## 4. Archivos del repo que importan

Solo estos; el resto (entrenamiento, notebooks, pruebas) se puede ignorar.

| Archivo | Para qué |
|---|---|
| `src/config.py` | Clase `Config` con todos los hiperparámetros (`fs=130`, `window_samples=650`, `embedding_dim=128`, ...). Usa `Config()` por defecto: debe coincidir con el modelo. |
| `src/model.py` | La red `ECGEmbeddingNet`. |
| `src/checkpoint.py` | `load_checkpoint(path, cfg)` devuelve la red lista en modo evaluación. |
| `src/windowing.py` | `signal_to_windows`: filtro, ventanas y z-score. |
| `src/evaluate.py` | `make_model_embed_fn(model)` (ventanas → embeddings sin gradientes) y `l2_normalize`. |
| `src/quality.py` | `check_quality(raw, fs, min_peaks)`: detecta señal plana, picos, ritmo imposible, huecos. |
| `src/live.py` | Lógica de autenticación continua ya probada: `make_template`, `score_window`, `Decider`, y el ciclo `monitor`. Es la referencia de cómo decidir. |
| `src/session.py` | `PolarSource`: lectura del H10 por Bluetooth (`start()`, `stop()`, `n_samples()`, `take(i0, i1)`). |

Dependencias: Python 3.10+ (probado en 3.14), `torch` (CPU), `numpy`, `scipy`, `scikit-learn`, `bleak`,
`polar-python`. **Ojo:** `requirements.txt` **no** lista `torch`; se instala aparte:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 5. Dónde está el modelo entrenado

Los modelos están en `models/` (cada `.pt` pesa ~1.5 MB y lleva al lado un `.json` con su configuración y
métricas). Los datos de ECG **no** están en el repo, solo los modelos.

| Archivo | Qué es |
|---|---|
| `modelo_v1.pt` | Solo preentrenado con un dataset público (**no usar** para la web) |
| `modelo_v2.pt` | Preentrenado + fine-tuning con datos Polar |
| `modelo_v3.pt` | Entrenado solo con datos Polar (el que hoy usa por defecto `src.live`) |

Todavía **no está decidido** cuál será el definitivo (empatan en las pruebas y se reentrenará con más datos).
Cuando se decida se copiará a `models/modelo_ecg.pt` y el nombre será estable. **Haz que la ruta del modelo
sea configurable** y no escribas `v3` fijo en el código. La huella guardada depende del modelo (ver abajo).

---

## 6. Código de integración (probado)

Motor mínimo para el backend. Probado con grabaciones reales: con la huella de una persona, 0 de 30
decisiones propias rechazadas y 29 de 30 de otra persona rechazadas.

```python
"""Motor de autenticación continua para el backend web (sin hardware)."""
import numpy as np

from src.checkpoint import load_checkpoint
from src.config import Config
from src.evaluate import make_model_embed_fn
from src.live import AUTH, BAD_SIGNAL, Decider, make_template, score_window
from src.quality import check_quality
from src.windowing import signal_to_windows

FS = 130
BUFFER = 3 * 650          # 15 s: 5 s de ventana + 10 s de contexto para el filtro


class AuthEngine:
    def __init__(self, checkpoint: str, threshold: float = 0.31, smooth: int = 6, reset_after: int = 2):
        self.cfg = Config()
        self.embed = make_model_embed_fn(load_checkpoint(checkpoint, self.cfg))
        self.threshold, self.smooth, self.reset_after = threshold, smooth, reset_after

    def enroll(self, raw: np.ndarray) -> np.ndarray:
        """`raw`: ~60 s de muestras crudas del Polar (sin los primeros 5 s). Devuelve la huella (128,)."""
        q = check_quality(raw, FS)
        if not q.ok:
            raise ValueError("Registro con mala señal: " + " ".join(q.issues))
        return make_template(self.embed, signal_to_windows(raw, FS, self.cfg))

    def new_monitor(self, template: np.ndarray) -> "Monitor":
        return Monitor(self, template)


class Monitor:
    """Una instancia por usuario conectado; llamar `step` cada ~2.5 s."""

    def __init__(self, engine: AuthEngine, template: np.ndarray):
        self.e, self.template = engine, template
        self.decider, self.bad_run = Decider(engine.threshold, engine.smooth), 0

    def step(self, last_15s: np.ndarray) -> dict:
        """`last_15s`: las últimas 1950 muestras crudas (en µV). Devuelve un dict listo para JSON."""
        if len(last_15s) < BUFFER:
            return {"status": "WARMING_UP", "score": None, "smoothed": None}
        raw = np.asarray(last_15s[-BUFFER:], dtype=np.float64)
        windows = signal_to_windows(raw, FS, self.e.cfg)
        if len(windows) == 0 or not check_quality(raw[-2 * 650:], FS, min_peaks=5).ok:
            self.bad_run += 1
            if self.bad_run >= self.e.reset_after:
                self.decider.reset()
            return {"status": BAD_SIGNAL, "score": None, "smoothed": None}
        self.bad_run = 0
        score = score_window(self.e.embed, self.template, windows[-1])
        smoothed, status = self.decider.update(score)
        return {"status": status, "score": score, "smoothed": smoothed, "authenticated": status == AUTH}
```

Valores de `status` (cadenas de `src/live.py`): `"AUTENTICADO"`, `"NO AUTENTICADO"`, `"SEÑAL MALA"`, más
`"WARMING_UP"` del ejemplo. El campo `authenticated` solo existe cuando hubo decisión.

Uso típico:

```python
engine = AuthEngine("models/modelo_ecg.pt")      # cargar UNA vez al arrancar el servidor
template = engine.enroll(muestras_60s)           # np.ndarray (128,); guardar por usuario
monitor = engine.new_monitor(template)           # uno por usuario conectado
resultado = monitor.step(ultimas_1950_muestras)  # cada ~2.5 s
```

Guardar/cargar la huella: `np.savez(path, template=template, checkpoint=nombre_modelo)` y al cargar
**verificar que `checkpoint` coincide** con el modelo en uso; si no, pedir un nuevo registro (las huellas de
modelos distintos no son comparables).

### Endpoints sugeridos

| Método | Ruta | Función |
|---|---|---|
| POST | `/enroll/start` | Inicia el registro de un usuario (5 s de espera + 60 s) |
| GET | `/enroll/status` | Progreso y resultado del registro (o error de señal mala) |
| POST | `/monitor/start` | Empieza la vigilancia |
| WS | `/monitor/stream` | Empuja `{t, status, score, smoothed}` cada ~2.5 s |
| POST | `/monitor/stop` | Termina |

---

## 7. Reglas de decisión (ya implementadas; no las cambies sin hablar con el equipo)

- **Cada 2.5 s** se evalúa la ventana más reciente de 5 s (50 % de traslape).
- **Promedio de las últimas 6 ventanas** (~15 s): una ventana mala no bloquea al usuario legítimo.
- **Umbral actual: 0.31** (similitud coseno), calculado con las grabaciones del Polar existentes. Es
  provisional: se recalibrará con el modelo definitivo. **Haz que sea configurable.** Lo que se observó en pruebas:
  - Usuario legítimo: similitud ~0.90–0.99.
  - Otra persona: ~−0.10 a 0.30.
- **Señal mala = sin decisión** (ni acepta ni rechaza). Se detecta con `check_quality`.
- **Reinicio tras pausa:** con 2 pasos seguidos de señal mala (~5 s) se borra el promedio acumulado. Esto
  evita que alguien que se ponga la banda después herede el promedio del usuario anterior. No lo quites.
- Un impostor puede tardar hasta ~7 s en ser rechazado (por el promedio). Es normal.
- Si dejan de llegar muestras ~15 s, tratarlo como conexión perdida.

La interfaz debería mostrar tres estados distintos: autenticado, no autenticado y "sin señal / señal mala".

---

## 8. Limitaciones conocidas (pónlas en la interfaz o la documentación)

- **Se entrenó con ~15 personas** y pocas sesiones por persona. Con usuarios nuevos desconocidos el error
  puede ser mayor que el ~7 % medido con personas ya vistas.
- El umbral se calibró con personas que el modelo ya conocía.
- El rendimiento no es parejo entre personas: una de las personas del dataset quedó casi al azar y otras dos
  con ~8–9 % de error.
- Sensible a movimiento y mal contacto: la banda debe ir bien húmeda y ajustada.
- Con el modelo actual el registro y la prueba del mismo día son la versión fácil; entre días es más difícil.
- El modelo es de un solo canal (la derivación del Polar), no de ECG clínico.

---

## 9. Privacidad y datos

- El **ECG y las huellas son datos biométricos.** No los subas al repo. Están ignorados en `.gitignore`
  (`data/raw/`, `data/processed/`, `data/live/`, `data/live/`); respeta eso. Los modelos sí se versionan.
- La huella (`template`, 128 números) es el dato sensible mínimo: guárdala cifrada o fuera del repo, por
  usuario, con consentimiento, y permite borrarla.
- No guardes el ECG crudo de los usuarios de la web salvo que sea necesario y con consentimiento explícito.
- Los participantes de la recolección tienen ID anónimo (`P01`, `P02`, ...). No intentes ligarlos a personas.

---

## 10. Pendientes del equipo del modelo (pueden cambiar cosas)

- Elegir el modelo definitivo y copiarlo a `models/modelo_ecg.pt`.
- Recalibrar el umbral con el modelo final.
- Evaluar con personas nuevas entre días (faltan sesiones de las personas reservadas).
- Mejorar el requirements (`torch` no aparece).

Para dudas del modelo o de los datos, pregunta al equipo antes de modificar `src/`.
