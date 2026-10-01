# Proyecto: Autenticación biométrica continua por ECG

Este documento es el brief del proyecto de carrera. Úsalo como guía para desarrollar el sistema completo. Somos dos estudiantes de Ingeniería en Computación que vamos a **supervisar** el desarrollo y luego **explicarlo y defenderlo**, así que todo debe ser entendible, modular y bien documentado. Preferimos una solución clara y sólida sobre una "super compleja" que no podamos explicar.

---

## 1. Contexto

- Trabajo previo del equipo: un sistema embebido (ESP32-S3 + circuito analógico propio con INA128P y filtros 0.5–45 Hz) que identificaba a 5 personas con una 1D-CNN pequeña (clasificación cerrada con softmax, entrada de 2500 muestras a 500 Hz, TensorFlow Lite). Este proyecto **no** reutiliza ese hardware ni ese enfoque de clasificación.
- Ahora **no hay restricción de sistema embebido**: el procesamiento corre en una laptop.
- Hardware de adquisición: **Polar H10** (banda de pecho, una derivación, ECG a **130 Hz**, vía Bluetooth Low Energy).
- Equipo de cómputo: laptop con 16 GB de RAM, CPU de 6 núcleos, sin GPU dedicada. Google Colab disponible como respaldo si el entrenamiento fuera lento.

## 2. Objetivo

Una aplicación web de examen donde el alumno se autentica con su ECG y **el sistema sigue verificando continuamente** que sea la misma persona durante todo el examen. Si detecta que la persona cambió (por ejemplo, alguien más se puso la banda a medio examen) o la señal se pierde, el examen se bloquea.

Justificación: la autenticación inicial puede falsificarse o "pasarse" a otra persona después; la verificación continua cierra ese hueco. Aplicable a cualquier acceso a información sensible.

## 3. Decisión de diseño clave: verificación con embeddings, no clasificación

**No usar clasificación cerrada con softmax como sistema final.** Problemas:
- Softmax siempre asigna una de las clases conocidas, incluso a un impostor que nunca vio.
- Registrar a un usuario nuevo obligaría a reentrenar la red.

**Usar aprendizaje de embeddings (metric learning):**
- La red convierte una ventana de ECG en un vector (embedding) de 64–128 dimensiones.
- Se entrena para que los vectores de la misma persona queden cerca y los de personas distintas queden lejos.
- La comparación se hace con **similitud coseno** entre vectores.
- Debe funcionar con personas que la red **nunca vio** durante el entrenamiento.

### Las tres fases del sistema

| Fase | ¿Se entrena la red? | Qué pasa | Cuándo |
|---|---|---|---|
| Entrenamiento | Sí | La red aprende a convertir ECG en vectores útiles. Se guarda en `models/modelo_ecg.pt` | Durante el desarrollo |
| Registro (enrollment) | No (red congelada) | El usuario graba ~1–2 min, se calculan sus vectores y se guarda el promedio como plantilla en la base de datos | Cada usuario nuevo |
| Autenticación continua | No (red congelada) | Cada ventana nueva pasa por la red y su vector se compara con la plantilla | Durante el examen |

Separación importante a mantener en el código:
- **La red (pesos en `.pt`)** = el conocimiento de "cómo mirar" un ECG. No cambia al registrar o borrar usuarios.
- **La base de datos de plantillas** = la "memoria" de cada persona registrada.
- **La lógica de decisión** = umbral y reglas de bloqueo.

## 4. El combo ganador

| Pieza | Elección |
|---|---|
| Adquisición | Polar H10 vía BLE desde un backend en Python (`bleak`) |
| Preprocesamiento | Pasa-banda 0.5–40 Hz, detección de picos R con NeuroKit2, ventanas de ~5 s con traslape, z-score |
| Modelo | 1D-CNN residual pequeña → embedding de 64–128 dims (PyTorch) |
| Entrenamiento | Clasificador sobre muchas personas, luego usar la penúltima capa como embedding |
| Verificación | Similitud coseno contra plantilla del usuario + umbral elegido en el EER |
| Lógica continua | Promedio móvil de scores, índice de calidad de señal, regla de bloqueo con tolerancia |
| Baseline | Rasgos fiduciales + SVM |
| Web | Backend FastAPI con WebSocket hacia el frontend del examen, mostrando estado de autenticación en vivo |

### Detalles de cada pieza

**Adquisición (Polar H10)**
- ECG de una derivación a 130 Hz. El stream de ECG del H10 se activa mediante su servicio PMD (Polar Measurement Data) por BLE; investigar el protocolo y comandos necesarios para iniciar el stream desde Python con `bleak`.
- Guardar grabaciones crudas con metadatos (participante, sesión, fecha, actividad).
- Detectar y reportar desconexiones.

**Preprocesamiento**
- Filtro pasa-banda ~0.5–40 Hz (Nyquist = 65 Hz).
- Detección de picos R con NeuroKit2 (sirve también para el índice de calidad de señal).
- Ventanas de ~5 s (~650 muestras) con traslape (por ejemplo 50%).
- Normalización z-score por ventana.
- Representación principal: **señal 1D filtrada**. No usar Gramian Angular Field ni espectrogramas como enfoque principal (más pesados y sin ventaja clara a 130 Hz con una derivación).

**Modelo**
- 1D-CNN con bloques residuales, 4–6 capas convolucionales, terminando en un embedding de 64–128 dimensiones normalizado (L2).
- Estrategia de entrenamiento principal: entrenar como clasificador sobre todas las personas de entrenamiento y usar la penúltima capa como embedding. Alternativas a evaluar si hay tiempo: triplet loss o ArcFace.
- Debe entrenar en CPU en un tiempo razonable (minutos a pocas horas).
- Guardar checkpoints versionados (`modelo_v1.pt`, `modelo_v2.pt`, ...).

**Verificación y lógica continua**
- Plantilla del usuario = promedio de los embeddings de su registro.
- Score por ventana = similitud coseno con la plantilla.
- Umbral elegido en el punto de EER sobre el conjunto de validación.
- **Suavizado temporal:** promedio móvil de scores o regla tipo "N de las últimas M ventanas por debajo del umbral" antes de bloquear, para no bloquear por una sola ventana mala (tos, movimiento).
- **Índice de calidad de señal:** si no hay picos R plausibles o la frecuencia cardiaca es imposible, marcar la ventana como "no confiable", no como "impostor".
- **Pérdida de señal / desconexión:** pausar o bloquear el examen.
- Parámetros (umbral, N, M, tamaño de ventana, traslape) configurables en un solo archivo de configuración.
- Extensión opcional: **actualización adaptativa de plantilla** (mezclar poco a poco embeddings con alta confianza en la plantilla) sin reentrenar la red.

**Baseline**
- Rasgos fiduciales (amplitudes e intervalos: QRS, QT, PR, RR, etc.) + SVM con scikit-learn, para comparar contra la red.
- Experimento adicional: verificación con similitud directa sobre la señal cruda vs. sobre los embeddings, para demostrar qué aporta el entrenamiento.

**Web**
- Backend FastAPI que recibe el stream del Polar, procesa ventanas y ejecuta la verificación.
- WebSocket hacia el frontend para mostrar en vivo: estado (autenticado / dudoso / bloqueado), score actual y gráfica de la señal.
- Flujo de la página: registro → inicio de examen → monitoreo continuo → bloqueo si falla.
- Puede ser un examen sencillo de prueba; lo importante es el mecanismo de autenticación.

## 5. Datos

El cuello de botella del proyecto. Requisitos:
- Reclutar idealmente **20–30 participantes**, con consentimiento informado.
- **Varias sesiones por persona en días distintos** (la variabilidad entre sesiones, por colocación de la banda y estado fisiológico, es el reto real).
- Incluir condiciones realistas de examen: sentado en reposo, leyendo, resolviendo problemas, hablando.
- **Particiones por persona, no por ventana:** el conjunto de prueba debe incluir personas que la red nunca vio en entrenamiento, para evaluar impostores desconocidos.
- Evaluar también **entre sesiones** (registro con un día, prueba con otro día) y reportarlo honestamente.
- Opcional: preentrenar con bases públicas (por ejemplo ECG-ID de PhysioNet o Heartprint) remuestreadas a 130 Hz y luego ajustar con datos del Polar. Considerar que son derivaciones distintas a la banda de pecho.
- Aumento de datos opcional solo en entrenamiento (jitter, escalado, magnitude/time warping).

## 6. Evaluación y métricas

- **EER** (Equal Error Rate) y curva **ROC**.
- **FAR** y **FRR** en el umbral elegido.
- **Tiempo hasta detección:** experimento donde la persona A se registra e inicia el examen y a la mitad le pasa la banda a la persona B; medir cuántos segundos tarda el sistema en bloquear.
- **Tasa de falsos bloqueos** del usuario legítimo durante un examen completo.
- Comparativas: red vs. baseline SVM vs. señal cruda; misma sesión vs. entre sesiones; con y sin suavizado temporal.

## 7. Stack de desarrollo

- **Python** con entorno virtual (`venv` o conda).
- **PyTorch** (modelo y entrenamiento).
- **NeuroKit2, SciPy, NumPy** (procesamiento de señal).
- **scikit-learn** (métricas, baseline SVM).
- **Matplotlib** (gráficas).
- **bleak** (BLE con el Polar H10).
- **FastAPI + WebSockets** (servidor y comunicación en vivo).
- **SQLite** (plantillas de usuarios).
- **VS Code + notebooks de Jupyter** para exploración y explicación.
- **Git + GitHub** para trabajo en equipo.

## 8. Estructura del proyecto propuesta

```
ecg-auth/
├── data/
│   ├── raw/              ← grabaciones del Polar tal cual
│   └── processed/        ← ventanas ya filtradas y normalizadas
├── notebooks/            ← para explorar y entender cada etapa
├── src/
│   ├── acquisition.py    ← conexión al Polar H10
│   ├── preprocessing.py  ← filtros, picos R, ventanas
│   ├── model.py          ← definición de la red
│   ├── train.py          ← entrena y guarda el .pt
│   ├── evaluate.py       ← EER, FAR, FRR, curvas ROC
│   └── decision.py       ← lógica de autenticación continua
├── models/
│   └── modelo_ecg.pt     ← la red entrenada
├── app/                  ← servidor FastAPI + página del examen
└── templates.db          ← vectores de usuarios registrados
```

Flujo de uso:
1. `python src/train.py` → entrena y guarda `models/modelo_ecg.pt`.
2. `python app/...` (servidor) → al arrancar carga el `.pt` una vez y lo usa para registro y autenticación, sin volver a entrenar.

## 9. Cómo queremos trabajar

- **Construir un módulo a la vez**, en este orden sugerido: adquisición → preprocesamiento → modelo y entrenamiento → evaluación → lógica continua → web.
- **Cada módulo con un notebook** que lo muestre funcionando con gráficas: qué entra, qué sale y por qué (señal cruda vs. filtrada, picos R detectados, ventanas, embeddings proyectados en 2D con PCA o t-SNE, distribuciones de scores genuinos vs. impostores, etc.).
- Código comentado en español explicando el **porqué** de cada decisión, no solo el qué.
- Un README por módulo o uno general que explique la arquitectura en lenguaje sencillo.
- Pruebas básicas para preprocesamiento y lógica de decisión (por ejemplo, con señales sintéticas).
- Antes de decisiones importantes de diseño, explicarnos las opciones y el razonamiento.
- Priorizar claridad y reproducibilidad (semillas fijas, configuración centralizada, scripts que se puedan volver a correr).
