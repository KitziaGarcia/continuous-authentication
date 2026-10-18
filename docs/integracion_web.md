# Guía de integración: cómo conectar el frontend con el backend de autenticación continua

**Para quién es:** quien mantenga el frontend y el backend de la plataforma web.
El frontend vive en `web/` (React, Vite y Tailwind); este documento define el contrato HTTP/WebSocket que utiliza.

> **Estado: prototipo de investigación (tesis).** El error medido entre días con personas ya vistas es de ~7 %
> (EER). **No es un sistema de seguridad real**; no lo presentes como tal en la interfaz.

---

## 1. Idea general

```
Polar H10 --Bluetooth--> [Backend Python (este repo)] --HTTP + WebSocket (JSON)--> [Tu frontend]
```

- El **backend** lee la banda, corre el modelo y decide. El frontend **no** procesa ECG: solo manda órdenes
  (HTTP) y recibe eventos (WebSocket).
- El backend debe correr en la **misma máquina que tiene Bluetooth** y a la que está emparejada la banda.
- Hay **una sola banda y una sola operación a la vez** (un registro o una vigilancia). Si mandas otra mientras
  hay una en curso, responde `409 BUSY`.

## 2. Cómo correr el backend

Desde la raíz del repo (Python 3.10+; probado en 3.14):

```bash
python -m venv .venv && .venv\Scripts\activate            # Windows; en Linux/Mac: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # torch NO está en requirements.txt
pip install -r requirements.txt
```

| Comando | Para qué |
|---|---|
| `python -m src.api --simulate` | **Desarrollo del frontend sin banda.** Señal sintética que tú controlas (ver §7). |
| `python -m src.api --simulate --speed 20` | Igual, pero el tiempo corre 20 veces más rápido (el registro de 65 s dura ~3 s). |
| `python -m src.api` | Con la banda Polar H10 real (Bluetooth, banda puesta y húmeda). |

Opciones: `--host` (por defecto `127.0.0.1`), `--port` (8000), `--checkpoint models/modelo_X.pt` (por defecto
`models/modelo_ecg.pt` o el último `modelo_vN.pt`), `--threshold` (0.31), `--cors-origin http://localhost:5173`
(repetible; por defecto cualquier origen puede llamar, cómodo para desarrollo).

El servidor queda en `http://127.0.0.1:8000`. Documentación interactiva de las rutas HTTP: `/docs`; esquema
OpenAPI: `/openapi.json` (el WebSocket **no** aparece ahí: está en §5).

Los modelos entrenados están en `models/` (los datos de ECG **no** están en el repo). Las huellas de los
usuarios se guardan en `data/live/<user_id>.npz` (ignorado por git).

---

## 3. Flujo de la aplicación

```
                 POST /enroll/start                    POST /monitor/start
   [sin huella] ───────────────────► [registrando 65 s] ─────────────────► [vigilando]
        ▲                                   │ falla (señal mala)                 │
        │                                   ▼                                    │ POST /stop
        └──────────── reintentar ◄── enroll_result ok:false                      ▼
                                                                            [idle, banda cerrada]
```

1. Al cargar la página: abre el WebSocket (`/ws`) y comprueba `GET /health`.
2. Botón **"Tomar muestra inicial"** → `POST /enroll/start`. Dura **65 s** (5 s de estabilización + 60 s de
   registro). Escucha `enroll_progress` y al final `enroll_result`.
3. La aplicación solicita entre 5, 10, 20 o 30 preguntas del nivel y dificultad elegidos con
   `GET /exam/questions` y comienza el examen junto con
   `POST /monitor/start`. Llegan eventos `decision` cada ~2.5 s.
4. Tres rechazos consecutivos pausan el examen. El usuario puede ajustar la banda y pedir una muestra
   de verificación con `POST /monitor/recheck`; esa muestra se compara con la plantilla existente y nunca
   la reemplaza. Si no coincide, el frontend detiene el examen.
5. La calidad de señal baja genera avisos de colocación y pausa la interacción con las preguntas, pero
   **no** cuenta como rechazo de identidad. La señal debe recuperarse antes de continuar.
6. Al terminar, la aplicación envía las respuestas a `POST /exam/grade`; la calificación se calcula en el
   backend. El banco y sus claves viven en `data/exams/questions.json`, sin base de datos.
7. `POST /stop` cancela la vigilancia y cierra la conexión con la banda.

Mientras corre una operación, deshabilita los botones (usa `phase`, ver §5). Tras un registro o una vigilancia
la conexión con la banda queda abierta (`connected: true`) para poder encadenar la siguiente sin reconectar;
`POST /stop` la cierra.

---

## 4. Rutas HTTP

Todas devuelven JSON. Los `user_id` solo admiten letras, números y guiones (1 a 40 caracteres); otro valor
devuelve `422`.

| Método y ruta | Cuerpo | Respuesta |
|---|---|---|
| `GET /health` | — | `{"ok": true, "mode": "simulate" \| "polar", "model": "modelo_v3.pt", "threshold": 0.31}` |
| `GET /state` | — | El estado completo (igual que el evento `state`, sin `type`). Sirve para **sondeo** si no usas WebSocket. |
| `GET /users/{user_id}` | — | `{"user_id": "ana", "enrolled": true}` |
| `DELETE /users/{user_id}` | — | `{"user_id": "ana", "deleted": true}` (borra la huella y el contador de fallos) |
| `POST /enroll/start` | `{"user_id": "ana"}` | `202 {"accepted": true, "user_id": "ana"}`. El resultado llega **por WebSocket** o en `/state`. |
| `POST /monitor/start` | `{"user_id": "ana"}` | `202 {"accepted": true, "user_id": "ana"}` |
| `POST /monitor/recheck` | `{"user_id": "ana"}` | `202`; el resultado llega como `recheck_result` por WebSocket |
| `GET /exam/questions?level=secundaria&difficulty=facil&count=10` | — | Preguntas sin clave de respuesta; `count` acepta 5, 10, 20 o 30 y la API rechaza solicitudes que excedan el banco filtrado |
| `POST /exam/grade` | `{"answers": [{"question_id": "s01", "answer": "B"}]}` | Calificación y retroalimentación al finalizar |
| `POST /exam/metrics` | `{"outcome": "completed", "readings": [{"time_seconds": 12.5, "status": "AUTHENTICATED", "wearer_label": "original"}]}` | Guarda una fila CSV por decisión, con etiqueta de portador, verdad, tiempo, pregunta y similitud |
| `POST /stop` | — | `{"phase": "idle"}` |
| `POST /sim/scenario` | `{"scenario": "..."}` | Solo con `--simulate`. Ver §7. |

**Errores** (formato `{"detail": {"code": "...", "message": "..."}}`):

| Código HTTP | `code` | Cuándo |
|---|---|---|
| 404 | `NOT_ENROLLED` | `monitor/start` para alguien sin huella: hay que registrar primero |
| 409 | `BUSY` | Ya hay un registro o vigilancia en curso (si acaba de terminar uno, espera un instante y reintenta) |
| 409 | `TEMPLATE_MODEL_MISMATCH` | La huella es de otro modelo (se cambió el modelo): hay que registrar de nuevo |
| 422 | (lista de FastAPI) | `user_id` inválido o cuerpo mal formado; `detail` es una lista, no un objeto |

Los errores que ocurren **durante** una operación (banda no encontrada, conexión perdida) no son respuestas HTTP:
llegan como evento `error` por WebSocket y quedan en `last_error` de `/state`.

### Exportación local de métricas

Al completar el examen, la interfaz envía las lecturas a `POST /exam/metrics` con `outcome: "completed"`.
Una expulsión por baja coincidencia o la salida manual las guarda con `outcome: "interrupted"`.
El backend crea un CSV independiente en `metrics/examenes_completados/` o
`metrics/examenes_interrumpidos/`. Cada fila incluye un identificador aleatorio de sesión y la etiqueta
elegida por el operador: `original` se guarda como `genuine`, `impostor` como `impostor` y `unknown`
como `unknown`. «Original» significa la misma persona que aportó la muestra de enrolamiento; no hace
falta escribir ni recordar IDs de participantes en la interfaz. Durante el examen, el operador elige
«participante original», «otra persona» o «no confirmado». La etiqueta inicia en `unknown` y, al aplicar
el control, se asigna a las lecturas siguientes sin alterar la decisión del modelo. También se guarda el tiempo desde el
inicio de la vigilancia, número/ID/texto de la pregunta visible, similitud instantánea y suavizada, estado
de verificación, umbral operativo y ritmo cardiaco. Las respuestas no se incluyen.
Si el examen termina sin decisiones, igualmente se crea un CSV con estado `NO_READINGS`.

Las métricas son datos sensibles de investigación: permanecen localmente y `metrics/` está excluido de Git.
Las etiquetas describen quién llevaba la banda, no se infieren de la decisión del autenticador. Un cierre
inesperado de la pestaña o del navegador antes de enviar el resultado puede impedir el guardado.

Con datos genuine e impostor etiquetados de forma independiente, se calculan EER, AUC ROC, FAR y FRR en el
umbral operativo, además del tiempo entre la primera lectura marcada impostor y el primer rechazo del modelo.
Los episodios impostores que no recibieron un rechazo antes del final se cuentan como no detectados.
Para obtener estas métricas:

```powershell
.\.venv\Scripts\python.exe -m src.metrics
```

Por defecto analiza `smoothed_similarity`, la misma puntuación que compara el autenticador con su umbral;
se puede usar `--score-column similarity` para puntuaciones sin suavizar en EER/AUC. FAR y FRR resumen las
decisiones observadas (`AUTHENTICATED`/`REJECTED`) y excluyen señal mala. El tiempo de detección depende de
cuándo el operador cambia la etiqueta: aplica el control en cuanto cambia el portador. Ignora `unknown`,
`BAD_SIGNAL` y lecturas sin puntuación para EER/AUC. Las ventanas consecutivas de una persona están
correlacionadas: estas métricas son exploratorias, no ensayos independientes ni una evaluación clínica.

---

## 5. WebSocket `ws://127.0.0.1:8000/ws`

Al conectar, **lo primero que llega es un evento `state`** con la situación actual (así, si recargas la página
a media operación, te recuperas). Después llegan eventos; todos son JSON con un campo `type`.

### `state` — foto del estado (al conectar y en cada cambio de `phase`)

```json
{
  "type": "state",
  "phase": "idle",                 // "idle" | "connecting" | "enrolling" | "monitoring"
  "connected": false,              // ¿hay conexión abierta con la banda?
  "user_id": null,
  "enroll": {
    "stage": null,                 // null | "settling" | "recording"
    "elapsed_s": 0.0, "total_s": 60.0,
    "signal_ok": null,             // null = aún sin medir; true/false = calidad de señal reciente
    "codes": [],                   // problemas de señal actuales (ver §6)
    "consecutive_failures": 0,     // registros fallidos seguidos de ESTE usuario
    "last_failure": null           // {"codes": [...], "messages": [...]} del último registro fallido
  },
  "monitor": null,                 // último evento decision (o status "WARMING_UP" al empezar)
  "last_error": null               // {"code": "...", "message": "..."} o null
}
```

### `enroll_progress` — cada ~1 s durante el registro

```json
{"type": "enroll_progress", "stage": "recording", "elapsed_s": 35.0, "total_s": 60.0,
 "signal_ok": true, "codes": []}
```

`stage` pasa por `"settling"` (5 s, `total_s: 5`) y luego `"recording"` (60 s, `total_s: 60`). Progreso total =
`(stage == "settling" ? elapsed_s : 5 + elapsed_s) / 65`. `signal_ok` es `null` hasta que hay ~10 s de señal y
luego se actualiza cada 2 s: úsalo para avisar **en vivo** ("ponte derecho/a y no te muevas") sin esperar al final.

### `enroll_result` — una vez, al terminar el registro

```json
{"type": "enroll_result", "ok": true,  "user_id": "ana", "consecutive_failures": 0, "codes": [], "messages": []}
{"type": "enroll_result", "ok": false, "user_id": "ana", "consecutive_failures": 3,
 "codes": ["MOVEMENT_ARTIFACT"], "messages": ["Hay movimiento o mal contacto: ponte derecho/a y no te muevas."]}
```

- `ok: true` → la huella quedó guardada; `consecutive_failures` vuelve a 0.
- `ok: false` → **no** se guardó nada. `consecutive_failures` cuenta los fallos seguidos de ese `user_id`
  (se guarda en memoria del servidor: se pierde al reiniciarlo). `messages` son textos en español listos para mostrar.

### `decision` — cada ~2.5 s durante la vigilancia

```json
{"type": "decision", "t": 3.01, "status": "AUTHENTICATED", "authenticated": true,
 "score": 0.987, "smoothed": 0.987, "threshold": 0.31,
 "bad_signal_streak": 0, "rejection_streak": 0, "hr_bpm": 70.3, "codes": []}
```

| Campo | Significado |
|---|---|
| `t` | Segundos desde que empezó la vigilancia |
| `status` | `"AUTHENTICATED"`, `"REJECTED"` o `"BAD_SIGNAL"` (sin decisión) |
| `authenticated` | `true`/`false`, o `null` si `BAD_SIGNAL`. **Es el campo para el semáforo.** |
| `score` | Similitud (coseno, de −1 a 1) de **esta** ventana con la huella; `null` si `BAD_SIGNAL` |
| `smoothed` | Promedio de las últimas 6 ventanas (~15 s). **Con este se decide** (≥ `threshold` ⇒ autenticado) |
| `threshold` | Umbral usado (0.31) |
| `bad_signal_streak` | Decisiones **seguidas** con señal mala. 3 ≈ 7.5 s |
| `rejection_streak` | Decisiones **seguidas** `REJECTED`. 3 ≈ 7.5 s |
| `hr_bpm` | Ritmo cardiaco estimado (puede ser `null`) |
| `codes` | Si `BAD_SIGNAL`, motivo(s) (ver §6); si no, `[]` |

Valores típicos: el usuario legítimo da `score` de 0.90 a 0.99; otra persona, de −0.10 a 0.30. Al cambiar de
persona el rechazo puede tardar hasta ~7 s (por el promedio). Durante los primeros segundos de la vigilancia
`/state` trae `monitor.status = "WARMING_UP"` (hasta ~15 s si la conexión era nueva).

### `error` — algo falló a media operación

```json
{"type": "error", "code": "NO_DEVICE", "message": "No se encontró el sensor. ..."}
```

| `code` | Significa | Qué mostrar |
|---|---|---|
| `NO_DEVICE` | No se encontró la banda tras ~10 s de búsqueda | "Ponte la banda, humedécela y reintenta" |
| `CONNECTION_FAILED` | Bluetooth apagado/sin permisos u otro fallo al conectar | Mostrar `message` |
| `CONNECTION_LOST` | Dejó de llegar señal (~5 s en registro, ~15 s en vigilancia) | "Se perdió la banda" |

Después de un `error` el servidor vuelve a `phase: "idle"` y cierra la conexión con la banda.

### Revalidación y examen

`POST /monitor/recheck` realiza 3 s de estabilización y 10 s de lectura. Envía progreso como
`recheck_progress` y termina con uno de estos eventos:

```json
{"type": "recheck_result", "kind": "verified", "ok": true, "score": 0.97, "threshold": 0.31}
{"type": "recheck_result", "kind": "identity", "ok": false, "score": 0.12, "threshold": 0.31}
{"type": "recheck_result", "kind": "signal", "ok": false, "score": null,
 "codes": ["FLAT_SIGNAL"], "messages": ["No llega señal: revisa la banda."]}
```

`kind: "signal"` permite volver a colocar la banda y reintentar; no se interpreta como impostura.
`kind: "identity"` confirma que la nueva muestra no superó el umbral. Una muestra verificada reanuda
`POST /monitor/start` con la plantilla original.

Los eventos `decision` separan `BAD_SIGNAL` de `REJECTED`. Tras tres ventanas consecutivas `BAD_SIGNAL`,
el frontend muestra un aviso de calidad y desactiva las respuestas mientras no haya señal confiable. Ante
el primer y segundo `REJECTED` consecutivos, muestra un aviso para revisar el ajuste de la banda y el examen
continúa. El tercer `REJECTED` consecutivo finaliza la sesión y muestra el motivo; no se toma una muestra de
revalidación. Las ventanas con señal mala no incrementan el contador de rechazos del modelo.

El API del examen devuelve solo enunciados, opciones y lecturas. La respuesta correcta permanece en
`data/exams/questions.json` y se revela únicamente en el resultado de `POST /exam/grade`.

---

## 6. "¿De dónde saco...?" — recetas para la interfaz

| Quiero... | Lo obtengo de... |
|---|---|
| Botón "tomar muestra inicial" | `POST /enroll/start {"user_id"}` |
| Saber si hay que registrar primero | `GET /users/{id}` → `enrolled` |
| Barra de progreso del registro | `enroll_progress` (ver fórmula en §5) |
| Aviso en vivo "ponte derecho/a y no te muevas" durante el registro | `enroll_progress.signal_ok === false` (y `codes`) |
| **Pop-up "falló 3 veces"** | `enroll_result.ok === false && enroll_result.consecutive_failures >= 3` (también en `state.enroll.consecutive_failures`) |
| Texto a mostrar tras un fallo | `enroll_result.messages` (o traducir `codes` con la tabla de abajo) |
| Semáforo verde/rojo/gris | `decision.authenticated` → `true` / `false` / `null` |
| Aviso "se te está rechazando" | `decision.rejection_streak >= 3` |
| Aviso "ajusta la banda" en vigilancia | `decision.bad_signal_streak >= 3` (motivo en `codes`) |
| Mostrar ritmo cardiaco | `decision.hr_bpm` |
| Medidor / gráfica de similitud | `decision.smoothed` frente a `decision.threshold` |
| Banda desconectada | evento `error` con `CONNECTION_LOST` / `NO_DEVICE` |
| Bloquear botones mientras trabaja | `state.phase !== "idle"` |

Los números "3" de los avisos son decisión del frontend: el backend solo entrega los contadores.

### Códigos de problema de señal (`codes`)

| `code` | Significado | Mensaje en español (viene en `messages`) |
|---|---|---|
| `FLAT_SIGNAL` | No llega señal (banda floja/seca) | No llega señal: revisa que la banda esté puesta y con los electrodos húmedos. |
| `INVALID_VALUES` | Valores no válidos | La señal trae valores no válidos: ajusta la banda y repite. |
| `MOVEMENT_ARTIFACT` | Picos por movimiento o mal contacto | Hay movimiento o mal contacto: ponte derecho/a y no te muevas. |
| `FEW_BEATS` | Casi no se detectan latidos | Casi no se detectan latidos: ajusta la banda y humedece los electrodos. |
| `IMPOSSIBLE_HEART_RATE` | Ritmo imposible (fuera de 40–180 lpm) | Se detecta un ritmo cardiaco imposible: probablemente mal contacto. |
| `IRREGULAR_RHYTHM` | Ritmo muy irregular | El ritmo es muy irregular: quédate quieto/a y respira normal. |
| `SIGNAL_GAP` | Hueco de más de 2 s sin latidos | Se perdió la señal unos segundos: la banda se despegó o se movió. |
| `POOR_SIGNAL` | Señal mala sin motivo específico | La señal no es buena: ajusta la banda y quédate quieto/a. |

Puede venir más de un código a la vez. Usa los `codes` como identificadores estables; los textos pueden cambiar.

---

## 7. Desarrollar sin banda: modo simulado

Con `--simulate` el backend genera un ECG sintético y puedes **provocar situaciones** para probar la interfaz:

`POST /sim/scenario` con `{"scenario": "<nombre>"}` — cambia lo que "pasa" desde ese instante:

| `scenario` | Efecto | Lo que verás |
|---|---|---|
| `same_person` | La misma persona del registro (por defecto) | `AUTHENTICATED`, `score` ~0.98 |
| `other_person` | Otra persona se pone la banda | Tras unos segundos `REJECTED`, `score` ~0.2, sube `rejection_streak` |
| `bad_signal` | Banda floja (señal plana) | `BAD_SIGNAL` con `codes: ["FLAT_SIGNAL"]`; si se pone **antes** de un registro, el registro falla |
| `movement` | Picos de movimiento | `BAD_SIGNAL` con `MOVEMENT_ARTIFACT` |

Receta para probar el pop-up de "3 fallos": `POST /sim/scenario {"scenario":"bad_signal"}` y llama a
`POST /enroll/start` tres veces (con `--speed 20` cada registro dura segundos).

Esta señal es **sintética**; los puntajes reales con la banda serán distintos (ver §9). El modo simulado sirve
para probar la interfaz, no para evaluar el modelo.

---

## 8. Ejemplo completo de uso (HTTP + WebSocket, Python)

```python
import json, httpx
from websockets.sync.client import connect

B = "http://127.0.0.1:8000"
with connect("ws://127.0.0.1:8000/ws") as ws:
    print(json.loads(ws.recv())["phase"])                    # estado inicial: "idle"
    httpx.post(B + "/enroll/start", json={"user_id": "ana"})
    while (ev := json.loads(ws.recv()))["type"] != "enroll_result":
        pass                                                  # aquí actualizas la barra con enroll_progress
    print(ev["ok"], ev["consecutive_failures"])
    while json.loads(ws.recv()).get("phase") != "idle":      # espera el state que cierra la operación
        pass
    httpx.post(B + "/monitor/start", json={"user_id": "ana"})
    for _ in range(10):
        ev = json.loads(ws.recv())
        if ev["type"] == "decision":
            print(ev["status"], ev["smoothed"], ev["rejection_streak"])
    httpx.post(B + "/stop")
```

En JavaScript el WebSocket es el estándar del navegador: `new WebSocket("ws://127.0.0.1:8000/ws")`,
`onmessage = e => JSON.parse(e.data)`.

---

## 9. Lo que el frontend debe tener presente (limitaciones)

- Entrenado con ~15 personas y pocas sesiones: con usuarios nuevos el error puede ser mayor que el ~7 % medido
  con personas ya vistas. El umbral (0.31) es provisional y se recalibrará con el modelo final.
- El rendimiento no es parejo entre personas (una del dataset quedó casi al azar).
- Es sensible a movimiento y mal contacto: la banda debe ir bien húmeda y ajustada.
- Un impostor puede tardar hasta ~7 s en ser rechazado; tras una pausa de señal (≥ 2 decisiones malas, ~5 s)
  se reinicia el promedio para que nadie herede la sesión del usuario anterior.
- Una operación a la vez, una sola banda.
- Si el backend se reinicia se pierde el contador `consecutive_failures` (las huellas sí persisten).

## 10. Privacidad

- El ECG y las huellas son **datos biométricos**. No subas al repo `data/raw/`, `data/processed/` ni `data/live/`
  (están en `.gitignore`). Los **modelos entrenados sí** se versionan en `models/`.
- La huella (128 números) es el dato sensible mínimo: guárdala por usuario, con consentimiento, y ofrece
  borrarla (`DELETE /users/{id}`). El backend no guarda el ECG crudo.
- La web genera una clave aleatoria `EX-...` para cada sesión; no solicita nombre ni correo. La calificación
  y las respuestas se mantienen en memoria del navegador y no se persisten en el backend.
- Los participantes de la recolección tienen ID anónimo (`P01`, `P02`...). No intentes ligarlos a personas.

## 11. Para quien mantenga el backend

- Implementación de referencia: `src/api.py` (servidor), `src/live.py` (lógica de decisión: huella, promedio de
  ventanas, reinicio), `src/session.py::PolarSource` (lectura de la banda), `src/quality.py` (calidad de señal).
- **Contrato de la señal (no cambiar):** 130 Hz, muestras crudas en µV, ventana de 650 muestras (5 s), filtro
  pasa-banda 0.5–40 Hz de fase cero, z-score por ventana, embedding de 128 números normalizado a longitud 1;
  similitud = producto punto. Todo el preprocesamiento está en `src/windowing.py::signal_to_windows`: úsalo, no lo
  reescribas. Para cada decisión se filtran las últimas 1950 muestras (15 s) y se evalúa la última ventana.
- La huella guardada depende del modelo (`data/live/<id>.npz` incluye el nombre del `.pt`); si cambias de
  modelo hay que registrar de nuevo (el backend lo avisa con `TEMPLATE_MODEL_MISMATCH`).
- Pruebas: `python -m pytest tests/test_api.py tests/test_exam.py` (usa modo simulado acelerado, no necesita banda).
