# Comandos útiles: plataforma web

Los comandos siguientes están pensados para PowerShell en Windows y se ejecutan desde la raíz del repositorio,
salvo que se indique `web/`.

## Requisitos

- Python 3.10 o posterior (el proyecto se ha ejecutado con Python 3.14).
- Node.js y npm.
- Para probar en modo real: Polar H10 emparejada por Bluetooth, con electrodos húmedos y colocada.

## Instalación inicial de Python

Crear y activar un entorno virtual:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar PyTorch para CPU y las dependencias del proyecto:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Si PowerShell bloquea la activación, ejecutar el Python del entorno directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Instalar el frontend

La primera vez, desde la raíz:

```powershell
Set-Location web
npm install
```

Esto crea `web/package-lock.json`; las siguientes instalaciones pueden usar `npm ci`.

## Arranque recomendado: simulación sin banda

Abrir **dos terminales** en la raíz del proyecto.

**Terminal 1 — FastAPI, señal sintética acelerada:**

```powershell
.\.venv\Scripts\python.exe -m src.api --simulate --speed 20 --cors-origin http://127.0.0.1:5173
```

El backend queda en `http://127.0.0.1:8000`. La captura de registro y la revalidación se aceleran para
probar el flujo; esto no sirve para evaluar la red ni sustituye una prueba con la banda real.

**Terminal 2 — React/Vite:**

```powershell
Set-Location web
npm run dev
```

Abrir [http://localhost:5173](http://localhost:5173) o [http://127.0.0.1:5173](http://127.0.0.1:5173).
El backend permite ambos nombres locales para el puerto indicado en `--cors-origin`. Para iniciar, elegir
un examen, registrar la muestra y empezar.

## Arranque con Polar H10 real

En una terminal, desde la raíz, iniciar el backend **en el mismo equipo que tiene Bluetooth**:

```powershell
.\.venv\Scripts\python.exe -m src.api --cors-origin http://127.0.0.1:5173
```

En otra terminal:

```powershell
Set-Location web
npm run dev
```

Abrir [http://localhost:5173](http://localhost:5173) o [http://127.0.0.1:5173](http://127.0.0.1:5173),
confirmar que el backend aparezca conectado y seguir las indicaciones para ajustar la banda. El backend usa
`models/modelo_ecg.pt` o el último checkpoint
versionado disponible; se puede seleccionar uno explícitamente:

```powershell
.\.venv\Scripts\python.exe -m src.api --checkpoint models/modelo_v3.pt --cors-origin http://127.0.0.1:5173
```

## Probar escenarios del simulador

Con el backend iniciado con `--simulate`, cambiar la señal en cualquier momento:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/sim/scenario -Method Post -ContentType "application/json" -Body '{"scenario":"same_person"}'
```

Escenarios disponibles:

```text
same_person   señal de la persona registrada
other_person  simula el cambio a otra persona
bad_signal    simula señal plana / banda sin buen contacto
movement      simula artefacto por movimiento
```

Por ejemplo, probar los avisos de calidad:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/sim/scenario -Method Post -ContentType "application/json" -Body '{"scenario":"bad_signal"}'
```

Para continuar con una lectura buena:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/sim/scenario -Method Post -ContentType "application/json" -Body '{"scenario":"same_person"}'
```

La simulación permite comprobar interfaz y transiciones. Los valores sintéticos no son evidencia del
desempeño biométrico del modelo.

## Verificaciones

Desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py tests\test_exam.py
```

Construir el frontend para comprobar que compila:

```powershell
Set-Location web
npm run build
```

Ver el contrato de FastAPI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## URL alternativa del API

Si FastAPI corre en otra dirección, definir `VITE_API_URL` antes de levantar Vite:

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Hay que permitir ese origen con `--cors-origin` al arrancar FastAPI. El frontend espera el WebSocket en
`/ws` en la misma dirección base del API.
