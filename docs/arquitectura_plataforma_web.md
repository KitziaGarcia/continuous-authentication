# Arquitectura y flujo de la plataforma web

## Propósito

La plataforma ofrece un examen simulado de comprensión lectora y demuestra el flujo de autenticación
continua por ECG descrito en el proyecto. Es un prototipo académico, no un sistema de seguridad ni una
herramienta para decisiones de alto impacto.

## Stack

- **Frontend:** React, Vite y Tailwind CSS; interfaz adaptable con tema claro/oscuro.
- **Backend:** FastAPI, reutilizando la adquisición BLE, el modelo y la decisión continua existentes.
- **Tiempo real:** WebSocket `/ws` para progreso, señal, eventos de autenticación y errores.
- **Preguntas:** documento `data/exams/questions.json`, leído por el backend. No hay base de datos.
- **Plantillas:** archivos biométricos `.npz` en `data/live/`; fuera del control de versiones.

El backend debe ejecutarse en el equipo que tiene Bluetooth y acceso al Polar H10. El navegador no procesa
ECG ni recibe la clave de respuestas.

## Componentes

```text
Polar H10 ─BLE─> FastAPI / modelo / verificación
                         │ HTTP + WebSocket
                         ▼
                 React + Tailwind
                ├── Configuración del examen
                ├── Registro inicial (enrollment)
                ├── Examen de lectura
                ├── Avisos y revalidación
                └── Resultado de la simulación
```

`src/api.py` conserva las rutas existentes de salud, estado, registro, vigilancia, borrado de huella y
simulación. Se añadieron las rutas para preguntas y calificación, y una revalidación que compara una muestra
nueva contra la plantilla original. La web consume este contrato en `web/src/api.js`.

## Flujo principal

1. El estudiante elige nivel (secundaria, preparatoria, universidad o mixto), dificultad y número de preguntas.
2. La página genera una clave de sesión aleatoria `EX-...`; no solicita nombre ni correo.
3. Antes del examen, se coloca la banda, se revisa su ajuste y se inicia el registro. El backend estabiliza
   la señal por 5 segundos y obtiene una muestra de 60 segundos.
4. El backend calcula una plantilla temporal desde los embeddings del registro. El frontend no recibe ni
   recalcula la señal.
5. El backend entrega lecturas y preguntas sin respuestas correctas; durante el examen comunica decisiones
   en vivo por WebSocket.
6. Al terminar, el backend califica las respuestas y la página muestra el porcentaje.
7. Si la señal se pierde o la sesión se detiene por sospecha de cambio de persona, el examen no se califica.

## Avisos, rechazo y bloqueo

La interfaz distingue entre **calidad de señal** y **coincidencia de identidad**. Una lectura deficiente no
es evidencia de que cambió la persona.

### Señal deficiente

- Un evento `BAD_SIGNAL` no incrementa el contador de rechazos del modelo.
- La interfaz desactiva la interacción con las preguntas mientras la señal no es confiable.
- Después de tres lecturas malas consecutivas aparece un aviso (por ejemplo, humedecer y ajustar la banda,
  sentarse derecho y permanecer quieto).
- Cuando vuelve una lectura utilizable, desaparece el aviso y la actividad puede continuar.
- Si se pierde la conexión, se pausa la sesión y se pide revisar Bluetooth y la banda.

### Rechazo biométrico

- Los scores se suavizan en el backend; una sola ventana bajo el umbral no expulsa a nadie.
- La configuración permite elegir entre dos modos:
  - **Modo prueba** (predeterminado): muestra alertas en el primer y segundo rechazo; el tercero finaliza
    la sesión y las respuestas no se califican.
  - **Modo métricas**: sigue examinando la señal tras rechazos, incluso cuando cambia el portador, para
    registrar su evolución sin detener el examen. El operador debe actualizar la etiqueta de portador para
    que las lecturas guardadas en CSV queden correctamente anotadas.
- Ambos modos conservan el manejo separado de `BAD_SIGNAL`; el modo métricas no desactiva los controles de
  calidad de señal ni las alertas de desconexión.

## Banco de preguntas

El JSON del servidor contiene lecturas, preguntas, opciones y respuestas correctas. `GET /exam/questions`
selecciona el nivel, dificultad y cantidad, y omite las claves. `POST /exam/grade` recibe las respuestas
finales y devuelve el resumen. No se guardan intentos ni calificaciones en una base persistente.

El banco incluido contiene preguntas originales de demostración: 30 preguntas por dificultad
(fácil, intermedia y avanzada) para cada nivel (secundaria, preparatoria y universidad), distribuidas
en lecturas breves. La interfaz permite elegir 5, 10, 20 o 30 preguntas del nivel y dificultad elegidos.
Las respuestas no contestadas se califican como
incorrectas, sin sustituirlas por una opción arbitraria.

## Datos y límites

- La clave de sesión es seudónima; las plantillas ECG siguen siendo datos biométricos sensibles.
- La plantilla se escribe en `data/live/` y puede borrarse con `DELETE /users/{user_id}`.
- No se persisten nombre, correo, respuestas, resultados ni ECG crudo para esta demostración.
- El proceso admite una sola banda y una operación biométrica a la vez.
- La precisión experimental aún es preliminar; no se debe presentar el semáforo como una certeza.
- La revalidación es una comprobación de una nueva muestra, no un segundo enrolamiento.

## Directorios relevantes

- `web/`: aplicación React y configuración de Vite/Tailwind.
- `src/api.py`: servidor HTTP/WebSocket y controlador de la banda.
- `src/exam.py`: lectura, selección y calificación del banco JSON.
- `data/exams/questions.json`: contenido de demostración y claves privadas del examen.
- `docs/comandos_web.md`: instalación, ejecución y pruebas locales.
