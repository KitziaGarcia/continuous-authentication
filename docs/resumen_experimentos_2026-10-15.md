# Resumen de experimentos — 15 de octubre de 2026

Estado del proyecto de autenticación continua con ECG (Polar H10) después de la primera corrida completa
del proceso: preentrenamiento, entrenamiento, validación y prueba en vivo.

> **Aviso importante:** todos los números de este documento salen de pocos datos (15 personas de
> entrenamiento y casi una sesión de entrenamiento por persona). Sirven para saber si el sistema funciona,
> **no** para citarse como resultado final de la tesis.

---

## 1. Datos disponibles

| Conjunto | Contenido |
|---|---|
| ECG-ID (público, PhysioNet) | 310 grabaciones de 90 personas, ya convertidas a ventanas |
| Polar H10 (propio) | 18 personas con datos procesados (P01–P14, P16–P19); P15 aún sin datos |

Las señales se filtran (0.5–40 Hz), se cortan en **ventanas de 5 s** (650 muestras a 130 Hz, traslape 50 %)
y cada ventana se normaliza con z-score. La red convierte cada ventana en un **embedding** (vector de
128 números) que funciona como "huella" del latido.

### Sesiones por persona al momento de la corrida (con la 2ª sesión de P02 ya grabada)

| Sesiones | Personas |
|---|---|
| 3 | P01, P03, P05, P07 |
| 2 | P02, P04, P06, P08, P09, P10, P11, P12, P16, P17, P18 |
| 1 | P13, P14, P19 |
| 0 | P15 (solo hay archivos descartados) |

---

## 2. Reparto de datos usado

El código separa **siempre por persona y por sesión**, nunca por ventana: ventanas vecinas se parecen mucho
y mezclarlas entre entrenamiento y prueba inflaría los resultados.

| Grupo | Personas | Uso |
|---|---|---|
| Entrenamiento | P01–P12, P16–P18 (15 personas), todas sus sesiones excepto la última | 2 128 ventanas |
| Validación | La **última sesión** de esas mismas 15 personas (otro día) | 1 680 ventanas |
| Hold-out (nunca entrenan) | **P13, P14** | Prueba con personas no vistas |
| Fuera de esta corrida | P19 | Solo tiene 1 sesión; entra a entrenamiento cuando tenga la 2ª |

- `HOLDOUT_IDS = ("P13", "P14")` en `src/config.py`.
- Los hold-out se eligieron por tener una sola sesión, no al azar como indica el comentario original de
  `config.py`. No se eligieron mirando resultados, pero conviene mencionarlo en la tesis.

---

## 3. Preentrenamiento (ECG-ID)

**Objetivo:** que la red aprenda "cómo mirar un ECG" con muchas personas antes de ver las nuestras.

- Datos: ECG-ID, 90 personas; el 10 % de las personas se aparta para validar (separado por persona).
- Entrenamiento tipo clasificación de personas (pérdida con *label smoothing* 0.1), AdamW, lr 1e-3,
  hasta 30 épocas con parada anticipada (paciencia 5).
- Validación: EER con personas que la red no vio, protocolo "misma sesión".

**Resultado:** EER de validación **2.8 %**, el entrenamiento paró en la época **14**.
Guardado como `models/modelo_v1.pt`.

Notas:
- ECG-ID es un dataset fácil (otro equipo, otras derivaciones). Ese 2.8 % **no** predice el desempeño
  con el Polar.
- Se corrió una segunda vez por error y salió idéntico (mismos datos, misma semilla); esa copia se borró.

---

## 4. Entrenamiento con datos Polar

Se entrenaron dos modelos con el mismo reparto de datos.

| Modelo | Archivo | Cómo se entrenó |
|---|---|---|
| **Preentrenado + fine-tuning** | `modelo_v2` | Parte de `modelo_v1`. Etapa A: capas tempranas congeladas, 5 épocas, lr 1e-3. Etapa B: todo descongelado, 20 épocas, lr 1e-4. |
| **Solo Polar** | `modelo_v3` | Desde cero, solo con datos Polar: 40 épocas, lr 1e-3. |

Ambos usan aumento de datos solo en entrenamiento (ruido, deformación temporal y de amplitud, deriva de
línea base), parada anticipada con paciencia 5 y se quedan con la mejor época según la EER de validación.

**EER de validación al elegir la mejor época:** fine-tuning 7.5 %, solo Polar 6.6 %.

---

## 5. Validación y resultados

### Cómo se mide

- **Registro:** la primera sesión de cada persona; su plantilla es el promedio de sus embeddings.
- **Prueba:** su última sesión (otro día). Cada ventana se compara por similitud coseno contra la plantilla
  de **todas** las personas: contra la propia es un intento genuino, contra las demás es un impostor.
- **EER:** porcentaje de error en el punto donde aceptar a un impostor y rechazar al dueño ocurren con la
  misma frecuencia. Menos es mejor; 50 % es azar. **AUC:** 1.0 es perfecto, 0.5 es azar.
- Como línea base se usa la **señal cruda** sin red, para comprobar que el entrenamiento aporta algo.

### Resultados

| Modelo | Prueba | EER | IC 95 % | AUC |
|---|---|---|---|---|
| Solo Polar | Vistos, entre sesiones (val) | **6.6 %** | 0.7–17.5 % | 0.981 |
| Preentrenado + fine-tuning | Vistos, entre sesiones (val) | **7.5 %** | 2.4–17.8 % | 0.967 |
| Señal cruda (sin red) | Vistos, entre sesiones (val) | 49.6 % | 48.0–51.0 % | 0.499 |
| Solo Polar | Vistos, misma sesión | 0.6 % | 0.0–1.4 % | 1.000 |
| Preentrenado + fine-tuning | Vistos, misma sesión | 0.5 % | 0.0–1.0 % | 1.000 |
| Solo Polar | **NO vistos (P13, P14), misma sesión** | 0.0 % | n/a (<3 personas) | 1.000 |
| Preentrenado + fine-tuning | **NO vistos (P13, P14), misma sesión** | 0.0 % | n/a (<3 personas) | 1.000 |
| Señal cruda | NO vistos (P13, P14), misma sesión | 54.5 % | n/a | 0.479 |

### Cómo leerlo

1. **La red aprendió una huella:** entre días baja el error de 50 % (azar) a 6–8 %.
2. **Los dos modelos empatan:** los intervalos de confianza se traslapan por completo. Con tan pocas
   personas no se puede decir que el preentrenamiento ayude o no. Se vuelve a comparar con todas las sesiones.
3. **El hold-out es la versión fácil:** P13 y P14 tienen una sola sesión, así que se evaluó con "misma
   sesión" (primera mitad registra, segunda prueba), con 2 personas y 112 pares, sin intervalo de confianza.
   Solo indica que las huellas de personas nuevas se separan; **no es** una cifra citable. La evaluación
   entre días requiere su 2ª sesión.

### EER por persona (preentrenado + fine-tuning, entre sesiones)

| Persona | EER | Persona | EER |
|---|---|---|---|
| P01 | 0.9 % | P09 | **42.8 %** |
| P02 | 8.2 % | P10 | 0.8 % |
| P03 | 0.8 % | P11 | 0.0 % |
| P04 | 1.0 % | P12 | 1.1 % |
| P05 | 0.0 % | P16 | 9.0 % |
| P06 | 0.0 % | P17 | 0.8 % |
| P07 | 0.0 % | P18 | 0.0 % |
| P08 | 0.0 % | | |

11 de 15 personas quedan con 1 % de error o menos. **P09 (42.8 %)** es casi azar y es el principal
responsable de que el promedio no sea mejor; **no se ha investigado** la causa (queda pendiente).
P02 y P16 (8–9 %) también destacan.

### Comparación con la literatura

Solo se revisaron resúmenes; los protocolos y datasets son distintos, así que no son comparaciones directas.

| Escenario | EER | Fuente |
|---|---|---|
| Entre sesiones, dataset CYBHi (equipo tipo wearable) | ~10 % | [Bento 2019](https://run.unl.pt/bitstream/10362/75491/1/Bento_2019.pdf) |
| PTB, una sola sesión / varias sesiones | 0.14 % / 2.06 % | [Melzi et al.](https://www.arxiv.org/abs/2204.03992) |
| Red siamesa (embeddings), ECG-ID | 1.29 % | [EDITH](https://qspace.qu.edu.qa/handle/10576/30597) |
| Banda de pecho de un canal, autenticación continua | 0 intrusiones en 33 personas (SVM; no reporta EER) | [Smyth et al.](https://arxiv.org/pdf/2105.05126) |
| Pulso en smartband (no es ECG), uso real | 4.4 % | [arXiv 1912.04760](https://arxiv.org/pdf/1912.04760) |

El 6.6–7.5 % entre días queda en el orden de magnitud de lo publicado para el escenario difícil
(entre sesiones con equipo wearable). No se encontró ningún trabajo con Polar H10 y redes de embeddings
(solo dos búsquedas).

---

## 6. Prueba en vivo (simulación del sistema final)

Se creó `src/live.py`: registra la huella del usuario (60 s con la banda puesta) y después decide cada
2.5 s si la persona sigue siendo la misma, promediando las últimas 6 ventanas (~15 s). Imprime
`AUTENTICADO` / `NO AUTENTICADO` en consola. Umbral usado: **0.31** (calculado con las grabaciones Polar
existentes). Modelo: el último guardado por defecto (`modelo_v3`, solo Polar).

### Verificación previa con grabaciones reales (P13 vs. P14)

| Quién | Decisiones | Rechazos |
|---|---|---|
| P13, la persona registrada | 80 | **0** |
| P14, como impostor | 89 | **78** |

Los 11 aceptados de P14 ocurren justo al cambiar de persona, por el promedio de ventanas anteriores.

### Resultados de tu prueba en vivo (tú y tu hermano, ambos presentes en el dataset)

| Tramo | Qué pasó | Similitud | Decisión |
|---|---|---|---|
| 00:02 – 01:00 | Tú, con la banda puesta (25 decisiones) | 0.79 – 1.00 (casi siempre ≥ 0.9) | AUTENTICADO en todas |
| 01:02 – 01:30 | Cambio de persona | — | "Sin decisión" (señal mala, 12 pasos) |
| 01:33 – 02:08 | Tu hermano (15 decisiones) | −0.10 a −0.01 (una de 0.35) | 12 `NO AUTENTICADO`; **3 aceptadas por error** |
| 02:10 – 02:33 | Cambio de persona | — | "Sin decisión" |
| 02:36 – 03:08 | Tú otra vez | 0.93 – 0.98 | 1 rechazo inicial indebido, después AUTENTICADO |

**Lectura:**
- **La separación es muy clara:** tú ≈ 0.95–0.99 y tu hermano ≈ −0.08, lejísimos del umbral 0.31.
- **Defecto encontrado:** durante las pausas de señal el programa conservaba los puntajes viejos, así que
  (a) tu hermano fue aceptado 3 veces al inicio con "promedio 0.81" aunque su similitud real era −0.08, y
  (b) a ti te rechazó un paso al volver, por el promedio de tu hermano.
- **Corrección aplicada después de tu prueba:** si hay 2 pasos seguidos sin señal buena (~5 s), se borra el
  promedio acumulado. Está cubierta por una prueba automática; **falta repetir la prueba con la banda real**.

### Limitaciones de esta prueba

- Tú y tu hermano están en el entrenamiento: la red fue entrenada explícitamente para distinguirlos. Con
  una persona desconocida (el caso real de un usuario nuevo) el resultado puede ser peor.
- Registro y vigilancia fueron del mismo día y la misma colocación de banda: es la versión fácil.
- El umbral 0.31 salió de personas que el modelo ya conocía; para alguien nuevo puede quedar desajustado.
  Se puede cambiar con `--threshold`.

---

## 7. Pendientes

1. Tomar las sesiones faltantes (ver plan de reparto en la sección 8).
2. Evaluar el hold-out **entre días** (P13, P14 y P15 necesitan su 2ª sesión).
3. Reentrenar con el reparto final y volver a comparar fine-tuning vs. solo Polar.
4. Investigar la señal de P09 (EER 42.8 %).
5. Probar el modo en vivo con alguien que **no** esté en el dataset, y repetir la prueba de quitarse y
   ponerse la banda entre el registro y la vigilancia.
6. Medir la autenticación continua con ventanas promediadas y el desglose por actividad (reposo, lectura,
   hablando, problemas), que aún no se han calculado de forma sistemática.

## 8. Plan de reparto final propuesto

Sesiones finales previstas por persona: 3 sesiones → P01–P07 y P10 (P02, P04, P06 y P10 llegan a 3);
2 sesiones → P08, P09, P11, P12, P16, P17, P18, P19; 1 sesión → P13, P14, P15.

- **Entrenamiento:** las 2 primeras sesiones de quienes tengan 2 o más.
- **Validación A (misma persona, sesión nueva):** la 3ª sesión de quienes tengan 3.
- **Validación B / prueba de personas nuevas:** P13, P14 y P15 completos.
- Para la cifra final, validación cruzada dejando una sesión fuera por rondas, o un grupo de personas
  reservado como prueba intocable. Conviene que el reparto viva en una tabla `(persona, sesión) →
  train/val/test`; `split_polar` actual solo usa "la última sesión de cada persona" como validación.

---

## 9. Cambios al código en esta etapa

| Archivo | Cambio |
|---|---|
| `src/config.py` | `HOLDOUT_IDS = ("P13", "P14")` |
| `src/evaluate.py` | Si algún hold-out tiene una sola sesión, evalúa con "misma sesión" y lo indica en la fila |
| `src/live.py` (nuevo) | Autenticación continua en vivo por consola |
| `tests/test_evaluate.py`, `tests/test_live.py` (nuevo) | Pruebas de lo anterior; la suite completa pasa (222) |
| `.gitignore` | Se agregó `data/live/` (ahí se guarda la huella biométrica) |

Modelos en `models/`: `modelo_v1` (preentrenado), `modelo_v2` (fine-tuning), `modelo_v3` (solo Polar).
