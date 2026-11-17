# Resultados y discusión — Autenticación continua con ECG (Polar H10)

> Borrador para el reporte. Las cifras salen de `docs/resumen_experimentos_2026-10-15.md`, de `models/modelo_v*.json` y de los CSV de `metrics/examenes_completados/` (calculados con `src/metrics.py` y un script de apoyo). Los marcadores **[GRÁFICA]** y **[TABLA]** indican qué armar y con qué datos.

---

## 1. Plataforma de demostración (resumen corto para el reporte)

Para validar el sistema en condiciones cercanas al uso real se construyó una plataforma web:

- **Adquisición:** banda Polar H10 por Bluetooth (biblioteca `bleak`), ECG a 130 Hz.
- **Backend:** Python con FastAPI/Uvicorn; PyTorch para la red; NumPy/SciPy para filtrado y ventaneo.
- **Frontend:** React 18 + Vite + Tailwind CSS.
- **Flujo:** registro de la huella (60 s) → examen de lectura de preguntas durante el cual el sistema decide cada 2.5 s si el usuario sigue siendo el mismo → se guarda un CSV por examen con similitud, similitud suavizada, umbral y decisión.

---

## 2. La red y su entrenamiento

### 2.1 Qué es la red
Red convolucional 1D con bloques residuales (stem + 4 etapas de 32/64/128/128 canales, kernel 5), **372 959 parámetros**. Recibe una ventana de 5 s (650 muestras) filtrada (0.5–40 Hz) y con z-score, y produce un **embedding de 128 números** (la "huella"). La huella de un usuario es el promedio de los embeddings de su registro; se compara con **similitud coseno**.

### 2.2 Datos
| Conjunto | Contenido |
|---|---|
| ECG-ID (PhysioNet) | 310 grabaciones, 90 personas |
| Polar H10 (propio) | 18 personas procesadas (P01–P14, P16–P19), 4 actividades por sesión (reposo, lectura, hablando, problemas), hasta 3 sesiones por persona |

Reparto (siempre por persona y por sesión, nunca por ventana, para no inflar resultados):

| Grupo | Personas | Ventanas |
|---|---|---|
| Entrenamiento | 15 (P01–P12, P16–P18), todas sus sesiones menos la última | 2 128 |
| Validación (otro día) | La última sesión de esas 15 | 1 680 |
| Hold-out (nunca entrenan) | P13, P14 | — |

### 2.3 Modelos entrenados
| Modelo | Cómo se entrenó | EER de validación |
|---|---|---|
| `modelo_v1` | Preentrenamiento en ECG-ID (clasificación, label smoothing 0.1, AdamW, lr 1e-3, paró en época 14) | 2.8 % (ECG-ID, misma sesión; **no** predice el desempeño con Polar) |
| `modelo_v2` | v1 + fine-tuning (A: capas tempranas congeladas, 5 épocas; B: todo descongelado, lr 1e-4) | 7.5 % |
| `modelo_v3` | Solo datos Polar, desde cero, 40 épocas máx., lr 1e-3 (paró en época 17, mejor en la 12) | **6.6 %** |

Aumento de datos solo en entrenamiento: ruido, deformación de amplitud y de tiempo, deriva de línea base.

> ⚠️ **Dato importante sobre el modelo de los exámenes:** si se corre el backend sin `--checkpoint` y no existe `models/modelo_ecg.pt` (hoy no existe), el sistema carga **el último modelo numerado = `modelo_v3` (solo Polar)**. Es decir, **los exámenes se hicieron con el modelo entrenado únicamente con datos Polar, no con el preentrenado en ECG-ID**. En el reporte conviene decir: *"el modelo desplegado fue la variante entrenada solo con el dataset propio, que obtuvo el menor EER de validación (6.6 % vs. 7.5 %)"*. Si en realidad se arrancó con `--checkpoint modelo_v2.pt`, hay que corregirlo; el JSON de la huella guardada en `data/live/*.npz` incluye el campo `checkpoint` y lo confirma (se puede revisar con `np.load(...)["checkpoint"]`).

### 2.4 Resultados de validación offline (otro día)
| Modelo | Prueba | EER | IC 95 % | AUC |
|---|---|---|---|---|
| Solo Polar (v3) | Vistos, entre sesiones | **6.6 %** | 0.7–17.5 | 0.981 |
| Preentrenado + fine-tuning (v2) | Vistos, entre sesiones | 7.5 % | 2.4–17.8 | 0.967 |
| Señal cruda (sin red) | Vistos, entre sesiones | 49.6 % | 48.0–51.0 | 0.499 |
| Solo Polar | Vistos, misma sesión | 0.6 % | 0.0–1.4 | 1.000 |
| Fine-tuning | Vistos, misma sesión | 0.5 % | 0.0–1.0 | 1.000 |
| Ambos | No vistos P13/P14, misma sesión | 0.0 % | n/a (<3 personas) | 1.000 |
| Señal cruda | No vistos P13/P14, misma sesión | 54.5 % | n/a | 0.479 |

**Lectura:**
1. **La red sí aprende una huella:** la señal cruda está en azar (≈50 %); con la red el error entre días baja a 6–8 %.
2. **v2 y v3 empatan estadísticamente** (los intervalos se traslapan). Con 15 personas no se puede afirmar que el preentrenamiento ayude o estorbe.
3. **El caso "misma sesión" es optimista** (0.5 %): ventanas de la misma grabación se parecen mucho. La cifra realista es la de entre sesiones.
4. **EER por persona** (v2, entre sesiones): 11 de 15 personas ≤ 1 %; P02 8.2 %, P16 9.0 % y **P09 42.8 % (≈ azar)**, que es la principal causa de que el promedio no sea mejor. La causa de P09 no se investigó.

**[GRÁFICA 1] Curvas de entrenamiento** (líneas: EER de validación vs. época; una línea por modelo).
- v1 (ECG-ID): 11.1, 11.1, 8.3, 7.8, 8.0, 10.9, 7.8, 8.3, 2.8, 6.2, 8.3, 5.6, 5.6, 2.8 (épocas 1–14, en %).
- v3 (solo Polar): 18.9, 11.7, 8.0, 6.8, 7.1, 6.7, 6.9, 6.7, 6.8, 6.8, 6.7, 6.6, 6.6, 6.7, 6.7, 6.7, 6.7 (épocas 1–17).
- v2, etapa A (épocas 1–5): 8.5, 7.8, 7.9, 7.5, 7.6; etapa B (épocas 1–5): 7.6, 7.7, 7.6, 7.8, 7.5.
- Opcional, segundo eje: pérdida de entrenamiento de v3: 1.48, 0.69, 0.62, 0.60, 0.60, 0.59, 0.59, 0.58, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57.

**[GRÁFICA 2] EER por persona** (barras, v2 entre sesiones): P01 0.9, P02 8.2, P03 0.8, P04 1.0, P05 0.0, P06 0.0, P07 0.0, P08 0.0, P09 42.8, P10 0.8, P11 0.0, P12 1.1, P16 9.0, P17 0.8, P18 0.0 (%).

**[TABLA 1]** La tabla de la sección 2.4 tal cual.

---

## 3. Validación con exámenes en la plataforma (prueba principal)

### 3.1 Procedimiento
- El usuario original se registra 60 s con la banda (mismo día y misma colocación que el examen; no hubo tiempo intermedio).
- Durante el examen se grababa la señal y cada **2.5 s** se calculaba la similitud de la última ventana de 5 s contra la huella; la **decisión usa el promedio de las últimas 6 ventanas** (≈15 s). Umbral operativo: **0.31** (calculado antes con grabaciones Polar de personas conocidas por el modelo).
- Roles: **original** (el registrado) e **impostor** (otra persona con la banda). Hubo impostores que pertenecen al dataset de entrenamiento y otros totalmente desconocidos para el modelo; un mismo individuo nunca fue original e impostor en el mismo examen.
- Las etiquetas (`unknown` / `original` / `impostor`) se asignaron **manualmente**: primero `unknown` al retirar la banda y cambiar de persona, luego `original` o `impostor` al colocarla.
- Solo se usan decisiones `AUTHENTICATED`/`REJECTED` con etiqueta `genuine`/`impostor`. Se excluyen `BAD_SIGNAL` (sin decisión) y `unknown`.
- Se analizan **5 exámenes completados con etiquetas** (2 de ~10 s sin etiquetas y los 17 interrumpidos quedan fuera).

### 3.2 Resultados globales
| Métrica | Valor |
|---|---|
| Exámenes / duración total | 5 / ≈ 1 957 s (32 min 37 s) |
| Decisiones genuinas / impostoras | 221 / 166 |
| Similitud suavizada genuina: media (± desv.) | 0.934 (± 0.080); mediana 0.962; rango 0.521–0.996 |
| Similitud suavizada impostora: media (± desv.) | 0.244 (± 0.143); mediana 0.285; rango −0.072–0.500 |
| **AUC** | **1.000** (0.9999 sin suavizar) |
| **EER** | **0.0 %** (0.6 % sin suavizar) — umbral de cruce ≈ 0.51 |
| **FRR @ 0.31** (operativo) | **0.0 %** (0/221) |
| **FAR @ 0.31** (operativo) | **35.5 %** (59/166; IC95 28.7–43.1 %) |
| Impostores detectados | 6/7 episodios (el restante fue 100 % `BAD_SIGNAL`, sin decisión) |
| Latencia de detección @ 0.31 | media 18.1 s, mediana 12.6 s |

**Interpretación en una frase:** la red separa perfectamente a original e impostor en estos datos (AUC 1.0), pero el umbral 0.31 con el que se hizo la prueba quedó demasiado permisivo y dejó pasar a más de un tercio de las decisiones de impostor.

### 3.3 Por examen
| Examen | Duración (s) | Genuinas (FR) | Impostoras (FA @0.31) | Sim. genuina media | Sim. impostora media | Impostor máx. |
|---|---|---|---|---|---|---|
| 1 `01e6a227` | 681 | 81 (0) | 67 (41) | 0.882 | 0.348 | 0.498 |
| 2 `afd3371d` | 431 | 40 (0) | 40 (17) | 0.980 | 0.298 | 0.500 |
| 3 `d87b90d9` | 391 | 35 (0) | 24 (1) | 0.922 | 0.180 | 0.319 |
| 4 `7dbdf2ca` | 260 | 33 (0) | 18 (0) | 0.970 | −0.047 | −0.028 |
| 5 `bac163a4` | 194 | 32 (0) | 17 (0) | 0.984 | 0.108 | 0.123 |

Hallazgos:
- **Exámenes 3, 4 y 5:** separación enorme; el impostor queda entre −0.05 y 0.32 y casi nunca se acepta.
- **Exámenes 1 y 2 concentran 58 de los 59 falsos positivos.** Sus impostores tienen similitud 0.30–0.35 de media (llegan a 0.50), claramente más altas que en los demás. Esto sugiere que esos impostores (o la combinación original-impostor) tienen un ECG más parecido, o que el registro de ese original fue menos discriminativo. **[Completar:** indica si en esos dos exámenes el impostor era de tu dataset o desconocido, y si había parentesco o cercanía fisiológica].
- El examen 1 es además el más ruidoso (el original baja a 0.63 en algún momento, media 0.882), posiblemente por movimiento o por ser el más largo (11 min); no se verificó.

### 3.4 Barrido de umbral (a posteriori, sobre los mismos datos)
| Umbral | FAR | FRR | Falsos aceptados | Falsos rechazados | Latencia media de detección |
|---|---|---|---|---|---|
| 0.31 (operativo) | 35.5 % | 0.0 % | 59/166 | 0/221 | 18.1 s |
| 0.40 | 10.2 % | 0.0 % | 17/166 | 0/221 | 9.7 s |
| 0.45 | 4.2 % | 0.0 % | 7/166 | 0/221 | 8.8 s |
| **0.50** | **0.0 %** (IC95 0–2.3 %) | **0.0 %** | 0/166 | 0/221 | 8.4 s |
| 0.55 | 0.0 % | 0.5 % | 0/166 | 1/221 | 8.4 s |
| 0.60 | 0.0 % | 0.5 % | 0/166 | 1/221 | 8.4 s |
| 0.70 | 0.0 % | 3.6 % | 0/166 | ~8/221 | — |
| 0.80 | 0.0 % | 7.2 % | 0/166 | ~16/221 | — |
| 0.90 | 0.0 % | 20.8 % | 0/166 | ~46/221 | — |

(Latencia calculada rehaciendo la decisión con el umbral indicado sobre la similitud suavizada guardada.)

**[GRÁFICA 3] Histograma de similitud suavizada, genuinos vs. impostores** (la gráfica más importante; barras agrupadas o dos histogramas superpuestos, con una línea vertical en 0.31 y otra en 0.50). Bins de 0.1:

| Bin (similitud) | Impostor | Genuino |
|---|---|---|
| [−0.1, 0.0) | 18 | 0 |
| [0.0, 0.1) | 4 | 0 |
| [0.1, 0.2) | 33 | 0 |
| [0.2, 0.3) | 43 | 0 |
| [0.3, 0.4) | 51 | 0 |
| [0.4, 0.5) | 17 | 0 |
| [0.5, 0.6) | 0 | 1 |
| [0.6, 0.7) | 0 | 7 |
| [0.7, 0.8) | 0 | 8 |
| [0.8, 0.9) | 0 | 30 |
| [0.9, 1.0] | 0 | 175 |

Se ve que las dos distribuciones **no se traspasan** (hueco de ≈0.02 entre 0.500 y 0.521), pero el margen es estrecho y el umbral 0.31 cae dentro de la nube de impostores.

**[GRÁFICA 4] Curva FAR y FRR vs. umbral** (dos líneas; el cruce está en ≈0.51). Usa la tabla 3.4. Alternativa: curva ROC (con AUC = 1.0 es una L, se ve poco informativa; es mejor la gráfica 3 o 4).

**[GRÁFICA 5] Serie de tiempo de un examen** (la que mejor "cuenta la historia"). Recomiendo el examen 3 (`d87b90d9`) o el 5 (`bac163a4`): eje x = `time_seconds`, eje y = `smoothed_similarity`, color o sombreado por `ground_truth` (genuine/impostor/unknown), línea horizontal en el umbral. Los datos están directo en el CSV (`time_seconds`, `smoothed_similarity`, `ground_truth`, `status`). Para mostrar el problema del umbral, se puede poner también el examen 1 (similitud del impostor hasta 0.46–0.50 sostenida por encima de 0.31).

**[TABLA 2]** Tablas de 3.2, 3.3 y 3.4.

### 3.5 Tiempo de detección de impostores
| Examen | Episodio | Duración del episodio | Detectado en (s) @0.31 | Falsas aceptaciones antes de rechazar |
|---|---|---|---|---|
| 1 | 1.º | 143.7 s | 25.2 | 4 |
| 1 | 2.º | 133.6 s | 55.5 | 7 (y luego solo 1 rechazo en total) |
| 2 | 1.º | 108.5 s | 12.6 | 1 |
| 3 | 1.º | 63.1 s | 5.0 | 0 |
| 4 | 1.º | 65.5 s | sin decisión (todo `BAD_SIGNAL`) | — |
| 4 | 2.º | 42.9 s | 0.0 | 0 |
| 5 | 1.º | 50.3 s | 10.1 | 0 |

Con umbral 0.5 la media baja a ≈ 8.4 s y la mediana a ≈ 10.1 s. El valor típico de 5–10 s es consistente con el diseño: ventana de 5 s + decisión cada 2.5 s + promedio de 6 ventanas. El 0.0 s del examen 4 (2.º episodio) indica que la etiqueta de impostor se puso cuando ya se había cambiado de persona y el promedio ya no recordaba al original (ver limitaciones).

**[GRÁFICA 6, opcional] Barras de latencia de detección por episodio** con los valores de la tabla.

### 3.6 Calidad de señal
De las lecturas etiquetadas, 38 de 259 genuinas (14.7 %) y 82 de 248 impostoras (33.1 %) fueron `BAD_SIGNAL`: el sistema **no decide** con señal mala (ni acepta ni rechaza). Esto es un comportamiento diseñado para evitar falsos rechazos al dueño por movimiento o contacto flojo; el costo es que durante esos tramos no hay verificación. Una porción de las lecturas `BAD_SIGNAL` del impostor se debe a que, al cambiar de persona, la banda se quitaba/reacomodaba antes de que se actualizara la etiqueta.

**[TABLA 3, opcional]** Lecturas por estado: genuinas (221 decididas / 38 señal mala), impostoras (166 / 82).

---

## 4. Discusión

### 4.1 Hallazgos positivos
1. **La red extrae una huella discriminativa.** Frente a la señal cruda (EER ≈ 50 %), el sistema logra EER 6.6 % entre días en validación offline y EER 0 % / AUC 1.0 en los exámenes en vivo.
2. **El original casi nunca es rechazado:** FRR 0/221 (con umbral 0.31) y 1/221 (0.5 %) con umbrales 0.55–0.60. Es importante para autenticación continua, donde bloquear al usuario legítimo frustra su uso.
3. **Se detectan los cambios de usuario en segundos:** latencia mediana ≈ 10 s (8–13 s según el umbral) en 6 de 7 episodios (el restante fue indecidible por falta de señal).
4. **Generaliza a personas no vistas:** algunos impostores fueron totalmente desconocidos para el modelo y aun así se separaron (**[confirmar en qué exámenes]**; en los exámenes 3–5 el impostor máx. fue ≤ 0.32). El hold-out P13/P14 también dio EER 0 % (aunque con evaluación fácil).
5. **Valor de la red frente a no usarla:** señal cruda AUC 0.50 vs. 0.98 (offline) y 1.00 (en vivo).

### 4.2 Hallazgos negativos y limitaciones
1. **El umbral operativo (0.31) fue inadecuado.** FAR 35.5 %. Se calibró con personas que el modelo ya conocía (similitudes más bajas entre personas conocidas, no representativas del caso real). Con 0.5 el FAR bajaría a 0 %, **pero esa cifra se eligió viendo los mismos datos, así que es optimista**: debe validarse en exámenes nuevos antes de citarse como resultado.
2. **Margen estrecho:** el impostor más alto (0.500) y el genuino más bajo (0.521) están a 0.02. Con más gente y más sesiones es esperable que haya traslape y aparezca un EER distinto de cero.
3. **Dos de cinco exámenes concentran los errores** (impostores con similitud 0.30–0.50). No se investigó el motivo (parecido fisiológico, calidad de registro, ruido). Algo análogo ocurre con P09 (EER 42.8 %) en la validación offline.
4. **Muestra muy pequeña:** 5 exámenes, 387 decisiones, pocos originales e impostores distintos. Las decisiones consecutivas están correlacionadas (cada una usa 15 s de señal con 83 % de traslape), por lo que los intervalos de confianza reales son más anchos que los reportados y las métricas son **exploratorias**.
5. **Condiciones favorables:** registro y examen el mismo día, con la misma colocación de la banda, sin cambios de postura grandes. No se evaluó robustez entre días, quitarse/ponerse la banda con tiempo de por medio, ni ejercicio o estrés fuerte.
6. **Etiquetado manual:** pueden existir desfases de unos segundos entre el cambio real de persona y la etiqueta. Esto afecta (a) la latencia medida (p. ej. 0.0 s en el examen 4), (b) el FAR, porque el promedio de 15 s todavía recuerda a la persona anterior justo después del cambio. Algunas veces se retiró la banda sin poner `unknown`; esas lecturas aparecen como `BAD_SIGNAL` y se excluyeron del cálculo.
7. **Preentrenamiento no demostró ventaja:** v2 (7.5 %) vs. v3 (6.6 %) están dentro del intervalo de confianza. Además, el modelo desplegado fue v3 (solo Polar), por lo que **el reporte no puede afirmar que el uso de ECG-ID mejore el sistema**.
8. **Un modelo, una corrida.** El entrenamiento oficial se corrió una vez (semilla 42). No hay repeticiones ni validación cruzada, y los datos de recolección (15 de 19 personas, mayoría con 2 sesiones) todavía están por completarse.
9. **Cobertura intermitente:** un tercio de las lecturas del impostor y 15 % de las del original fueron `BAD_SIGNAL`; durante ellas no hay protección.

### 4.3 Comparación con la literatura (no comparable directamente: protocolos y datasets distintos)
| Escenario | EER | Fuente |
|---|---|---|
| Este trabajo, entre sesiones, validación offline | 6.6 % | — |
| Este trabajo, exámenes en vivo, mismo día (umbral a posteriori) | 0 % (exploratorio) | — |
| Entre sesiones, CYBHi (wearable) | ≈ 10 % | Bento 2019 |
| PTB, una sesión / varias sesiones | 0.14 % / 2.06 % | Melzi et al. |
| Red siamesa, ECG-ID | 1.29 % | EDITH |
| Banda de pecho 1 canal, autenticación continua | 0 intrusiones en 33 personas | Smyth et al. |
| Pulso en smartband | 4.4 % | arXiv 1912.04760 |

El 6.6 % entre días queda en el mismo orden de magnitud que lo publicado para el escenario difícil (entre sesiones con wearable). El resultado en vivo es mejor, pero corresponde al escenario fácil (mismo día, pocas personas), por lo que no es comparable con los de otras sesiones.

### 4.4 Conclusiones sugeridas
1. La red de embeddings entrenada con datos Polar separa al usuario registrado de impostores, tanto offline (EER 6.6 % entre días) como en una prueba en vivo con la plataforma (AUC 1.0, 5 exámenes).
2. Con el umbral inicial 0.31 el sistema es muy permisivo (FAR 35.5 %, FRR 0 %); con un umbral de ≈0.5 los datos sugieren FAR/FRR ≈ 0 % y detección en ≈ 8–10 s, pendiente de confirmarse con exámenes nuevos.
3. Las limitaciones principales son el tamaño de muestra, el carácter exploratorio de las métricas y que no se evaluó entre días ni con condiciones adversas.

### 4.5 Trabajo pendiente (para cerrar la sección)
- Repetir los exámenes con umbral ≈ 0.5 fijado de antemano (esa sería la validación limpia del umbral).
- Completar las sesiones faltantes y reentrenar con el reparto final; repetir la comparación v2 vs. v3.
- Examinar a P09 y a los impostores de los exámenes 1–2 para entender por qué son más parecidos.
- Probar registro y examen en días distintos, y con actividad física.

---

## 5. Resumen de gráficas y tablas a armar
| # | Qué | Datos |
|---|---|---|
| T1 | Resultados de validación offline | Sección 2.4 |
| T2 | Resultados globales, por examen y por umbral | Secciones 3.2, 3.3, 3.4 |
| T3 | Detección por episodio | Sección 3.5 |
| G1 | Curvas de entrenamiento (EER vs. época) | Sección 2.4 |
| G2 | EER por persona | Sección 2.4 |
| **G3** | **Histograma de similitud original vs. impostor** | **Sección 3.4 (la más importante)** |
| G4 | FAR y FRR vs. umbral | Tabla 3.4 |
| G5 | Serie de tiempo de un examen | CSV `d87b90d9` o `bac163a4` |
| G6 | Latencia de detección por episodio (opcional) | Tabla 3.5 |

## 6. Cosas por confirmar antes de entregar
- Qué modelo cargó realmente el backend en los exámenes (esperado: `modelo_v3`; revisar `checkpoint` en `data/live/*.npz`).
- Qué impostor/original fue del dataset y cuál desconocido en cada examen (en especial los exámenes 1 y 2).
- Un párrafo sobre ética/consentimiento de los voluntarios y que los datos ECG no se publican.
