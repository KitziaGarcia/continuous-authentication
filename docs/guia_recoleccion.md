# Guía de recolección de datos con el Polar H10

Esta guía es para ustedes y para los voluntarios. Si solo recuerdan una cosa:

> **El ID es de la persona y nunca cambia (`P01`, `P02`, ...). Lo que cambia entre grabaciones es el día.**
> El programa pone la fecha y el nombre de los archivos solo. Ustedes nunca escriben ni cambian nombres de archivos.

## 1. Una sola vez por persona

1. **Asignen un ID**: `P01`, `P02`, ... en orden. Anótenlo.
2. **Consentimiento informado**: la persona debe saber que se graba su ECG, para qué se usa y que puede retirarse. Es un dato biométrico.
3. **Tabla "ID → persona"**: guárdenla aparte (un archivo personal, **fuera del repositorio**). En los datos solo aparece `P03`, nunca un nombre.

## 2. Cuántas veces viene cada persona

**3 sesiones, en días distintos, separadas por 2 o 3 días**, quitándose y poniéndose la banda de nuevo cada vez.
Cada sesión dura unos 5.5 minutos. En total son unos 17 minutos por persona.

Para ver a quién le toca volver:

```
python -m src.status
```

Muestra una tabla con los días grabados de cada persona, los minutos y un estado como "toca ya la sesión 2" o "esperar hasta 2026-10-11".

## 3. Antes de cada sesión

- **Electrodos bien húmedos** (agua o gel) y banda ajustada, justo bajo el pecho. Si están secos, la señal sale plana o con ruido.
- **Cierren cualquier otra aplicación que use la banda** (Polar Beat, el celular, etc.): el H10 no siempre acepta dos conexiones a la vez.
- **Persona sentada**, derecha, con la laptop a menos de 2 metros.
- Activen el Bluetooth de la laptop.

## 4. Grabar una sesión

```
python -m src.session --id P03
```

El programa los guía con instrucciones en pantalla y cuenta regresiva:

| Actividad | Tiempo | Qué hacer |
|---|---|---|
| reposo | 2 min | Sentado, respirando normal, sin moverse ni hablar |
| lectura | 1.5 min | Leer en silencio cualquier texto |
| problemas | 1 min | Restar 7 de cabeza a partir de 100 |
| hablando | 0.5 min | Hablar con normalidad, sin moverse mucho |

Antes de empezar hay 10 segundos para que la señal se estabilice, y 5 segundos de aviso antes de cada actividad. El programa descarta esos segundos solo.

Al terminar cada actividad dice **OK** o **REPETIR**, con el motivo (por ejemplo "Pico de amplitud anormal", "Hueco sin latidos"). Al final abre una imagen con las 4 señales: tienen que verse latidos regulares con picos altos.

## 5. Si algo sale mal

| Qué pasó | Qué hacer |
|---|---|
| Una actividad dijo **REPETIR** | `python -m src.session --id P03 --redo reposo` (puede ir más de una: `--redo reposo lectura`). Lo anterior se mueve a `data/descartadas/`, nunca se borra. |
| Se interrumpió a la mitad | Corran el mismo comando otra vez: graba solo lo que falta. |
| Dice que ya completó la sesión de hoy | Es normal si ya terminó. Si una actividad salió mal, usen `--redo`. Si se quitó y se volvió a poner la banda y quieren otra sentada ese mismo día, `--another-sitting`. |
| `ID inválido` | El ID debe ser `P` mayúscula y número: `P03`, no `p3` ni `Juan`. |
| `No se encontró el sensor` | Humedezcan los electrodos, póngansela bien y revisen que ninguna otra app la esté usando. |
| `Se perdió la conexión con la banda` | Acerquen la laptop, revisen la banda y repitan la sesión. Lo ya grabado queda guardado. |

## 6. Qué no hacer

- **No cambien el ID** entre días (`P03` siempre `P03`, nunca `P03b` ni `P033`).
- **No renombren ni muevan archivos** de `data/processed/polar/` ni de `data/raw/polar/`.
- **No suban los datos a git**: las carpetas `data/` ya están ignoradas, y hay una prueba automática que lo vigila.
- **No graben dos personas con el mismo ID** ni una persona con dos IDs.

## 7. Dónde queda todo

| Qué | Dónde |
|---|---|
| Señal lista para el modelo (una por actividad) | `data/processed/polar/P03__2026-10-06__reposo.npz` |
| Señal cruda original | `data/raw/polar/P03__2026-10-06__reposo.npy` |
| Imagen de la sesión | `reports/recordings/P03__2026-10-06.png` |
| Grabaciones malas o reemplazadas | `data/descartadas/` |
| Las pruebas de Juan y Kitzia | `data/piloto/` (no se usan en el experimento) |

## 8. Después de la recolección

Cuando haya al menos 8 personas con 3 días grabados (idealmente 15), el siguiente paso es:

1. **Elegir al azar** quiénes quedan apartadas (hold-out, de 3 a 5 personas), **antes** de entrenar con datos del Polar, y anotarlas en `HOLDOUT_IDS` en `src/config.py`. No se cambia después.
2. Entrenar con `python -m src.train finetune --from-checkpoint models/modelo_v1.pt` y comparar contra entrenar solo con Polar (`python -m src.evaluate compare ...`).

Detalles en el README y en `docs/superpowers/specs/`.
