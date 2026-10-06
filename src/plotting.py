"""Guardado de imágenes que no falla si el archivo está abierto en el visor (Windows)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg


def save_figure(fig, path, dpi: int = 110) -> Path:
    """Guarda `fig` como PNG y devuelve la ruta realmente usada.

    En Windows no se puede reemplazar un archivo que otro programa tiene abierto (p. ej. la imagen
    del intento anterior en el visor). En ese caso no se pierde la nueva imagen: se guarda con otro
    nombre (`<nombre>_<hora>.png`) y se avisa; el archivo original no se toca.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)       # backend propio: no toca el backend global de matplotlib
    try:
        fig.savefig(str(path), dpi=dpi)
        return path
    except OSError:
        fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%H%M%S_%f')}{path.suffix}")
        fig.savefig(str(fallback), dpi=dpi)
        print(f"AVISO: {path.name} está en uso (¿abierta en el visor?); se guardó como {fallback.name}")
        return fallback
