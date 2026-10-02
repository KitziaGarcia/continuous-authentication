"""Excepciones propias: mensajes claros en vez de errores crípticos de numpy/torch."""


class DataError(ValueError):
    """Los datos no cumplen el contrato (forma, NaN, sesiones insuficientes...)."""


class LeakageError(RuntimeError):
    """Una persona reservada (hold-out) apareció donde no debía: invalida la evaluación."""


class EvaluationError(ValueError):
    """No hay suficientes datos o pares para evaluar de forma válida."""


class CheckpointError(Exception):
    """El checkpoint no existe o no es compatible con la configuración actual.

    Hereda de Exception y NO de RuntimeError a propósito: load_checkpoint captura los
    RuntimeError de PyTorch (tamaños distintos) y no debe capturar este error por accidente.
    """
