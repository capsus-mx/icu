
class CoreBaseException(Exception):
    """
    Clase base para todas las excepciones del Core Científico.
    Permite al Backend capturar cualquier error derivado del motor.
    """
    pass

class ConfigurationError(CoreBaseException):
    """Se levanta cuando faltan variables de entorno o rutas de archivos."""
    pass

class GeometryError(CoreBaseException):
    """
    Se levanta cuando los polígonos (GeoJSON) están fuera del BBox
    o son inválidos geométricamente[cite: 186].
    """
    pass

class DataMismatchError(CoreBaseException):
    """
    Se levanta si las dimensiones de los arrays no coinciden,
    como entre el raster base y la máscara de intervención[cite: 187].
    """
    pass

class ModelError(CoreBaseException):
    """
    Se levanta ante fallos internos dentro del modelo de Machine Learning
    (ej: problemas de inferencia con HistGradientBoosting) [cite: 188].
    """
    pass
