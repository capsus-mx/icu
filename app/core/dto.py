
import numpy as np
import rasterio
import pandas as pd 
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class SimulationResult:
    # Datos Raster (Numpy Arrays)
    base_lst: np.ndarray
    intervention_lst: np.ndarray
    difference_lst: np.ndarray
    base_icu_classes: np.ndarray
    icu_classes: np.ndarray
    
    # Metadatos Geoespaciales
    bbox: List[float]
    profile: Dict[str, Any]
    
    # Capas Adicionales
    ozone_twin: np.ndarray
    ozone_intervention: np.ndarray
    population_sampled: np.ndarray
    overall_intervention_mask_sampled: np.ndarray
    modified_features: Dict[str, np.ndarray]
    
    # Resultados de Impacto Energético
    energy_summaries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Datos Vectoriales y Estadísticas
    population_zonal_stats_df: pd.DataFrame = None
    
    # Información de Ubicación
    municipality_name: str = "N/A"
    state_name: str = "N/A"
    
    # KPIs resumidos
    kpis: Dict[str, Any] = field(default_factory=dict)