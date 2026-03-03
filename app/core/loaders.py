
import os
import joblib
import rasterio
import pandas as pd
import numpy as np
from threading import Lock
import logging
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS

from app.core import exceptions # Added import

logger = logging.getLogger(__name__)

class GlobalDataLoader:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(GlobalDataLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_dir="./data", target_crs: str = None, model_path: str = None):
        if hasattr(self, 'initialized') and self.target_crs == target_crs:
            return

        self.data_dir = data_dir
        self.model_path = model_path if model_path else os.path.join("./models", "heat_island_v1.joblib")
        self.ml_model = None
        self.target_crs = target_crs

        self.data_dir = data_dir or os.getenv("ICU_DATA_DIR") #- ICU_DATA_DIR=/app/data
        self.model_path = model_path or os.getenv("ICU_MODEL_PATH") # - ICU_MODEL_PATH=/app/models/heat_v1.joblib

        if not self.data_dir or not self.model_path:
            raise exceptions.ConfigurationError(
                "Faltan rutas de datos. Define ICU_DATA_DIR e ICU_MODEL_PATH "
                "como variables de entorno o pásalas al constructor."
            )

        logger.info(f"[Loader] Cargando datos desde: {self.data_dir}")
        self.target_crs = target_crs
        self.ml_model = None

        self.raster_paths = {
            'LST': os.path.join(data_dir, 'city_rasters/lst.tif'),
            'NDVI': os.path.join(data_dir, 'city_rasters/ndvi.tif'),
            'Albedo': os.path.join(data_dir, 'city_rasters/albedo.tif'),
            'MNDWI': os.path.join(data_dir, 'city_rasters/mndwi.tif'),
            'BSI': os.path.join(data_dir, 'city_rasters/bsi.tif'),
            'Building_Height': os.path.join(data_dir, 'city_rasters/height.tif'),
            'Nighttime_Lights': os.path.join(data_dir, 'city_rasters/lights.tif'),
            'Elevation': os.path.join(data_dir, 'city_rasters/dem.tif'),
            'Built_Surface': os.path.join(data_dir, 'city_rasters/builtup.tif'),
            'Land_Cover': os.path.join(data_dir, 'city_rasters/land_cover.tif'),
            'Population': os.path.join(data_dir, 'city_rasters/population.tif'),
        }

        self._load_model()

        self.energy_df = None
        self.thresholds_df = None
        self.municipalities_df = None
        self.all_intervention_scenarios = {}
        self._load_core_tabular_data()
        self._load_intervention_cs_data()

        self.initialized = True

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.ml_model = joblib.load(self.model_path)
        else:
            # Modified to raise ModelError
            raise exceptions.ModelError(f"❌ Model not found at {self.model_path}")

    def _load_core_tabular_data(self):
        energy_path = os.path.join(self.data_dir, "csv", "electricidad_per_capita.csv")
        thresh_path = os.path.join(self.data_dir, "csv", "temperature_thresholds.csv")
        municipalities_path = os.path.join(self.data_dir, "csv", "municipios_mexico.csv")

        if os.path.exists(energy_path):
            self.energy_df = pd.read_csv(energy_path)
            self.energy_df.columns = self.energy_df.columns.str.replace(' ', '').str.strip()
            logger.info(f"   Loaded energy data from {energy_path}")
        else:
            # Modified to raise DataMismatchError
            raise exceptions.DataMismatchError(f"❌ Energy data CSV not found at {energy_path}") from None

        if os.path.exists(thresh_path):
            self.thresholds_df = pd.read_csv(thresh_path)
            self.thresholds_df.columns = self.thresholds_df.columns.str.replace(' ', '').str.strip()
            logger.info(f"   Loaded temperature thresholds from {thresh_path}")
        else:
            # Modified to raise DataMismatchError
            raise exceptions.DataMismatchError(f"❌ Temperature thresholds CSV not found at {thresh_path}") from None

        if os.path.exists(municipalities_path):
            self.municipalities_df = pd.read_csv(municipalities_path, encoding='utf-8')
            self.municipalities_df['CVEGEO'] = self.municipalities_df['CVEGEO'].astype(str).str.zfill(5)
            logger.info(f"   Loaded municipalities data from {municipalities_path}")
        else:
            logger.warning(f"⚠️ WARNING: Municipalities data CSV not found at {municipalities_path}. Municipality names may not resolve.")
    # ... (dentro de la clase GlobalDataLoader)

    def clear_cache(self):
        """
        Libera los recursos cargados en memoria y permite una re-inicialización fresca.
        Esencial para actualizar datos de origen sin reiniciar el servicio.
        """
        with self._lock:
            logger.info("[Loader] Limpiando caché de datos globales y liberando memoria...")
            
            # 1. Eliminar referencias a objetos pesados
            self.ml_model = None
            
            # Atributos de datos (rasters y dataframes)
            # Asumiendo los nombres de atributos que ya manejas en tu clase
            if hasattr(self, 'all_intervention_scenarios'):
                self.all_intervention_scenarios = None
            
            # Si tienes cargados DataFrames de energía o municipios
            self.energy_df = None
            self.thresholds_df = None
            self.municipalities_df = None
            
            # 2. Forzar limpieza de estado para permitir re-init
            if hasattr(self, 'initialized'):
                delattr(self, 'initialized')
            
            # 3. Llamar al recolector de basura de Python
            import gc
            gc.collect()
            
            logger.info("[Loader] Caché liberada exitosamente.")

    def reload_data(self, **kwargs):
        """
        Método de conveniencia para limpiar y recargar en un solo paso.
        """
        self.clear_cache()
        self.__init__(**kwargs)
        logger.info("[Loader] Datos recargados con nueva configuración.")

    def _load_intervention_cs_data(self):
        intervention_csv_filenames = {
            'Vialidad': 'intervenciones_vialidades.csv',
            'Construido': 'intervenciones_construido.csv',
            'Agua': 'intervenciones_agua.csv',
            'Areás Verdes': 'intervenciones_areas_verdes.csv',
            'Suelo Descubierto': 'intervenciones_suelo_descubierto.csv'
        }

        for lc_type, csv_name in intervention_csv_filenames.items():
            csv_path = os.path.join(self.data_dir, "csv", csv_name)
            if os.path.exists(csv_path):
                try:
                    df_temp = pd.read_csv(csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df_temp = pd.read_csv(csv_path, encoding='latin-1')

                df_temp.columns = df_temp.columns.str.strip()
                for col in ['file', 'intervencion']:
                    if col in df_temp.columns:
                        df_temp[col] = df_temp[col].astype(str).str.strip()

                self.all_intervention_scenarios[lc_type] = df_temp
                logger.info(f"   Loaded intervention CSV '{csv_name}' for '{lc_type}'.")
            else:
                logger.warning(f"   ⚠️ WARNING: Intervention CSV '{csv_name}' not found at '{csv_path}'.")

    def get_project_context(self, municipio_clave=None, bbox=None):
        context = {}
        if bbox:
            context['bbox'] = bbox
        target_crs_obj = None

        if self.target_crs:
            target_crs_obj = CRS.from_string(self.target_crs)
            logger.info(f"   Using explicit target CRS: {target_crs_obj}")
        else:
            population_path = self.raster_paths.get('Population')
            if population_path and os.path.exists(population_path):
                with rasterio.open(population_path) as src:
                    target_crs_obj = src.crs
                    self.target_crs = str(src.crs)
                    logger.info(f"   Inferred target CRS from population raster: {target_crs_obj}")
            else:
                target_crs_obj = CRS.from_epsg(4326)
                self.target_crs = "EPSG:4326"
                logger.warning(f"   No target CRS specified or inferred. Defaulting to {target_crs_obj.to_string()}")

        base_profile = None
        for name, path in self.raster_paths.items():
            logger.debug(f"   Checking raster: {path}, exists: {os.path.exists(path)}")
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    src_data = src.read(1)
                    src_profile = src.profile.copy()

                    if not np.issubdtype(src_data.dtype, np.floating):
                        src_data = src_data.astype(np.float32)

                    if src.nodata is not None:
                        src_data = np.where(src_data == src.nodata, np.nan, src_data)

                    if base_profile is None or src.crs != target_crs_obj:
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            src.crs, target_crs_obj, src.width, src.height, *src.bounds
                        )

                        reprojected_data = np.empty((dst_height, dst_width), dtype=src_data.dtype)

                        reproject(
                            source=src_data,
                            destination=reprojected_data,
                            src_transform=src_profile['transform'],
                            src_crs=src_profile['crs'],
                            dst_transform=dst_transform,
                            dst_crs=target_crs_obj,
                            resampling=Resampling.nearest,
                            num_threads=4,
                        )

                        base_profile = src_profile.copy()
                        base_profile.update(crs=target_crs_obj, transform=dst_transform, width=dst_width, height=dst_height)
                        context[name] = reprojected_data
                        logger.info(f"   \u2705 Loaded and reprojected {name} to {target_crs_obj.to_string()} (first raster or different CRS).")

                    else:
                        if (src_profile['height'] != base_profile['height'] or
                                src_profile['width'] != base_profile['width'] or
                                src.transform != base_profile['transform']):
                            reprojected_data = np.empty((base_profile['height'], base_profile['width']), dtype=src_data.dtype)
                            reproject(
                                source=src_data,
                                destination=reprojected_data,
                                src_transform=src_profile['transform'],
                                src_crs=src_profile['crs'],
                                dst_transform=base_profile['transform'],
                                dst_crs=base_profile['crs'],
                                resampling=Resampling.nearest,
                                num_threads=4
                            )
                            context[name] = reprojected_data
                            logger.info(f"   \u2705 Loaded {name} (CRS matches, but reprojected to match base spatial properties).")
                        else:
                            context[name] = src_data
                            logger.info(f"   \u2705 Loaded {name} (CRS and spatial properties match).")
            else:
                logger.warning(f"   \u2139\ufe0f WARNING: Raster for '{name}' not found at '{path}'. It will be missing from context.")

        if base_profile:
            context['profile'] = base_profile
            context['transform'] = base_profile['transform']
        else:
            # Modified to raise DataMismatchError
            raise exceptions.DataMismatchError("No valid raster was loaded. Cannot establish a common spatial context.")

        if municipio_clave and self.energy_df is not None and self.thresholds_df is not None:
            cve_mun = int(municipio_clave)
            cve_ent = int(municipio_clave[:2])

            mun_data = self.energy_df[self.energy_df['CVEGEOMUN'] == cve_mun]
            ent_data = self.thresholds_df[self.thresholds_df['CVE_ENT'] == cve_ent]

            if not mun_data.empty and not ent_data.empty:
                context['energy_params'] = {
                    'baseline_consumption': float(mun_data['consumo_per_cap_2020'].iloc[0]),
                    'comfort_temp': float(ent_data['Temperatura_Comfort'].iloc[0]),
                    'consumption_rate': float(ent_data['consumption_sensitivity_per_increased_degree_KWh_month_person'].iloc[0])
                }
            else:
                 logger.warning(f"No energy data found for municipality {municipio_clave}")

        if 'energy_params' not in context:
            context['energy_params'] = {
                'baseline_consumption': 150.0,
                'comfort_temp': 24.0,
                'consumption_rate': 2.5
            }

        if self.municipalities_df is not None:
            context['municipalities_df'] = self.municipalities_df

        context['intervention_scenarios'] = self.all_intervention_scenarios
        context['raster_paths'] = self.raster_paths

        return context

    def get_model(self):
        return self.ml_model
