
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
import gc
import warnings
import re # Import re for regex operations
from typing import Dict, List, Tuple, Any
import logging
from rasterio.warp import reproject, Resampling

from app.core.dto import SimulationResult
from app.core import exceptions # Added import

logger = logging.getLogger(__name__)

class SimulationEngine:
    LC_CODES = {
        'Unclassified': np.nan,
        'Street': 1,
        'Builtup': 2,
        'Shallow_Water': 3,
        'Deep_Water': 4,
        'Sparse_Green': 5,
        'Moderate_Green': 6,
        'Dense_Green': 7,
        'Bareland': 8
    }

    LC_TO_CSV_KEY = {
        LC_CODES['Street']: 'Vialidad',
        LC_CODES['Builtup']: 'Construido',
        LC_CODES['Shallow_Water']: 'Agua',
        LC_CODES['Deep_Water']: 'Agua',
        LC_CODES['Sparse_Green']: 'Areás Verdes',
        LC_CODES['Moderate_Green']: 'Areás Verdes',
        LC_CODES['Dense_Green']: 'Areás Verdes',
        LC_CODES['Bareland']: 'Suelo Descubierto'
    }

    FEATURE_COLS = ['Albedo', 'NDVI', 'MNDWI', 'Building_Height', 'Nighttime_Lights', 'Elevation']

    # Target attributes for population bands (from ResultProcessor) - will be loaded dynamically
    POPULATION_TARGET_ATTRIBUTES = [
        "POB1", "POB8", "POB12", "POB24", "POB54", "HOGAR19", "HOGAR1",
        "DISC1", "VIV18", "VIV1", "VIV2", "VIV16", "VIV10", "VIV7",
        "VIV25", "EDU34", "HOGAR2", "VIV_25_inv"
    ]

    POPULATION_TARGET_ATTRIBUTES_NAMES = [
        "Población total", "Población infantil", "Población adolescente",
        "Población de 3ra edad", "Población femenina",
        "Hogares con jefes de familia de 60 años y más", "Total de hogares",
        "Población con discapacidad", "Viviendas particulares habitadas sin agua entubada",
        "Vivienda totales", "Vivienda particulares habitadas",
        "Vivienda particulares habitadas sin electricidad", "Viviendas con hacinamiento",
        "Viviendas con piso de tierra",
        "Viviendas particulares habitadas con todos los servicios",
        "Población con educación básica incompleta", "Hogares con jefas de familia",
        "Viviendas particulares habitadas sin todos los servicios"
    ]

    def __init__(self, context: dict, simulation_sampling_factor: int = 6, sigma_value: int = 1, prediction_batch_size: int = 10000):
        """
        Initializes the Engine with a pre-loaded Context.
        The context must contain:
        - Rasters (NDVI, Albedo, etc.)
        - 'profile', 'transform'
        - 'energy_params' (dict with comfort_temp, consumption_rate, baseline_consumption)
        - 'intervention_scenarios' (dict of DataFrames for intervention rules)
        - 'ml_model' (The loaded joblib model)
        """
        logger.info("--- [ENGINE] Initializing ---")
        self.context = context

        # 1. Extract Pre-calculated Parameters
        self.ml_model = context.get('ml_model')
        self.energy_params = context.get('energy_params', {})
        self.all_intervention_scenarios = context.get('intervention_scenarios', {})

        # 2. Extract Configuration
        self.simulation_sampling_factor = simulation_sampling_factor
        self.sigma_value = sigma_value
        self.prediction_batch_size = prediction_batch_size

        # 3. Validate Critical Data
        if not self.ml_model:
            logger.error("Context is missing 'ml_model'")
            raise exceptions.ModelError("ML model not found in context. Simulation cannot proceed.")
        if not self.energy_params:
            logger.warning("Context is missing 'energy_params'. KPI calculations may fail.")
        if not self.all_intervention_scenarios:
            logger.warning("Context is missing 'intervention_scenarios'. Intervention rules may not apply.")

        # Suppress sklearn warnings
        warnings.filterwarnings('ignore', message='X does not have valid feature names')

    def ingest_user_interventions(self, gdf_interventions: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
        """
        Vectorization: Converts user polygons (vectors) into a raster mask aligned with the base raster.
        """
        if gdf_interventions.crs is None:
            gdf_interventions.set_crs(profile['crs'], inplace=True)
        else:
            gdf_interventions = gdf_interventions.to_crs(profile['crs'])

        intervention_shapes = ((geom, 1) for geom in gdf_interventions.geometry)

        # Handle cases where geometry might be empty
        try:
            polygon_mask_full_res = rasterize(
                shapes=intervention_shapes,
                out_shape=(profile['height'], profile['width']),
                transform=profile['transform'],
                fill=0,
                dtype='uint8'
            )
        except ValueError:
             # Fallback for empty shapes
            polygon_mask_full_res = np.zeros((profile['height'], profile['width']), dtype='uint8')

        return (polygon_mask_full_res == 1)
    
    def _validate_interventions(self, gdf: gpd.GeoDataFrame):
        """
        Valida que el GeoJSON del usuario cumpla con los requisitos técnicos
        antes de iniciar la simulación.
        """
        # 1. Validar si el GeoDataFrame está vacío
        if gdf.empty:
            raise exceptions.GeometryError("El GeoJSON no contiene geometrías válidas.")

        # 2. Validar que existan las columnas necesarias
        if 'intervention_type' not in gdf.columns:
            raise exceptions.GeometryError(
                "Falta la columna 'intervention_type' en los datos de entrada."
            )

        # 3. Validar consistencia de los tipos de intervención
        valid_types = set(self.LC_CODES.keys())
        input_types = set(gdf['intervention_type'].unique())
        
        invalid_types = input_types - valid_types
        if invalid_types:
            raise exceptions.GeometryError(
                f"Tipos de intervención no reconocidos: {invalid_types}. "
                f"Valores permitidos: {valid_types}"
            )

        # 4. Validar geometría (opcional pero recomendado)
        if not gdf.is_valid.all():
            raise exceptions.GeometryError(
                "Se detectaron geometrías inválidas (polígonos abiertos o auto-intersecados)."
            )

        logger.info("[Engine] Validación de intervenciones exitosa.")

    def _calculate_ozone(self, lst_temp: np.ndarray) -> np.ndarray:
        """
        Calculates estimated maximum Ozone concentration based on Land Surface Temperature.
        Formula source: Peralta et al. (2025), Atmosphere 16(12).

        Args:
            lst_temp (float or np.array): Land Surface Temperature in Celsius.
        Returns:
            float or np.array: Estimated Ozone concentration ([O3]).
        """
        a = 9.33
        b = 0.0957
        return a * np.exp(b * lst_temp)

    def calculate_energy_impact(self, 
                                diff_lst: np.ndarray, 
                                population_mask: np.ndarray, 
                                energy_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcula el impacto en el consumo eléctrico basado en el cambio de temperatura.
        REQUERIMIENTO: Desacoplado de la simulación espacial principal.
        """
        try:
            # Extraer parámetros del contexto (inyectados desde loaders.py)
            rate = energy_params.get('consumption_rate', 2.5) # KWh/persona/grado
            baseline = energy_params.get('baseline_consumption', 150.0)
            
            # El ahorro ocurre donde la temperatura bajó (diff_lst es negativo)
            # Promedio de reducción de grados en áreas pobladas
            avg_temp_reduction = np.nanmean(diff_lst[population_mask > 0])
            
            if np.isnan(avg_temp_reduction):
                return {"monthly_savings_kwh": 0, "percent_reduction": 0}

            # Cálculo: Delta T * Tasa de Consumo * Población Total
            total_population = np.nansum(population_mask)
            monthly_savings = abs(avg_temp_reduction) * rate * total_population
            
            percent_reduction = (monthly_savings / (baseline * total_population)) * 100 if total_population > 0 else 0

            return {
                "monthly_savings_kwh": float(monthly_savings),
                "percent_reduction": float(percent_reduction),
                "avg_temp_reduction": float(avg_temp_reduction)
            }
        except Exception as e:
            logger.error(f"Error en cálculo energético: {e}")
            return {"error": "No se pudo calcular el impacto energético"}

    def _calculate_energy_kpis_for_area(self, lst_data: np.ndarray, population_data: np.ndarray, study_area_mask: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates energy consumption KPIs and intermediate arrays for a given LST array, population data, and study area mask.
        Returns:
            Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
                - total_energy_consumption_MWh_year
                - total_population_in_area
                - delta_lst_positive (intermediate array)
                - increased_per_capita (intermediate array)
                - per_capita_total_consumption_arr (intermediate array)
        """
        # Retrieve parameters from self.energy_params
        comfort_temp = self.energy_params.get('comfort_temp', 24.0)
        consumption_rate = self.energy_params.get('consumption_rate', 2.5)
        baseline_consumption = self.energy_params.get('baseline_consumption', 150.0)

        # 3. Calculate delta_lst_positive
        delta_lst_positive = np.maximum(0, lst_data - comfort_temp)
        delta_lst_positive[study_area_mask == 0] = np.nan

        # 4. Calculate increased_per_capita (annualized)
        increased_per_capita = delta_lst_positive * consumption_rate * 12

        # 6. Calculate per_capita_total_consumption_arr
        # Annualize baseline_consumption before adding
        per_capita_total_consumption_arr = increased_per_capita + baseline_consumption
        per_capita_total_consumption_arr[study_area_mask == 0] = np.nan # Ensure consistency

        # 7. Calculate total_scaled
        total_scaled = per_capita_total_consumption_arr * population_data

        # 8. Handle NaN values in total_scaled
        total_scaled[np.isnan(population_data)] = np.nan
        total_scaled[population_data == 0] = 0.0

        # 9. Calculate total_energy_consumption_MWh_year
        total_energy_consumption_MWh_year = float(np.nansum(total_scaled) / 1000.0)

        # 10. Calculate total_population_in_area
        total_population_in_area = float(np.nansum(population_data[~np.isnan(population_data) & (study_area_mask == 1)]))

        return total_energy_consumption_MWh_year, total_population_in_area, delta_lst_positive, increased_per_capita, per_capita_total_consumption_arr

    def _generate_energy_summary_text(self, polygon_identifier: str, total_population_in_area: float,
                                      actual_total_sum_twin_consumption_scaled: float,
                                      actual_total_sum_intervention_consumption_scaled: float,
                                      actual_total_sum_energy_consumption_difference_scaled: float,
                                      avg_delta_lst_positive_twin: float, avg_delta_lst_positive_intervention: float,
                                      avg_increased_per_capita_twin: float, avg_increased_per_capita_intervention: float,
                                      avg_per_capita_total_twin: float, avg_per_capita_total_intervention: float,
                                      avg_per_capita_difference: float) -> str:
        comfort_temp = self.energy_params.get('comfort_temp', 24.0)
        consumption_rate = self.energy_params.get('consumption_rate', 2.5)
        per_capita_baseline = self.energy_params.get('baseline_consumption', 150.0)

        percentage_change = np.nan
        if actual_total_sum_twin_consumption_scaled != 0:
            percentage_change = (1 - (actual_total_sum_intervention_consumption_scaled / actual_total_sum_twin_consumption_scaled)) * 100

        summary_lines = []
        def print_to_summary(text):
            # Strip ANSI escape codes before adding to summary_lines
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            summary_lines.append(clean_text)

        print_to_summary(f"Resumen de Resultados del Análisis de Consumo de Electricidad: {polygon_identifier}")
        print_to_summary(f" Umbral de temperatura confortable: {float(comfort_temp):.2f} \u00b0C")
        print_to_summary(f" Tasa de consumo de energía por grado de aumento: {float(consumption_rate):.2f} KWh/mes per cápita/\u00b0C")
        print_to_summary(f" Consumo de energía per cápita base (sin aumento por calor): {float(per_capita_baseline):.2f} KWh/mes per cápita\n")

        print_to_summary(f"Deltas de Temperatura Positivas (promedio sobre el polígono de intervención)")
        print_to_summary(f"  Escenario Base Delta LST positivo promedio: {float(avg_delta_lst_positive_twin):.2f} \u00b0C")
        print_to_summary(f"  Escenario Intervención Delta LST positivo promedio: {float(avg_delta_lst_positive_intervention):.2f} \u00b0C")

        print_to_summary(f"Consumo Adicional de Energía Per Cápita (promedio sobre el polígono de intervención)")
        print_to_summary(f"  Escenario Base consumo adicional promedio: {float(avg_increased_per_capita_twin):.2f} KWh/a\u00f1o per cápita")
        print_to_summary(f"  Escenario Intervención consumo adicional promedio: {float(avg_increased_per_capita_intervention):.2f} KWh/a\u00f1o per cápita")

        print_to_summary(f"Consumo Total de Energía Per Cápita (promedio sobre el polígono de intervención)")
        print_to_summary(f"  Escenario Base consumo total promedio: {float(avg_per_capita_total_twin):.2f} KWh/a\u00f1o per cápita")
        print_to_summary(f"  Escenario Intervención consumo total promedio: {float(avg_per_capita_total_intervention):.2f} KWh/a\u00f1o per cápita")
        print_to_summary(f"  Diferencia promedio (Base - Intervención) per cápita: {float(avg_per_capita_difference):.2f} KWh/a\u00f1o per cápita")

        print_to_summary(f"Consumo Residencial Total de Electricidad")
        print_to_summary(f"  Población total en área de análisis: {float(total_population_in_area):.0f} personas")
        print_to_summary(f"  Consumo Residencial Total (Base) : {float(actual_total_sum_twin_consumption_scaled):.0f} MWh/a\u00f1o")
        print_to_summary(f"  Consumo Residencial Total (Intervención) : {float(actual_total_sum_intervention_consumption_scaled):.0f} MWh/a\u00f1o")
        print_to_summary(f"  Diferencia Total del consumo total de energía eléctrica (Base - Intervención): {float(actual_total_sum_energy_consumption_difference_scaled):.2f} MWh/a\u00f1o")
        print_to_summary(f"Porcentaje de ahorro en el consumo total de energía eléctrica: {float(percentage_change):.2f}%")
        print_to_summary(f"\n")

        if actual_total_sum_energy_consumption_difference_scaled > 0:
            print_to_summary(f"Las intervenciones resultaron en un ahorro total estimado de energía de {float(actual_total_sum_energy_consumption_difference_scaled):.0f} MWh/a\u00f1o en el área de análisis.")
        elif actual_total_sum_energy_consumption_difference_scaled < 0:
            print_to_summary(f"Las intervenciones resultaron en un aumento total estimado de energía de {float(-actual_total_sum_energy_consumption_difference_scaled):.0f} MWh/a\u00f1o en el área de análisis.")
        else:
            print_to_summary(f"Las intervenciones no mostraron un cambio significativo en el consumo total de energía escalado por población.")

        return "\n".join(summary_lines)

    def _summarize_energy_consumption_for_polygon(
        self, polygon_identifier: str, twin_lst_data: np.ndarray, intervention_lst_data: np.ndarray,
        population_data: np.ndarray, study_area_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generates a summary dictionary for energy consumption for a given area.
        """
        # Calculate energy KPIs for twin and intervention scenarios using the helper
        actual_total_sum_twin_consumption_scaled, total_population_in_area, \
            delta_lst_positive_twin, increased_per_capita_twin, per_capita_total_twin = \
            self._calculate_energy_kpis_for_area(lst_data=twin_lst_data, population_data=population_data, study_area_mask=study_area_mask)

        actual_total_sum_intervention_consumption_scaled, _, \
            delta_lst_positive_intervention, increased_per_capita_intervention, per_capita_total_intervention = \
            self._calculate_energy_kpis_for_area(lst_data=intervention_lst_data, population_data=population_data, study_area_mask=study_area_mask)

        actual_total_sum_energy_consumption_difference_scaled = (actual_total_sum_twin_consumption_scaled - actual_total_sum_intervention_consumption_scaled)

        # Calculate average metrics from the intermediate arrays
        avg_delta_lst_positive_twin = np.nanmean(delta_lst_positive_twin)
        avg_delta_lst_positive_intervention = np.nanmean(delta_lst_positive_intervention)
        avg_increased_per_capita_twin = np.nanmean(increased_per_capita_twin)
        avg_increased_per_capita_intervention = np.nanmean(increased_per_capita_intervention)
        avg_per_capita_total_twin = np.nanmean(per_capita_total_twin)
        avg_per_capita_total_intervention = np.nanmean(per_capita_total_intervention)
        avg_per_capita_difference = np.nanmean(per_capita_total_twin - per_capita_total_intervention) # Base - Intervención

        # Generate summary text using the new helper method
        summary_text = self._generate_energy_summary_text(
            polygon_identifier,
            total_population_in_area,
            actual_total_sum_twin_consumption_scaled,
            actual_total_sum_intervention_consumption_scaled,
            actual_total_sum_energy_consumption_difference_scaled,
            avg_delta_lst_positive_twin, avg_delta_lst_positive_intervention,
            avg_increased_per_capita_twin, avg_increased_per_capita_intervention,
            avg_per_capita_total_twin, avg_per_capita_total_intervention,
            avg_per_capita_difference
        )

        return {
            'polygon_identifier': polygon_identifier,
            'actual_total_sum_twin_consumption_scaled': float(actual_total_sum_twin_consumption_scaled),
            'actual_total_sum_intervention_consumption_scaled': float(actual_total_sum_intervention_consumption_scaled),
            'summary_text': summary_text
        }

    def _reproject_raster_to_profile(self, source_data: np.ndarray, source_profile: Dict[str, Any], target_profile: Dict[str, Any]) -> np.ndarray:
        """
        Reprojects and resamples a raster dataset to match a target rasterio profile.
        """
        if source_profile['crs'] != target_profile['crs'] or \
           source_profile['width'] != target_profile['width'] or \
           source_profile['height'] != target_profile['height'] or \
           source_profile['transform'] != target_profile['transform']:

            reprojected_data = np.empty((target_profile['height'], target_profile['width']), dtype=source_data.dtype)

            reproject(
                source=source_data,
                destination=reprojected_data,
                src_transform=source_profile['transform'],
                src_crs=source_profile['crs'],
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=Resampling.nearest, # Use nearest for classification data
                num_threads=4
            )
            return reprojected_data
        else:
            return source_data


    def _load_population_bands_and_align_to_target_profile(self, population_raster_path: str, target_profile: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        """
        Loads population bands from the population raster path and aligns them to match a target rasterio profile.
        """
        source_rasters_info = []
        try:
            with rasterio.open(population_raster_path) as src_pop:
                logger.info(f"   Loading population raster: {population_raster_path}")
                for i in range(1, src_pop.count + 1):
                    # Fix: Read band `i` instead of `1` always
                    band_data_src = src_pop.read(i)

                    if (i - 1) < len(self.POPULATION_TARGET_ATTRIBUTES_NAMES):
                        band_name = self.POPULATION_TARGET_ATTRIBUTES_NAMES[i-1]
                    else:
                        band_name = src_pop.descriptions[i-1] if src_pop.descriptions and src_pop.descriptions[i-1] else f'Band {i}'

                    # Reproject each band to the target profile
                    reprojected_band_data = self._reproject_raster_to_profile(band_data_src, src_pop.profile, target_profile)

                    source_rasters_info.append({
                        'data': reprojected_band_data.copy(),
                        'name': band_name
                    })
            logger.info(f"   ✅ Successfully loaded and aligned {len(source_rasters_info)} population bands to target profile.")
        except Exception as e:
            logger.error(f"   ❌ Error loading or aligning population bands from {population_raster_path}: {e}")
            # Modified to raise DataMismatchError
            raise exceptions.DataMismatchError(f"Error loading or aligning population bands from {population_raster_path}") from e
        return source_rasters_info

    def _calculate_zonal_statistics(self, source_band_info: Dict[str, Any], zone_layer_info: Dict[str, Any], polygon_id: str, polygon_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculates zonal statistics for a given source band and zone layer within a specified polygon mask.
        All input arrays (source_band_info['data'], zone_layer_info['data'], polygon_mask) must be at the same resolution.
        """
        stats_results = []

        source_data = source_band_info['data']
        source_band_name = source_band_info['name']

        zone_data_raw = zone_layer_info['data']
        zone_layer_name = zone_layer_info['name']

        # Apply polygon mask to both source and zone data
        polygon_source_data = source_data.copy()
        polygon_source_data[~polygon_mask] = np.nan

        polygon_zone_data_raw = zone_data_raw.copy()
        polygon_zone_data_raw[~polygon_mask] = np.nan

        # Identify and manage no-data values for source_data (NaN only, allow 0 as valid data)
        valid_source_mask = ~np.isnan(polygon_source_data)

        # Identify and manage no-data values for zone_data (NaN and 0)
        zone_data_int = np.nan_to_num(polygon_zone_data_raw, nan=0).astype(int)
        valid_zone_mask = (zone_data_int != 0)

        # Create a combined mask that includes only pixels where both source and zone data are valid
        combined_mask = valid_source_mask & valid_zone_mask

        values = polygon_source_data[combined_mask]
        zones = zone_data_int[combined_mask]

        unique_zones = np.unique(zones)
        if 0 in unique_zones:
            unique_zones = unique_zones[unique_zones != 0] # Exclude 0 (nodata/unclassified)

        if not unique_zones.size:
            return stats_results # Return empty if no valid zones

        for zone_id in sorted(unique_zones):
            zone_values = values[zones == zone_id]

            if zone_values.size > 0:
                stat_count = zone_values.size
                stat_mean = np.mean(zone_values)
                stat_sum = np.sum(zone_values)
                stat_min = np.min(zone_values)
                stat_max = np.max(zone_values)
            else:
                stat_count = 0
                stat_mean = np.nan
                stat_sum = np.nan
                stat_min = np.nan
                stat_max = np.nan

            stats_results.append({
                'polygon_id': polygon_id,
                'zone_layer': zone_layer_name,
                'source_band': source_band_name,
                'zone_id': zone_id,
                'count': stat_count,
                'mean': stat_mean,
                'sum': stat_sum,
                'min': stat_min,
                'max': stat_max
            })
        return stats_results

    def _apply_interventions_and_effects(
        self, feature_arrays_full_res: Dict[str, np.ndarray], land_cover_data_full_res: np.ndarray,
        gdf_interventions: gpd.GeoDataFrame, overall_intervention_mask_boolean: np.ndarray,
        full_res_study_area_mask: np.ndarray,
        profile: dict
    ) -> Dict[str, np.ndarray]:

        feature_arrays_modified_full_res = {col: arr.copy() for col, arr in feature_arrays_full_res.items()}

        for index, feature in gdf_interventions.iterrows():
            polygon_geometry = feature.geometry

            # Rasterize single polygon to get its specific pixel mask
            single_polygon_mask_full_res_arr = rasterize(
                shapes=[(polygon_geometry, 1)],
                out_shape=(profile['height'], profile['width']),
                transform=profile['transform'],
                fill=0,
                dtype='uint8'
            )
            single_polygon_pixel_mask_full_res = (single_polygon_mask_full_res_arr == 1)

            if not np.any(single_polygon_pixel_mask_full_res):
                logger.debug(f"  Polygon ID {feature.get('id', index)} does not intersect with any raster pixels. Skipping.")
                continue

            logger.debug(f"  Processing Polygon ID {feature.get('id', index)}...")

            # Extract polygon-specific intervention parameters from GeoJSON attributes
            polygon_energy_efficiency = feature.get('energy_efficiency', 0.0) # Default to 0 if not present
            polygon_building_height_increase = feature.get('building_height', 0.0) # Default to 0 if not present

            # Get unique land cover codes within this specific polygon (FULL RESOLUTION)
            if land_cover_data_full_res is not None:
                # Only consider land cover pixels that are part of this polygon AND the overall intervention area
                pixels_in_this_polygon_lc = land_cover_data_full_res[single_polygon_pixel_mask_full_res]
                unique_lc_codes_in_polygon = np.unique(pixels_in_this_polygon_lc[~np.isnan(pixels_in_this_polygon_lc)])

                for lc_code_in_polygon_float in unique_lc_codes_in_polygon:
                    lc_code_in_polygon = int(lc_code_in_polygon_float)

                    intervention_to_apply = None
                    lc_name_internal = next((name for name, code in self.LC_CODES.items() if code == lc_code_in_polygon), 'Unknown')
                    csv_key = self.LC_TO_CSV_KEY.get(lc_code_in_polygon, None)

                    # Determine if an intervention flag is set in the GeoJSON for this LC type
                    scenario_attribute_name = lc_name_internal # e.g., 'Builtup', 'Street'

                    if scenario_attribute_name in feature.index:
                        intervention_flag = int(feature[scenario_attribute_name]) # Get the 0, 1, 2... flag

                        if intervention_flag == 0: # No intervention
                            intervention_to_apply = None
                            logger.debug(f"      Polygon ID {feature.get('id', index)}, LC '{lc_name_internal}': Flag 0 en GeoJSON. No se aplicará intervención específica.")
                        elif intervention_flag > 0: # Only apply if a specific scenario is flagged (1-based index in GeoJSON)
                            if csv_key and csv_key in self.all_intervention_scenarios and not self.all_intervention_scenarios[csv_key].empty:
                                csv_index = intervention_flag - 1 # Convert to 0-based for DataFrame iloc
                                if csv_index < len(self.all_intervention_scenarios[csv_key]):
                                    intervention_to_apply = self.all_intervention_scenarios[csv_key].iloc[csv_index]
                                    logger.debug(f"      Applying intervention scenario {intervention_flag} from '{csv_key}' for LC '{lc_name_internal}'.")
                                else:
                                    logger.warning(f"      Polygon ID {feature.get('id', index)}, LC '{lc_name_internal}': Flag {intervention_flag} exceeds available scenarios in '{csv_key}'.")
                            else:
                                logger.warning(f"      Polygon ID {feature.get('id', index)}, LC '{lc_name_internal}': Intervention CSV for '{csv_key}' is empty or not found.")
                    else:
                         logger.debug(f"      Polygon ID {feature.get('id', index)}, LC '{lc_name_internal}': No scenario attribute '{scenario_attribute_name}' in GeoJSON. No specific intervention applied.")

                    # Apply the determined intervention to the pixels within this polygon and LC type
                    if intervention_to_apply is not None:
                        # Create a combined mask for pixels belonging to this polygon AND this specific LC type
                        specific_lc_polygon_mask_full_res = (single_polygon_pixel_mask_full_res) & (land_cover_data_full_res == lc_code_in_polygon)

                        if np.any(specific_lc_polygon_mask_full_res):
                            for variable_name in ['Albedo', 'NDVI', 'MNDWI']:
                                if variable_name in intervention_to_apply:
                                    intervention_value = intervention_to_apply[variable_name]
                                    current_values = feature_arrays_modified_full_res[variable_name][specific_lc_polygon_mask_full_res]

                                    if variable_name == 'Albedo': # Albedo is a direct replacement
                                        feature_arrays_modified_full_res[variable_name][specific_lc_polygon_mask_full_res] = float(intervention_value)
                                    else: # NDVI and MNDWI are additive changes
                                        feature_arrays_modified_full_res[variable_name][specific_lc_polygon_mask_full_res] = current_values + float(intervention_value)

                            # Applying polygon-specific building height increase (only to Builtup areas)
                            if lc_code_in_polygon == self.LC_CODES['Builtup'] and polygon_building_height_increase > 0:
                                # Modifies based strictly on the original baseline height
                                feature_arrays_modified_full_res['Building_Height'][specific_lc_polygon_mask_full_res] = feature_arrays_full_res['Building_Height'][specific_lc_polygon_mask_full_res] + polygon_building_height_increase
                                logger.debug(f"        Building_Height increased by {polygon_building_height_increase} floors for LC 'Builtup'.")

                            # Applying polygon-specific energy efficiency percentage (to Builtup and Street areas)
                            if polygon_energy_efficiency > 0 and (lc_code_in_polygon == self.LC_CODES['Builtup'] or lc_code_in_polygon == self.LC_CODES['Street']):
                                reduction_factor = 1 - ((polygon_energy_efficiency / 100.0) * 0.2) # Multiplier adjusted to be consistent with previous logic
                                feature_arrays_modified_full_res['Nighttime_Lights'][specific_lc_polygon_mask_full_res] *= reduction_factor
                                logger.debug(f"        Nighttime_Lights reduced by {polygon_energy_efficiency}% for LC '{lc_name_internal}'.")
                        else:
                            logger.debug(f"      No pixels of LC '{lc_name_internal}' found within Polygon ID {feature.get('id', index)}.")
                    else:
                        logger.debug(f"      No specific intervention selected for Polygon ID {feature.get('id', index)}, LC '{lc_name_internal}'. No changes applied.")
            else:
                logger.warning(f"  Land Cover data is missing or empty for Polygon ID {feature.get('id', index)}. Cannot apply LC-specific interventions.")

        # Apply Gaussian Halo Effect (applies to the differences between original and modified arrays)
        feature_arrays_halo_full_res = {col: arr.copy() for col, arr in feature_arrays_modified_full_res.items()}

        for variable_name in self.FEATURE_COLS:
            # Elevation is not intervened, so skip halo
            if variable_name == 'Elevation' or variable_name not in feature_arrays_full_res: # Also check if variable_name exists
                continue

            original_data = feature_arrays_full_res[variable_name]
            intervened_data = feature_arrays_modified_full_res[variable_name]

            # Calculate the difference between intervened and original state
            diff_full_res = intervened_data - original_data

            # Create a mask for valid differences exactly as requested (all original non-NaN data)
            valid_diff_mask = ~np.isnan(original_data)

            if np.any(valid_diff_mask):
                # Initialize temporary difference array with zeros for filtering
                diff_temp_for_filter = np.zeros_like(diff_full_res, dtype=np.float32)
                # Populate with actual differences only for valid pixels
                diff_temp_for_filter[valid_diff_mask] = diff_full_res[valid_diff_mask]

                # Apply Gaussian filter to the difference
                blurred_diff = gaussian_filter(diff_temp_for_filter, sigma=self.sigma_value, mode='nearest')

                # Add the blurred difference back to the ORIGINAL full-resolution data
                halo_applied_data = original_data + blurred_diff

                # Mask outside study area if needed
                final_halo_data = halo_applied_data
                final_halo_data[full_res_study_area_mask == 0] = np.nan
                feature_arrays_modified_full_res[variable_name] = final_halo_data

                logger.debug(f"    ✅ Halo effect applied to: {variable_name}")

        return feature_arrays_modified_full_res

    def _resample_layers(
        self, feature_arrays_modified_full_res: Dict[str, np.ndarray], original_feature_arrays_full_res: Dict[str, np.ndarray],
        land_cover_data_full_res: np.ndarray, population_data_full_res: np.ndarray,
        full_res_study_area_mask: np.ndarray, overall_intervention_mask_boolean: np.ndarray,
        full_res_profile: dict
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
        """ Resamples full resolution layers to the simulation sampling factor. """
        feature_arrays_sampled = {}
        original_feature_arrays_sampled = {}

        # Slicing for subsampling
        s = slice(None, None, self.simulation_sampling_factor)

        for col in self.FEATURE_COLS:
            if col in feature_arrays_modified_full_res:
                feature_arrays_sampled[col] = feature_arrays_modified_full_res[col][s, s]
            if col in original_feature_arrays_full_res:
                original_feature_arrays_sampled[col] = original_feature_arrays_full_res[col][s, s]

        # Handle None layers gracefully
        lc_sampled = land_cover_data_full_res[s, s] if land_cover_data_full_res is not None else None
        pop_sampled = population_data_full_res[s, s] if population_data_full_res is not None else None
        mask_sampled = full_res_study_area_mask[s, s]

        # Sample the overall_intervention_mask_boolean (this mask is at LST-aligned full resolution)
        overall_intervention_mask_sampled_LST_res = overall_intervention_mask_boolean[s, s]

        sampled_height, sampled_width = mask_sampled.shape
        sampled_transform = full_res_profile['transform'] * full_res_profile['transform'].scale(self.simulation_sampling_factor, self.simulation_sampling_factor)

        sampled_profile = full_res_profile.copy()
        sampled_profile.update({
            'height': sampled_height,
            'width': sampled_width,
            'transform': sampled_transform
        })

        return (feature_arrays_sampled, original_feature_arrays_sampled, lc_sampled,
                pop_sampled, mask_sampled, sampled_profile, overall_intervention_mask_sampled_LST_res)

    def _prepare_feature_arrays(self, feature_arrays: Dict[str, np.ndarray], original_feature_arrays: Dict[str, np.ndarray], sampled_profile: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        reshaped_feature_arrays = {col: arr.flatten() for col, arr in feature_arrays.items()}
        reshaped_original_feature_arrays = {col: arr.flatten() for col, arr in original_feature_arrays.items()}
        return reshaped_feature_arrays, reshaped_original_feature_arrays

    def _predict_lst(self, reshaped_feature_arrays: Dict[str, np.ndarray], reshaped_original_feature_arrays: Dict[str, np.ndarray], sampled_profile: dict, study_area_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_pixels = sampled_profile['height'] * sampled_profile['width']
        predicted_lst_intervention = np.full(total_pixels, np.nan, dtype=np.float32)
        predicted_lst_twin = np.full(total_pixels, np.nan, dtype=np.float32)

        # Batch Prediction Logic
        for i in range(0, total_pixels, self.prediction_batch_size):
            batch_end = min(i + self.prediction_batch_size, total_pixels)

            # Helper to predict batch
            def predict_batch(feature_dict: Dict[str, np.ndarray], output_array: np.ndarray):
                X_batch_list = [feature_dict[col][i:batch_end] for col in self.FEATURE_COLS]
                X_batch = np.stack(X_batch_list, axis=1)
                valid_mask = ~np.any(np.isnan(X_batch), axis=1)

                if np.any(valid_mask):
                    preds = self.ml_model.predict(X_batch[valid_mask])
                    output_slice = np.full(batch_end - i, np.nan, dtype=np.float32)
                    output_slice[valid_mask] = preds
                    output_array[i:batch_end] = output_slice

            predict_batch(reshaped_feature_arrays, predicted_lst_intervention)
            predict_batch(reshaped_original_feature_arrays, predicted_lst_twin)

        # Reshape
        shape = (sampled_profile['height'], sampled_profile['width'])
        predicted_lst_intervention = predicted_lst_intervention.reshape(shape)
        predicted_lst_twin = predicted_lst_twin.reshape(shape)

        # Apply Mask
        if study_area_mask is not None:
            predicted_lst_intervention[study_area_mask == 0] = np.nan
            predicted_lst_twin[study_area_mask == 0] = np.nan

        return predicted_lst_intervention, predicted_lst_twin

    def _classify_icu_zones(self, lst_map: np.ndarray, mean_lst: float) -> np.ndarray:
        """
        Segmenta el mapa de LST en zonas de Isla de Calor Urbana:
        1 = Normal, 2 = Transición, 3 = Impacto, 4 = Núcleo.
        Utiliza el mean_lst del escenario base como referencia.
        """
        icu_zones_map = np.full(lst_map.shape, np.nan, dtype=np.float32)

        if np.isnan(mean_lst):
            return icu_zones_map

        # Normal: LST < average
        mask_normal = lst_map < mean_lst
        icu_zones_map[mask_normal] = 1

        # Transition Zone: average <= LST < average + 1
        mask_transition = (lst_map >= mean_lst) & (lst_map < mean_lst + 1)
        icu_zones_map[mask_transition] = 2

        # Impact Zone: average + 1 <= LST < average + 3
        mask_impact = (lst_map >= mean_lst + 1) & (lst_map < mean_lst + 3) # Define mask_impact
        icu_zones_map[mask_impact] = 3

        # Core Zone: LST >= mean_lst + 3
        mask_core = lst_map >= mean_lst + 3
        icu_zones_map[mask_core] = 4

        # Re-apply NaNs from the original map
        icu_zones_map[np.isnan(lst_map)] = np.nan

        return icu_zones_map

    def _calculate_impact_metrics(
        self,
        twin_lst_data_sampled: np.ndarray, intervention_lst_data_sampled: np.ndarray,
        study_area_mask_sampled: np.ndarray,
        twin_lst_data_full_res_for_energy: np.ndarray, intervention_lst_data_full_res_for_energy: np.ndarray,
        population_data_full_res_for_energy: np.ndarray,
        study_area_mask_full_res_for_energy: np.ndarray, # This is overall_intervention_mask_aligned_LST_res_boolean
        ozone_twin_data_sampled: np.ndarray, ozone_intervention_data_sampled: np.ndarray,
        gdf_interventions: gpd.GeoDataFrame, sampled_profile: Dict[str, Any],
        global_total_population_sum_true_full_res: float
    ) -> Dict[str, Any]:

        global_kpis = {}
        all_energy_summaries = []

        # 1. Populate core KPIs (average temperatures, max cooling, average LST change) - using SAMPLED LST
        valid_twin_lst = twin_lst_data_sampled[~np.isnan(twin_lst_data_sampled) & (study_area_mask_sampled == 1)]
        valid_intervention_lst = intervention_lst_data_sampled[~np.isnan(intervention_lst_data_sampled) & (study_area_mask_sampled == 1)]

        if valid_twin_lst.size > 0:
            global_kpis['avg_temp_base'] = float(np.nanmean(valid_twin_lst)) # Use nanmean
        else:
            global_kpis['avg_temp_base'] = np.nan

        if valid_intervention_lst.size > 0:
            global_kpis['avg_temp_intervention'] = float(np.nanmean(valid_intervention_lst)) # Use nanmean
        else:
            global_kpis['avg_temp_intervention'] = np.nan

        difference_data_sampled = intervention_lst_data_sampled - twin_lst_data_sampled
        valid_difference_data_sampled = difference_data_sampled[~np.isnan(difference_data_sampled) & (study_area_mask_sampled == 1)]
        if valid_difference_data_sampled.size > 0:
            global_kpis['max_cooling'] = float(np.nanmin(valid_difference_data_sampled)) # Use nanmin
            global_kpis['avg_lst_change'] = float(np.nanmean(valid_difference_data_sampled)) # Use nanmean
        else:
            global_kpis['max_cooling'] = np.nan
            global_kpis['avg_lst_change'] = np.nan

        # 2. Calculate ozone-related KPIs (use provided ozone_twin_data_sampled and ozone_intervention_data_sampled, which are SAMPLED AND MASKED)
        global_kpis['total_ozone_twin_ppb'] = float(np.nansum(ozone_twin_data_sampled))
        global_kpis['total_ozone_intervention_ppb'] = float(np.nansum(ozone_intervention_data_sampled))
        global_kpis['ozone_reduction_ppb'] = global_kpis['total_ozone_twin_ppb'] - global_kpis['total_ozone_intervention_ppb']

        if global_kpis['total_ozone_twin_ppb'] > 0:
            global_kpis['ozone_percentage_improvement'] = float((1 - (global_kpis['total_ozone_intervention_ppb'] / global_kpis['total_ozone_twin_ppb'])) * 100)
        else:
            global_kpis['ozone_percentage_improvement'] = np.nan

        # 3. Call _summarize_energy_consumption_for_polygon for GLOBAL summary - using FULL RES for energy
        global_energy_summary = self._summarize_energy_consumption_for_polygon(
            polygon_identifier='Global',
            twin_lst_data=twin_lst_data_full_res_for_energy, # Use full res LST
            intervention_lst_data=intervention_lst_data_full_res_for_energy, # Use full res LST
            population_data=population_data_full_res_for_energy, # Use full res population
            study_area_mask=study_area_mask_full_res_for_energy # Use full res mask
        )
        all_energy_summaries.append(global_energy_summary)

        # Add global energy KPIs to global_kpis dictionary
        global_kpis['total_energy_consumption_twin_MWh_year'] = global_energy_summary['actual_total_sum_twin_consumption_scaled']
        global_kpis['total_energy_consumption_intervention_MWh_year'] = global_energy_summary['actual_total_sum_intervention_consumption_scaled']
        global_kpis['energy_savings_MWh'] = global_energy_summary['actual_total_sum_twin_consumption_scaled'] - global_energy_summary['actual_total_sum_intervention_consumption_scaled']
        global_kpis['total_population_global'] = global_total_population_sum_true_full_res


        # 4. Iterate through gdf_interventions for PER-POLYGON energy summaries
        for index, feature in gdf_interventions.iterrows():
            polygon_geometry = feature.geometry
            polygon_id = feature['id'] if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
            polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

            # Rasterize the current polygon to create its specific mask
            # Use the FULL LST resolution profile for this mask
            polygon_mask_data_full_res = rasterize(
                shapes=[(polygon_geometry, 1)],
                out_shape=study_area_mask_full_res_for_energy.shape, # Use the full res mask shape
                transform=self.context['profile']['transform'], # Use the full LST resolution transform
                fill=0,
                dtype='uint8'
            )
            polygon_pixel_mask_boolean_full_res = (polygon_mask_data_full_res == 1) & study_area_mask_full_res_for_energy # Combine with overall full res mask

            # Calculate energy summary for this polygon, using FULL RES data
            per_polygon_energy_summary = self._summarize_energy_consumption_for_polygon(
                polygon_identifier=f'Polígono ID {polygon_id_str}',
                twin_lst_data=twin_lst_data_full_res_for_energy,
                intervention_lst_data=intervention_lst_data_full_res_for_energy,
                population_data=population_data_full_res_for_energy,
                study_area_mask=polygon_pixel_mask_boolean_full_res
            )
            all_energy_summaries.append(per_polygon_energy_summary)

        # 5. Return composite dictionary
        return {'global_kpis': global_kpis, 'energy_summaries': all_energy_summaries}


    def run_simulation(self, gdf_interventions: gpd.GeoDataFrame) -> SimulationResult:
        """
        Orchestrates the entire simulation process using the internal CONTEXT.
        """
        logger.info("--- ➡ [ENGINE] Running simulation ---")

        # 1. Extract Basic Context Data
        profile = self.context['profile'] # Profile of the initial rasters (aligned full resolution from DataLoader)
        full_res_shape_aligned = (profile['height'], profile['width']) # Aligned full resolution shape

        # Extract Full Res Feature Arrays from Context
        feature_arrays_full_res_aligned = {col: self.context[col] for col in self.FEATURE_COLS if col in self.context}
        original_feature_arrays_full_res_aligned = {col: arr.copy() for col, arr in feature_arrays_full_res_aligned.items()}

        # Extract Ancillary Layers from Context
        land_cover_data_full_res_aligned = self.context.get('Land_Cover')
        # Population data at full resolution, ALIGNED TO LST PROFILE, used for energy calculations at full LST res
        population_data_full_res_aligned_for_energy = self.context.get('Population')

        # Create a default study area mask (assuming full area if not provided in context)
        # This mask should be at the ALIGNED FULL LST resolution initially.
        full_res_study_area_mask_aligned = np.ones(full_res_shape_aligned, dtype='uint8')

        # 1.5. Ingest user interventions (Overall mask of all polygons at ALIGNED FULL LST resolution)
        overall_intervention_mask_aligned_LST_res_boolean = self.ingest_user_interventions(gdf_interventions, profile)

        # 2. Apply interventions and halo effect
        feature_arrays_modified_full_res_aligned = self._apply_interventions_and_effects(
            original_feature_arrays_full_res_aligned.copy(),
            land_cover_data_full_res_aligned,
            gdf_interventions,
            overall_intervention_mask_aligned_LST_res_boolean,
            full_res_study_area_mask_aligned,
            profile
        )

        # 3. Resample layers for prediction (to SAMPLED LST resolution)
        (feature_arrays_sampled,
         original_feature_arrays_sampled,
         land_cover_data_sampled,
         population_distribution_layer_sampled, # population_distribution_layer_sampled is POB1 sampled to LST resolution
         study_area_mask_sampled, # This is the sampled study area mask (at LST sampled res)
         sampled_profile, # This is the sampled LST profile
         overall_intervention_mask_sampled_LST_res) = self._resample_layers(
            feature_arrays_modified_full_res_aligned, original_feature_arrays_full_res_aligned,
            land_cover_data_full_res_aligned, population_data_full_res_aligned_for_energy,
            full_res_study_area_mask_aligned, overall_intervention_mask_aligned_LST_res_boolean,
            profile # Pass the aligned full resolution profile
        )

        # 4. Prepare feature arrays for ML model
        (reshaped_feature_arrays, reshaped_original_feature_arrays) = self._prepare_feature_arrays(
            feature_arrays_sampled, original_feature_arrays_sampled, sampled_profile
        )

        # 5. Predict new LST for both scenarios (at sampled resolution)
        predicted_lst_intervention_sampled, predicted_lst_twin_sampled = self._predict_lst(
            reshaped_feature_arrays, reshaped_original_feature_arrays, sampled_profile,
            study_area_mask_sampled # This is sampled LST mask
        )

        # Reproject sampled LST predictions back to full LST resolution for energy calculations
        # and for other KPIs that should represent the "true" impact at full resolution.
        predicted_lst_intervention_full_res = np.empty((profile['height'], profile['width']), dtype=np.float32)
        reproject(
            source=predicted_lst_intervention_sampled,
            destination=predicted_lst_intervention_full_res,
            src_transform=sampled_profile['transform'],
            src_crs=sampled_profile['crs'],
            dst_transform=profile['transform'],
            dst_crs=profile['crs'],
            resampling=Resampling.nearest, # Using nearest here for simplicity and to preserve values
            num_threads=4
        )
        predicted_lst_twin_full_res = np.empty((profile['height'], profile['width']), dtype=np.float32)
        reproject(
            source=predicted_lst_twin_sampled,
            destination=predicted_lst_twin_full_res,
            src_transform=sampled_profile['transform'],
            src_crs=sampled_profile['crs'],
            dst_transform=profile['transform'],
            dst_crs=profile['crs'],
            resampling=Resampling.nearest, # Using nearest here for simplicity and to preserve values
            num_threads=4
        )
        # Ensure NaNs are consistent after reprojection. Be careful with direct comparison to `[0,0]` element if it could be NaN.
        # A more robust way is to reapply the mask if the reprojection fills NaNs.
        predicted_lst_intervention_full_res[overall_intervention_mask_aligned_LST_res_boolean == False] = np.nan
        predicted_lst_twin_full_res[overall_intervention_mask_aligned_LST_res_boolean == False] = np.nan

        # 5.5. Clasificar Zonas ICU (these still use sampled LST data for the DTO output maps)
        mean_lst_twin_sampled = float(np.nanmean(predicted_lst_twin_sampled[~np.isnan(predicted_lst_twin_sampled) & (study_area_mask_sampled == 1)])) if predicted_lst_twin_sampled[~np.isnan(predicted_lst_twin_sampled) & (study_area_mask_sampled == 1)].size > 0 else np.nan

        icu_classes_intervention = self._classify_icu_zones(predicted_lst_intervention_sampled, mean_lst_twin_sampled)
        base_icu_classes = self._classify_icu_zones(predicted_lst_twin_sampled, mean_lst_twin_sampled)

        # 5.6. Calculate Ozone Concentration (these should use sampled LST data for consistency with output maps)
        ozone_twin_data = self._calculate_ozone(predicted_lst_twin_sampled)
        ozone_intervention_data = self._calculate_ozone(predicted_lst_intervention_sampled)

        # IMPORTANT: Mask ozone data to *only the intervention areas* for summation, consistent with original cell 21be9bd9
        ozone_twin_data[overall_intervention_mask_sampled_LST_res == False] = np.nan
        ozone_intervention_data[overall_intervention_mask_sampled_LST_res == False] = np.nan

        # --- ZONAL STATISTICS PREP (for Population and ICU classifications) ---
        # Get the full original profile of the population raster
        population_raster_path = self.context['raster_paths']['Population']
        with rasterio.open(population_raster_path) as src_pop_full:
            full_pop_profile = src_pop_full.profile.copy()
            full_pop_shape = src_pop_full.shape
            full_pop_transform = src_pop_full.transform
            full_pop_crs = src_pop_full.crs

        # Reproject overall_intervention_mask from ALIGNED FULL LST resolution to FULL POPULATION resolution
        reprojected_overall_intervention_mask_POP_res = np.empty(full_pop_shape, dtype=np.uint8)
        reproject(
            source=overall_intervention_mask_aligned_LST_res_boolean.astype(np.uint8), # Use aligned LST res mask
            destination=reprojected_overall_intervention_mask_POP_res,
            src_transform=profile['transform'], # Transform of aligned LST resolution mask
            src_crs=profile['crs'],
            dst_transform=full_pop_transform,
            dst_crs=full_pop_crs,
            resampling=Resampling.nearest,
            num_threads=4
        )
        overall_intervention_mask_POP_res_boolean = (reprojected_overall_intervention_mask_POP_res == 1)

        # 6. Calculate impact metrics (KPIs) and energy summaries
        # global_total_population_sum_true_full_res needs to be derived from the true full resolution POB1 data
        # and the overall intervention mask reprojected to population resolution.
        population_bands_info_full_res_for_sum = self._load_population_bands_and_align_to_target_profile(population_raster_path, full_pop_profile)
        full_res_pob1_data_at_pop_res = population_bands_info_full_res_for_sum[0]['data'] # Assuming POB1 is the first band

        if full_res_pob1_data_at_pop_res is not None:
            # Mask where POB1 data is not NaN and within the overall intervention area (at population resolution)
            full_res_pob1_data_masked_at_pop_res = np.where(full_res_pob1_data_at_pop_res == 0, np.nan, full_res_pob1_data_at_pop_res)
            global_total_population_sum_true_full_res = np.nansum(full_res_pob1_data_masked_at_pop_res[overall_intervention_mask_POP_res_boolean])
        else:
            global_total_population_sum_true_full_res = np.nan
            logger.warning("Population data (POB1) not found for true full-resolution global population sum.")

        composite_kpis_dict = self._calculate_impact_metrics(
            predicted_lst_twin_sampled, predicted_lst_intervention_sampled, # Sampled LSTs for LST-related KPIs
            study_area_mask_sampled, # Sampled LST mask for LST-related KPIs
            predicted_lst_twin_full_res, predicted_lst_intervention_full_res, # Full res LST for energy
            population_data_full_res_aligned_for_energy, # Full res population for energy (aligned to LST full res)
            overall_intervention_mask_aligned_LST_res_boolean, # Full res LST mask for energy
            ozone_twin_data, ozone_intervention_data, # Sampled resolution, now masked to intervention area
            gdf_interventions, sampled_profile, # sampled_profile for per-polygon rasterize (needs to be consistent with sampled LST if used)
            global_total_population_sum_true_full_res
        )

        # 7. Calculate Population Zonal Statistics (at FULL population raster resolution)
        logger.info("--- [ENGINE] Calculating Population Zonal Statistics at FULL Resolution ---")
        all_population_zonal_stats = []

        # Use population_bands_info_full_res_for_sum as it's already loaded and aligned to full_pop_profile
        population_bands_info_full_res = population_bands_info_full_res_for_sum

        # Reproject LST classification maps (which are at sampled_profile resolution) to full_pop_profile
        logger.info("     Reprojecting LST classification maps to full population resolution...")
        full_res_base_icu_classes = self._reproject_raster_to_profile(base_icu_classes, sampled_profile, full_pop_profile)
        full_res_icu_classes_intervention = self._reproject_raster_to_profile(icu_classes_intervention, sampled_profile, full_pop_profile)

        # Ensure that non-data values from source are also non-data in destination.
        if sampled_profile['nodata'] is not None:
            # Assuming nodata is already handled as np.nan before reprojection, this might not be strictly necessary.
            # Instead, apply the full resolution mask where appropriate.
            full_res_base_icu_classes[overall_intervention_mask_POP_res_boolean == False] = np.nan
            full_res_icu_classes_intervention[overall_intervention_mask_POP_res_boolean == False] = np.nan

        # Prepare zone layers using the FULL resolution LST classification maps
        full_res_zone_layers_info = [
            {'data': full_res_base_icu_classes, 'name': 'lst_classification_twin_map.tif', 'scenario': 'Base'},
            {'data': full_res_icu_classes_intervention, 'name': 'lst_classification_intervention_map.tif', 'scenario': 'Intervención'}
        ]

        # Global Zonal Statistics - use the reprojected mask at population resolution
        for source_band in population_bands_info_full_res:
            for zone_layer in full_res_zone_layers_info:
                stats = self._calculate_zonal_statistics(source_band, zone_layer, 'Global', overall_intervention_mask_POP_res_boolean)
                all_population_zonal_stats.extend(stats)

        # Per-Polygon Zonal Statistics
        for index, feature in gdf_interventions.iterrows():
            polygon_geometry = feature.geometry
            polygon_id = feature['id'] if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
            polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

            # Rasterize to FULL population resolution
            polygon_mask_data_full_res_pop_res = rasterize(
                shapes=[(polygon_geometry, 1)],
                out_shape=full_pop_shape,
                transform=full_pop_transform,
                fill=0,
                dtype='uint8'
            )
            # Combine with the overall intervention mask (reprojected to population resolution)
            polygon_pixel_mask_boolean_full_res_pop_res = (polygon_mask_data_full_res_pop_res == 1) & overall_intervention_mask_POP_res_boolean

            if not np.any(polygon_pixel_mask_boolean_full_res_pop_res):
                logger.warning(f"       Polygon ID {polygon_id_str} does not intersect with any valid study area pixels at full resolution. Skipping zonal stats.")
                continue

            for source_band in population_bands_info_full_res:
                for zone_layer in full_res_zone_layers_info:
                    stats = self._calculate_zonal_statistics(source_band, zone_layer, polygon_id_str, polygon_pixel_mask_boolean_full_res_pop_res)
                    all_population_zonal_stats.extend(stats)

        population_zonal_stats_df = pd.DataFrame(all_population_zonal_stats)
        if not population_zonal_stats_df.empty:
            population_zonal_stats_df['Scenario'] = population_zonal_stats_df['zone_layer'].apply(lambda x:
                'Base' if 'twin' in x.lower() else 'Intervención'
            )
            population_zonal_stats_df['sum'] = pd.to_numeric(population_zonal_stats_df['sum'], errors='coerce')
            population_zonal_stats_df['percentage_of_sum'] = population_zonal_stats_df.groupby(['polygon_id', 'source_band', 'Scenario'])['sum'].transform(lambda x: (x / x.sum()) * 100 if x.sum() != 0 else np.nan)
        else:
            logger.warning("No population zonal statistics were calculated.")

        logger.info("--- [ENGINE] Population Zonal Statistics Calculation Complete ---")

        logger.info("--- ➡ [ENGINE] Simulation complete ---")

        # Retrieve municipality and state names
        municipality_name = "Desconocido"
        state_name = "Desconocido"
        municipalities_df = self.context.get('municipalities_df')
        municipio_clave = self.context.get('municipio_clave') # Assuming municipio_clave is now in context

        if municipalities_df is not None and municipio_clave is not None:
            filtered_mun_row = municipalities_df[municipalities_df['CVEGEO'] == municipio_clave]
            if not filtered_mun_row.empty:
                municipality_name = filtered_mun_row['NOM_MUN'].iloc[0]
                state_name = filtered_mun_row['NOM_ENT'].iloc[0]
            else:
                logger.warning(f"Municipality {municipio_clave} not found in municipalities_df.")


        return SimulationResult(
            base_lst=predicted_lst_twin_sampled,
            intervention_lst=predicted_lst_intervention_sampled,
            difference_lst=predicted_lst_intervention_sampled - predicted_lst_twin_sampled,
            base_icu_classes=base_icu_classes,
            icu_classes=icu_classes_intervention,
            kpis=composite_kpis_dict['global_kpis'],
            bbox=list(self.context.get('bbox', [0,0,0,0])),
            profile=sampled_profile,
            ozone_twin=ozone_twin_data,
            ozone_intervention=ozone_intervention_data,
            population_sampled=population_distribution_layer_sampled,
            energy_summaries=composite_kpis_dict.get('energy_summaries', []), 
            overall_intervention_mask_sampled=overall_intervention_mask_sampled_LST_res,
            modified_features=feature_arrays_modified_full_res_aligned,
            population_zonal_stats_df=population_zonal_stats_df,
            municipality_name=municipality_name,
            state_name=state_name
        )