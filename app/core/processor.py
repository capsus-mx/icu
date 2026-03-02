import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize
from reportlab.lib import colors
from reportlab.lib.units import inch

from app.core.dto import SimulationResult
from PIL import Image as PILImage
from rasterio.warp import reproject, Resampling
from babel.dates import format_date
from datetime import datetime
from reportlab.lib.utils import ImageReader
import contextily as ctx

# Configurar logger
logger = logging.getLogger(__name__)

class ResultProcessor:
    # Define target attributes and their display names globally within the class or module if they are constant
    # Or pass them into the constructor if dynamic.
    # For this subtask, assuming they are defined within the class for simplicity.
    target_attributes = [
        "POB1", "POB8", "POB12", "POB24", "POB54", "HOGAR19", "HOGAR1",
        "DISC1", "VIV18", "VIV1", "VIV2", "VIV16", "VIV10", "VIV7",
        "VIV25", "EDU34", "HOGAR2", "VIV_25_inv"
    ]

    target_attributes_names = [
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

    def _sanitize_band_name(self, band_name: str) -> str:
        """Sanitizes a band name for use in filenames by replacing spaces, slashes, and accents."""
        sanitized = band_name.replace(' ', '_').replace('/', '_')
        sanitized = sanitized.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        sanitized = sanitized.replace('ñ', 'n').replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N')
        return sanitized

    def __init__(self, result: SimulationResult, output_base_dir: str, project_name: str, municipio_clave: str, gdf_interventions: gpd.GeoDataFrame, local_population_raster_path: str, energy_summary_path: str, zonal_stats_path: str, local_logo_dir: str):
        self.result = result
        self.output_base_dir = output_base_dir
        self.project_name = project_name
        # Retrieve municipio_clave, municipality_name, state_name from SimulationResult
        self.municipio_clave = municipio_clave # Assigned directly from parameter
        self.municipality_name = result.municipality_name
        self.state_name = result.state_name
        self.gdf_interventions = gdf_interventions
        self.local_population_raster_path = local_population_raster_path
        self.energy_summary_path = energy_summary_path
        self.zonal_stats_path = zonal_stats_path

        # Construct the report_inputs path (within proyecto folder)
        self.report_inputs_dir = os.path.join(output_base_dir, "report_inputs")
        self.local_logo_dir = local_logo_dir # Use the passed parameter
        os.makedirs(self.report_inputs_dir, exist_ok=True)
        os.makedirs(self.local_logo_dir, exist_ok=True) # Ensure this path also exists

        # Other output directories for GeoTIFFs
        self.simulations_dir = os.path.join(output_base_dir, "simulations")
        self.lst_classification_dir = os.path.join(output_base_dir, "lst_classification")
        self.ozone_concentration_dir = os.path.join(output_base_dir, "ozone_concentration") # Assuming ozone geotiffs go here
        self.energy_consumption_dir = os.path.join(output_base_dir, "energy_consumption") # Assuming energy geotiffs go here

        os.makedirs(self.simulations_dir, exist_ok=True)
        os.makedirs(self.lst_classification_dir, exist_ok=True)
        os.makedirs(self.ozone_concentration_dir, exist_ok=True)
        os.makedirs(self.energy_consumption_dir, exist_ok=True)


        # Define all the fixed paths for generated assets
        self.paths = {
            "metrics": os.path.join(self.report_inputs_dir, "metrics.json"), # This will store the KPIs from SimulationResult
            "map_impact": os.path.join(self.report_inputs_dir, "heatmap_impact.png"),
            "map_base": os.path.join(self.report_inputs_dir, "lst_twin_map.png"),
            "map_intervention": os.path.join(self.report_inputs_dir, "lst_intervention_map.png"),
            "geotiff_base_lst": os.path.join(self.simulations_dir, "predicted_lst_twin_halo.tif"),
            "geotiff_intervention_lst": os.path.join(self.simulations_dir, "predicted_lst_intervention_halo.tif"),
            "geotiff_difference_lst": os.path.join(self.simulations_dir, "lst_difference_halo.tif"),
            "report_pdf": os.path.join(output_base_dir, f"informe_{project_name}.pdf") # Final PDF path
        }

        # LST Classification Constants (from ae2761d3)
        self.LST_CLASS_CODES = {
            'Normal': 1,
            'Transition Zone': 2,
            'Impact Zone': 3,
            'Core Zone': 4,
            'Unclassified': np.nan
        }
        self.LST_CLASS_COLORS = {
            1: '#00FF00',  # Green - Normal
            2: '#FFFF00',  # Yellow - Transition Zone
            3: '#FFA500',  # Orange - Impact Zone
            4: '#FF0000'   # Red - Core Zone
        }
        self.LST_CLASS_LABELS = {
            1: 'Fondo térmico urbano \n(LST < Promedio)',
            2: 'Zona \nde Transición \n( LST >= Promedio)',
            3: 'Zona \nImpactada \n( LST >= Promedio + 2)',
            4: 'Zona \nNúcleo \n( LST >= Promedio + 3)'
        }

        # Zonal Stats Chart Titles (from df0e84ac)
        # Dynamically build this based on POPULATION_TARGET_ATTRIBUTES_NAMES using sanitized names for keys
        self.chart_titles = {}
        for band_display_name in self.target_attributes_names:
            sanitized_band_name = self._sanitize_band_name(band_display_name)
            self.chart_titles[f'zonal_stats_percentage_Global_{sanitized_band_name}.png'] = f'{band_display_name} por Zona LST (Global)'

        # Add the non-zonal stats chart titles
        self.chart_titles['ozone_improvement_donut_chart_global.png'] = 'Porcentaje de cambio en la concentración de ozono (O3) Global'
        self.chart_titles['total_energy_consumption_bar_chart_global.png'] = 'Consumo Residencial Total de Electricidad Escalado por Población (Global)'

        # Initialize source_rasters_info and zonal_stats_df
        self.source_rasters_info = []
        self.zonal_stats_df = self.result.population_zonal_stats_df # Initialize from SimulationResult

        # Prepare colormap for LST classification
        unique_codes_in_map = np.unique(self.result.icu_classes[~np.isnan(self.result.icu_classes)]).astype(int)
        present_colors = [self.LST_CLASS_COLORS[code] for code in unique_codes_in_map if code in self.LST_CLASS_COLORS]
        if present_colors:
            self.cmap_classification = mcolors.ListedColormap(present_colors)
            self.cmap_classification.set_bad(color='white', alpha=0) # Make NaN transparent
            bounds = np.array(unique_codes_in_map) - 0.5
            bounds = np.append(bounds, bounds[-1] + 1)
            self.norm_classification = mcolors.BoundaryNorm(bounds, self.cmap_classification.N)
        else:
            self.cmap_classification = mcolors.ListedColormap(['#D3D3D3'])
            self.cmap_classification.set_bad('white', alpha=0)
            self.norm_classification = mcolors.BoundaryNorm([0, 1], self.cmap_classification.N)
        # Use fixed classes to ensure consistency between Base and Intervention maps
        fixed_codes = [1, 2, 3, 4]
        fixed_colors = [self.LST_CLASS_COLORS[code] for code in fixed_codes]
        
        self.cmap_classification = mcolors.ListedColormap(fixed_colors)
        self.cmap_classification.set_bad(color='white', alpha=0) # Make NaN transparent
        
        # Bounds for 1, 2, 3, 4
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.norm_classification = mcolors.BoundaryNorm(bounds, self.cmap_classification.N)

        # Proxy patch for legend
        self.polygon_boundary_patch = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, linestyle='--', label='Límite del Polígono de Intervención')

        self.NODATA_VALUE_OUTPUT = -9999.0 # Consistent nodata value for GeoTIFFs

        # NEW: Store generated per-polygon chart paths
        self.generated_per_polygon_chart_details = []

        # NEW: Store the desired order of social variable charts
        self.social_variable_chart_order_display_names = [
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

    def _save_geotiff(self, data, path, profile):
        output_profile = profile.copy()
        output_profile.update(
            {
                'dtype': rasterio.float32,
                'nodata': self.NODATA_VALUE_OUTPUT,
                'count': 1
            }
        )
        with rasterio.open(path, 'w', **output_profile) as dst:
            dst.write(np.nan_to_num(data, nan=self.NODATA_VALUE_OUTPUT).astype(rasterio.float32), 1)

    def _save_metrics(self):
        # Convert NumPy floats to Python floats for JSON serialization
        clean_kpis = {k: float(v) if v is not None else None for k, v in self.result.kpis.items()}
        with open(self.paths["metrics"], 'w') as f:
            json.dump(clean_kpis, f, indent=4)
        logger.info(f"   ✅ Metrics JSON saved to {self.paths['metrics']}")

    def _plot_and_save_map_helper(
        self, data_full_raster, title, filename_suffix, cmap, vmin, vmax, norm, cbar_label,
        current_polygon_geometry, current_polygon_id, row_start, col_start, row_end, col_end, clipped_extent, is_global=False
    ):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        if is_global:
            clipped_data = data_full_raster
        else:
            clipped_data = data_full_raster[row_start:row_end, col_start:col_end]

        if np.all(np.isnan(clipped_data)) or clipped_data.size == 0:
            logger.warning(f"  Advertencia: No hay datos válidos para {title} en el Polígono ID {current_polygon_id if current_polygon_id else 'global'} después del recorte. Omitiendo la generación del mapa.")
            plt.close()
            return None # Return None if chart not generated

        # Explicitly set x and y limits BEFORE adding the basemap
        ax.set_xlim(clipped_extent[0], clipped_extent[1])
        ax.set_ylim(clipped_extent[2], clipped_extent[3])

        # Add basemap AFTER setting limits, conditionally
        if not is_global:
            ctx.add_basemap(ax, crs=self.result.profile['crs'], source=ctx.providers.Esri.WorldImagery, alpha=1.0)

        # Set alpha based on whether it's a global map or a per-polygon map
        alpha_value = 1.0 if is_global else 0.5

        im = ax.imshow(
            clipped_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            extent=clipped_extent,
            interpolation='nearest' if norm else 'bilinear',
            alpha=alpha_value # Set alpha here
        )

        # Overlay polygon geometry only if it's a per-polygon map
        if not is_global and current_polygon_geometry is not None:
            gpd.GeoSeries([current_polygon_geometry]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
            ax.legend(handles=[self.polygon_boundary_patch], loc='lower right', bbox_to_anchor=(0.98, 0.02), frameon=True, fancybox=True, edgecolor='black', facecolor='white')

        # Colorbar setup
        if norm and len(self.LST_CLASS_CODES) > 0: # For classification maps
            # Map unique codes to labels for the colorbar ticks
            unique_codes = np.unique(clipped_data[~np.isnan(clipped_data)]).astype(int)
            cbar_ticks = [code for code in unique_codes if code in self.LST_CLASS_LABELS]
            cbar_labels = [self.LST_CLASS_LABELS[code] for code in cbar_ticks]

            if cbar_ticks:
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, ticks=cbar_ticks)
                cbar.ax.set_yticklabels(cbar_labels)
            else:
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7) # Fallback if no valid ticks
        else: # For continuous maps, let colorbar determine ticks automatically
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)

        cbar.set_label(cbar_label, fontsize=12)

        ax_title = f"{title}"
        if not is_global and current_polygon_id:
            ax_title = f"{title} - Polígono ID {current_polygon_id}"

        ax.set_title(ax_title, fontsize=14)

        # Add grid with coordinates
        # ax.set_xticks(np.linspace(clipped_extent[0], clipped_extent[1], 5))
        # ax.set_yticks(np.linspace(clipped_extent[2], clipped_extent[3], 5))
        # ax.xaxis.set_major_formatter(FormaLSTrFormatter('%.2f°'))
        # ax.yaxis.set_major_formatter(FormaLSTrFormatter('%.2f°'))
        # ax.tick_params(axis='x', rotation=0, labelsize=8)
        # ax.tick_params(axis='y', rotation=0, labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.set_xlabel('Longitud', fontsize=10)
        ax.set_ylabel('Latitud', fontsize=10)

        # Add intervention legend for global maps only
        if is_global: # Adjust for global maps
            # Overlay gdf_interventions for global maps (entire extent)
            self.gdf_interventions.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
            ax.legend(handles=[self.polygon_boundary_patch], bbox_to_anchor=(0.98, 0.02), loc='lower right', frameon=True, fancybox=True, edgecolor='black', facecolor='white')

        plt.tight_layout()
        output_filename = os.path.join(self.report_inputs_dir, f"{filename_suffix}{'_' + str(current_polygon_id) if not is_global else ''}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ Mapa guardado: {output_filename}")
        return output_filename # Return the path of the saved file



    def execute_export_pipeline(self):
        """Orchestrates the generation of all outputs."""
        logger.info("--- [PROCESSOR] Generating Outputs ---")
        self._save_metrics()
        self._generate_heatmaps_and_geotiffs()
        self._generate_lst_classification_maps()
        self._generate_ozone_charts()
        self._generate_energy_charts()
        self._generate_zonal_stats_charts()
        self._generate_pdf_report()
        return self.paths

    def get_web_payload(self):
        """Generates assets and returns data for web response."""
        # Generate all assets required for web view and report
        self._save_metrics()
        self._generate_heatmaps_and_geotiffs()
        self._generate_lst_classification_maps()
        self._generate_ozone_charts()
        self._generate_energy_charts()
        self._generate_zonal_stats_charts()
        
        return {
            "kpis": self.result.kpis,
            "paths": self.paths,
            "energy_impact": self.result.energy_summaries,
        }
    
    def export_to_geojson(self, simulation_result: SimulationResult, output_filename: str = "results.geojson"):
        """
        Convierte los resultados de la simulación y las estadísticas zonales 
        en un archivo GeoJSON consumible por el mapa del Frontend.
        """
        try:
            # 1. Recuperar el DataFrame de estadísticas zonales (población y temperatura por polígono)
            # Este DF fue generado en el SimulationEngine y guardado en el DTO
            df_stats = simulation_result.population_zonal_stats_df
            
            if df_stats is None or df_stats.empty:
                logger.warning("[Processor] No hay estadísticas zonales para exportar a GeoJSON.")
                return None

            # 2. Convertir a GeoDataFrame (asegurando que la geometría sea válida)
            gdf = gpd.GeoDataFrame(df_stats, geometry='geometry', crs=simulation_result.profile['crs'])
            
            # 3. Transformar a WGS84 (EPSG:4326) - REQUERIMIENTO ESTÁNDAR PARA WEB
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # 4. Limpieza de columnas para el JSON
            # Convertimos tipos de numpy a tipos nativos de Python para evitar errores de serialización
            output_path = os.path.join(self.simulations_dir, output_filename)
            
            # Seleccionamos solo las columnas clave para no inflar el peso del archivo
            columns_to_keep = [
                'geometry', 'intervention_type', 'avg_temp_reduction', 
                'population_affected', 'energy_savings_kwh'
            ]
            
            # Validamos que existan antes de filtrar
            existing_cols = [c for c in columns_to_keep if c in gdf.columns]
            
            # 5. Exportar
            gdf[existing_cols].to_file(output_path, driver='GeoJSON')
            
            logger.info(f"   ✅ GeoJSON de resultados generado en: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"   ❌ Error fatal al generar GeoJSON: {str(e)}")
            # Aquí es donde deberíamos usar nuestra excepción personalizada
            raise exceptions.CoreBaseException(f"Error en exportación vectorial: {e}")

    def generate_pdf_report(self):
        """Public method to generate the PDF report."""
        self._generate_pdf_report()
        return self.paths["report_pdf"]

    def export_geospatial_pkg(self):
        """Creates a ZIP package of geospatial outputs."""
        import zipfile
        zip_path = os.path.join(self.output_base_dir, "geospatial_pkg.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Simulation GeoTIFFs
            if os.path.exists(self.simulations_dir):
                for filename in os.listdir(self.simulations_dir):
                    if filename.endswith(".tif"):
                        zipf.write(os.path.join(self.simulations_dir, filename), 
                                   os.path.join("simulations", filename))
            
            # Add Classification GeoTIFFs
            if os.path.exists(self.lst_classification_dir):
                for filename in os.listdir(self.lst_classification_dir):
                    if filename.endswith(".tif"):
                        zipf.write(os.path.join(self.lst_classification_dir, filename),
                                   os.path.join("lst_classification", filename))
        return zip_path

    def _generate_heatmaps_and_geotiffs(self):
        """Generates visual assets (PNGs) and GeoTIFFs without a display."""

        # --- Generate GLOBAL PNGs (LST, Intervention, Difference) ---
        # Recalculate vmin_lst and vmax_lst based on percentiles for better visualization
        all_lst_for_percentiles = np.concatenate([
            self.result.base_lst[~np.isnan(self.result.base_lst)],
            self.result.intervention_lst[~np.isnan(self.result.intervention_lst)]
        ])

        vmin_lst_percentile = np.percentile(all_lst_for_percentiles, 2)
        vmax_lst_percentile = np.percentile(all_lst_for_percentiles, 98)

        west, south, east, north = array_bounds(
            self.result.profile['height'],
            self.result.profile['width'],
            self.result.profile['transform']
        )
        global_plot_extent = [west, east, south, north]

        # 1. Generate the 'Twin' LST global map
        self._plot_and_save_map_helper(self.result.base_lst, "LST Escenario Base", "lst_twin_map", plt.colormaps.get_cmap('RdYlBu_r'), vmin_lst_percentile, vmax_lst_percentile, None, 'Temperatura superficial (°C)', None, None, None, None, None, None, global_plot_extent, is_global=True)

        # 2. Generate the 'Intervention' LST global map
        self._plot_and_save_map_helper(self.result.intervention_lst, "LST Escenario Intervención", "lst_intervention_map", plt.colormaps.get_cmap('RdYlBu_r'), vmin_lst_percentile, vmax_lst_percentile, None, 'Temperatura superficial (°C)', None, None, None, None, None, None, global_plot_extent, is_global=True)

        # Calculate max_abs_diff_value for the difference map
        difference_data = self.result.difference_lst
        valid_difference_data = difference_data[~np.isnan(difference_data)]
        if valid_difference_data.size > 0:
            max_abs_diff_value = np.max(np.abs(valid_difference_data))
        else:
            max_abs_diff_value = 1 # Default to 1 if no valid data

        # 3. Generate the 'Difference' LST global map
        self._plot_and_save_map_helper(self.result.difference_lst, "Diferencia LST (Intervención - Base)", "heatmap_impact", plt.colormaps.get_cmap('RdBu_r'), -max_abs_diff_value, max_abs_diff_value, None, 'Temperatura superficial (°C)', None, None, None, None, None, None, global_plot_extent, is_global=True)

        logger.info(f"   ✅ Global Heatmap images (PNG) saved.")

        # --- Generate PER-POLYGON PNGs (LST, Intervention) ---
        for index, feature in self.gdf_interventions.iterrows():
            polygon_geometry = feature.geometry
            polygon_id = int(feature['id']) if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
            polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

            minx, miny, maxx, maxy = polygon_geometry.bounds
            window = from_bounds(minx, miny, maxx, maxy, self.result.profile['transform'],
                                 width=self.result.profile['width'], height=self.result.profile['height'])

            row_start = max(0, int(window.row_off))
            col_start = max(0, int(window.col_off))
            row_end = min(self.result.profile['height'], int(window.row_off + window.height))
            col_end = min(self.result.profile['width'], int(window.col_off + window.width))

            if row_start >= row_end or col_start >= col_end:
                logger.warning(f"  Polygon ID {polygon_id_str} does not intersect with the raster or is too small. Skipping per-polygon LST maps.")
                continue

            clipped_extent = [minx, maxx, miny, maxy]

            # Per-polygon Base LST map
            path = self._plot_and_save_map_helper(self.result.base_lst, "LST Escenario Base", "lst_twin_polygon", plt.colormaps.get_cmap('RdYlBu_r'), vmin_lst_percentile, vmax_lst_percentile, None, 'Temperatura superficial (°C)', polygon_geometry, polygon_id_str, row_start, col_start, row_end, col_end, clipped_extent, is_global=False)
            self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'lst_twin_polygon', 'path': path, 'title': "LST Escenario Base"})

            # Per-polygon Intervention LST map
            path = self._plot_and_save_map_helper(self.result.intervention_lst, "LST Escenario Intervención", "lst_intervention_polygon", plt.colormaps.get_cmap('RdYlBu_r'), vmin_lst_percentile, vmax_lst_percentile, None, 'Temperatura superficial (°C)', polygon_geometry, polygon_id_str, row_start, col_start, row_end, col_end, clipped_extent, is_global=False)
            self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'lst_intervention_polygon', 'path': path, 'title': "LST Escenario Intervención"})

            # Per-polygon Difference LST map (heatmap impact)
            path = self._plot_and_save_map_helper(self.result.difference_lst, "Diferencia LST (Intervención - Base)", "heatmap_impact", plt.colormaps.get_cmap('RdBu_r'), -max_abs_diff_value, max_abs_diff_value, None, 'Temperatura superficial (°C)', polygon_geometry, polygon_id_str, row_start, col_start, row_end, col_end, clipped_extent, is_global=False)
            self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'heatmap_impact', 'path': path, 'title': "Diferencia LST (Intervención - Base)"})
        logger.info(f"   ✅ Per-polygon Heatmap images (PNG) saved.")

        # --- Generate GeoTIFFs ---
        profile_for_export = self.result.profile.copy()
        self._save_geotiff(self.result.base_lst, self.paths["geotiff_base_lst"], profile_for_export)
        self._save_geotiff(self.result.intervention_lst, self.paths["geotiff_intervention_lst"], profile_for_export)
        self._save_geotiff(self.result.difference_lst, self.paths["geotiff_difference_lst"], profile_for_export)
        logger.info(f"   ✅ GeoTIFF files saved.")

    def _generate_lst_classification_maps(self):
        """Generates LST classification maps (global and per-polygon) and saves them to report_inputs."""
        logger.info("   Generating LST Classification maps...")

        # Prepare LST classification data from SimulationResult
        icu_classes_intervention_data = self.result.icu_classes
        icu_classes_base_data = self.result.base_icu_classes

        # Global extent
        west, south, east, north = array_bounds(
            self.result.profile['height'],
            self.result.profile['width'],
            self.result.profile['transform']
        )
        global_plot_extent = [west, east, south, north]


        # --- Save LST Classification GeoTIFFs ---
        profile_for_export = self.result.profile.copy()
        self._save_geotiff(icu_classes_base_data, os.path.join(self.lst_classification_dir, "lst_classification_twin_map.tif"), profile_for_export)
        self._save_geotiff(icu_classes_intervention_data, os.path.join(self.lst_classification_dir, "lst_classification_intervention_map.tif"), profile_for_export)
        logger.info(f"   ✅ LST Classification GeoTIFF files saved to {self.lst_classification_dir}.")


        # Generate the 'Base' LST Classification global map
        self._plot_and_save_map_helper(icu_classes_base_data, "Clasificación LST Escenario Base",
                                       "lst_classification_twin_map", self.cmap_classification,
                                       None, None, self.norm_classification, 'Categoría de LST',
                                       None, None, None, None, None, None, global_plot_extent, is_global=True)

        # Generate the 'Intervention' LST Classification global map
        self._plot_and_save_map_helper(icu_classes_intervention_data, "Clasificación LST Escenario Intervención",
                                       "lst_classification_intervention_map", self.cmap_classification,
                                       None, None, self.norm_classification, 'Categoría de LST',
                                       None, None, None, None, None, None, global_plot_extent, is_global=True)

        # Generate per-polygon LST Classification maps
        for index, feature in self.gdf_interventions.iterrows():
            polygon_geometry = feature.geometry
            polygon_id = int(feature['id']) if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
            polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

            minx, miny, maxx, maxy = polygon_geometry.bounds
            window = from_bounds(minx, miny, maxx, maxy, self.result.profile['transform'],
                                 width=self.result.profile['width'], height=self.result.profile['height'])

            row_start = max(0, int(window.row_off))
            col_start = max(0, int(window.col_off))
            row_end = min(self.result.profile['height'], int(window.row_off + window.height))
            col_end = min(self.result.profile['width'], int(window.col_off + window.width))

            if row_start >= row_end or col_start >= col_end:
                logger.warning(f"  Polygon ID {polygon_id_str} does not intersect with the raster or is too small. Skipping per-polygon LST classification map.")
                continue

            clipped_extent = [minx, maxx, miny, maxy]

            # Per-polygon Base LST classification map
            path = self._plot_and_save_map_helper(icu_classes_base_data, "Clasificación LST Escenario Base",
                                           "lst_classification_twin_polygon", self.cmap_classification,
                                           None, None, self.norm_classification, 'Categoría de LST',
                                           polygon_geometry, polygon_id_str, row_start, col_start, row_end, col_end, clipped_extent, is_global=False)
            self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'lst_classification_twin_polygon', 'path': path, 'title': "Clasificación LST Escenario Base"})

            # Per-polygon Intervention LST classification map
            path = self._plot_and_save_map_helper(icu_classes_intervention_data, "Clasificación LST Escenario Intervención",
                                           "lst_classification_intervention_polygon", self.cmap_classification,
                                           None, None, self.norm_classification, 'Categoría de LST',
                                           polygon_geometry, polygon_id_str, row_start, col_start, row_end, col_end, clipped_extent, is_global=False)
            self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'lst_classification_intervention_polygon', 'path': path, 'title': "Clasificación LST Escenario Intervención"})
        logger.info("   LST Classification maps generated.")

    def _generate_ozone_charts(self):
        """Generates ozone donut charts (global and per-polygon) and saves them to report_inputs."""
        logger.info("   Generating Ozone charts...")

        # Global ozone donut chart
        percentage_improvement = self.result.kpis.get('ozone_percentage_improvement')
        if percentage_improvement is not None and not np.isnan(percentage_improvement):
            path = self._plot_and_save_donut_chart(percentage_improvement, 'Global', 'global')
            if path:
                self.generated_per_polygon_chart_details.append({'polygon_id': 'Global', 'chart_type': 'ozone_donut', 'path': path, 'title': 'Porcentaje de cambio en la concentración de ozono (O3) Global'})
        else:
            logger.warning("  Ozone percentage improvement is NaN for global, skipping donut chart.")

        # Per-polygon ozone donut charts (requires iterating over polygons and recalculating)
        # Calculate per-polygon ozone improvement metrics
        for index, feature in self.gdf_interventions.iterrows():
            polygon_geometry = feature.geometry
            polygon_id = int(feature['id']) if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
            polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

            # Rasterize the current polygon to create its specific mask at LST resolution
            polygon_mask_data = rasterize(
                shapes=[(polygon_geometry, 1)],
                out_shape=(self.result.profile['height'], self.result.profile['width']),
                transform=self.result.profile['transform'],
                fill=0,
                dtype='uint8'
            )
            polygon_pixel_mask_boolean = (polygon_mask_data == 1)

            # Clip ozone data to the current polygon's mask
            ozone_twin_clipped = self.result.ozone_twin.copy()
            ozone_twin_clipped[~polygon_pixel_mask_boolean] = np.nan

            ozone_intervention_clipped = self.result.ozone_intervention.copy()
            ozone_intervention_clipped[~polygon_pixel_mask_boolean] = np.nan

            # Calculate total ozone for the clipped areas
            total_ozone_twin_polygon = np.nansum(ozone_twin_clipped)
            total_ozone_intervention_polygon = np.nansum(ozone_intervention_clipped)

            polygon_percentage_improvement = np.nan
            if total_ozone_twin_polygon > 0:
                polygon_percentage_improvement = (1 - (total_ozone_intervention_polygon / total_ozone_twin_polygon)) * 100

            if not np.isnan(polygon_percentage_improvement):
                path = self._plot_and_save_donut_chart(polygon_percentage_improvement, f'Polígono ID {polygon_id_str}', f'polygon_{polygon_id_str}')
                if path:
                    self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'ozone_donut', 'path': path, 'title': f'Porcentaje de cambio en la concentración de ozono (O3) - Polígono ID {polygon_id_str}'})
            else:
                logger.warning(f"  Ozone percentage improvement is NaN for Polygon ID {polygon_id_str}, skipping donut chart.")

        logger.info("   Ozone charts generated.")

    def _plot_and_save_donut_chart(self, percentage_improvement_val, title_suffix, filename_suffix):
        if np.isnan(percentage_improvement_val):
            logger.warning(f"❌ No se pudo generar el gráfico de donut de {title_suffix} debido a datos no válidos.")
            return None # Return None if chart not generated

        # Determine colors based on improvement value
        if percentage_improvement_val >= 0:
            chart_colors = ['#a4b465', '#cccccc'] # Green for improvement, Gray for remaining
            central_text_label = 'Reducción'
        else:
            # For negative improvement, treat it as an increase/worsening
            chart_colors = ['#FF0000', '#cccccc'] # Red for worsening, Gray for remaining
            central_text_label = 'Incremento'

        # Use absolute value for slice sizing, but original for text
        display_magnitude = abs(percentage_improvement_val)

        # Ensure display_magnitude is within [0, 100] for visualization
        display_size = max(0, min(100, display_magnitude))
        remaining_size = 100 - display_size

        # Data for the donut chart
        sizes = [display_size, remaining_size]
        # Set labels to None to suppress individual slice labels
        labels = None
        explode = (0, 0) # No explode effect

        fig, ax = plt.subplots(figsize=(8 * 0.8, 8 * 0.8)) # Reduced figsize by 20%
        # Set counterclock=False for clockwise drawing
        wedges, texts = ax.pie(sizes, explode=explode, labels=labels, colors=chart_colors,
                                          autopct=None, startangle=90, counterclock=False,
                                          wedgeprops=dict(width=0.3, edgecolor='w'))

        # Draw a circle in the center to make it a donut
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)

        # Add central text label using the original percentage value
        central_text = f'{percentage_improvement_val:.2f}%\n{central_text_label}'
        ax.text(0, 0, central_text, ha='center', va='center', fontsize=16, color='black', weight='bold')

        # ax.set_title(f'Porcentaje de cambio en la concentración de ozono (O3) {title_suffix}', fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.tight_layout()
        output_filename = os.path.join(self.report_inputs_dir, f'ozone_improvement_donut_chart_{filename_suffix}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"✅ Gráfico de donut de Mejora de Ozono ({title_suffix}) guardado en: {output_filename}")
        return output_filename # Return the path of the saved file

    def _generate_energy_charts(self):
        """Generates energy consumption bar charts (global and per-polygon) and saves them to report_inputs."""
        logger.info("   Generating Energy Consumption charts...")

        # Global energy consumption bar chart
        total_sum_twin = self.result.kpis.get('total_energy_consumption_twin_MWh_year')
        total_sum_intervention = self.result.kpis.get('total_energy_consumption_intervention_MWh_year')

        if total_sum_twin is not None and total_sum_intervention is not None and (abs(total_sum_twin) > 1e-6 or abs(total_sum_intervention) > 1e-6):
            path = self._plot_and_save_energy_bar_chart('Global', 'global', total_sum_twin, total_sum_intervention)
            if path:
                self.generated_per_polygon_chart_details.append({'polygon_id': 'Global', 'chart_type': 'energy_bar', 'path': path, 'title': 'Consumo Residencial Total de Electricidad Escalado por Población (Global)'})
        else:
            logger.warning("  Total energy consumption values are missing, zero, or NaN for global, skipping bar chart.")

        # Per-polygon energy consumption bar charts
        summaries = getattr(self.result, 'energy_summaries', [])
        for summary in summaries:
            if summary['polygon_identifier'] != 'Global':
                polygon_id_str = summary['polygon_identifier'].replace('Polígono ID ', '') # Extract just the ID
                total_sum_twin_polygon = summary['actual_total_sum_twin_consumption_scaled']
                total_sum_intervention_polygon = summary['actual_total_sum_intervention_consumption_scaled']

                if not np.isnan(total_sum_twin_polygon) and not np.isnan(total_sum_intervention_polygon) and (abs(total_sum_twin_polygon) > 1e-6 or abs(total_sum_intervention_polygon) > 1e-6):
                    path = self._plot_and_save_energy_bar_chart(f'Polígono ID {polygon_id_str}', f'Polígono_ID_{polygon_id_str}', total_sum_twin_polygon, total_sum_intervention_polygon)
                    if path:
                        self.generated_per_polygon_chart_details.append({'polygon_id': polygon_id_str, 'chart_type': 'energy_bar', 'path': path, 'title': f'Consumo Residencial Total de Electricidad - Polígono ID {polygon_id_str}'})
                else:
                    logger.warning(f"  Energy consumption values are NaN, zero, or missing for Polígono ID {polygon_id_str}, skipping bar chart.")

        logger.info("   Energy Consumption charts generated.")

    def _plot_and_save_energy_bar_chart(self, title_suffix, filename_suffix, total_sum_twin, total_sum_intervention):
        scenario_names = ['Base', 'Intervención']
        sum_values = [total_sum_twin, total_sum_intervention]
        colors = ['#626f47', '#a4b465'] # Custom colors for consistency

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(scenario_names, sum_values, color=colors)

        ax.set_ylabel('Total MWh/año', fontsize=12)
        ax.set_title(f'Consumo Residencial Total de Electricidad ({title_suffix})', fontsize=14)
        ax.set_xticks(scenario_names)
        ax.tick_params(axis='x', rotation=0)

        for i, v in enumerate(sum_values):
            ax.text(i, v + (max(sum_values)*0.01), f'{v:,.0f}', ha='center', va='bottom', fontsize=10, color='black', weight='bold')

        plt.tight_layout()
        output_filename = os.path.join(self.report_inputs_dir, f'total_energy_consumption_bar_chart_{filename_suffix}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"✅ Bar chart de Consumo de Energía ({title_suffix}) guardado en: {output_filename}")
        return output_filename # Return the path of the saved file

    def _generate_zonal_stats_charts(self):
        """Generates zonal statistics bar charts (global and per-polygon) and saves them to report_inputs."""
        logger.info("   Generating Zonal Statistics charts...")

        LST_ZONE_ORDER = ['Fondo Térmico Urbano', 'Zona de transición', 'Zona de impacto', 'Zona núcleo']
        custom_scenario_colors = {
            'Base': '#626f47',
            'Intervención': '#a4b465'
        }
        plot_column_order = ['Base', 'Intervención']

        if not self.zonal_stats_df.empty:
            self.zonal_stats_df['LST Zone'] = self.zonal_stats_df['zone_id'].map({
                1: 'Fondo Térmico Urbano',
                2: 'Zona de transición',
                3: 'Zona de impacto',
                4: 'Zona núcleo'
            })

            # --- Save Zonal Statistics to CSV ---
            # Use self.zonal_stats_path for the output CSV path, which is defined in __init__
            try:
                self.zonal_stats_df.to_csv(self.zonal_stats_path, index=False)
                logger.info(f"   ✅ Zonal statistics successfully saved to {self.zonal_stats_path}")
            except Exception as e:
                logger.error(f"   ❌ Error saving zonal statistics to CSV: {e}")


            # --- Generate Global Zonal Statistics Plots ---
            logger.info("     Generating Global Zonal Statistics Plots...")
            # Iterate through the predefined order of social variable charts
            for band_name_display in self.social_variable_chart_order_display_names:
                safe_source_band_name = self._sanitize_band_name(band_name_display)
                chart_filename = f"zonal_stats_percentage_Global_{safe_source_band_name}.png"
                chart_path_zonal = os.path.join(self.report_inputs_dir, chart_filename)

                plot_data = self.zonal_stats_df[
                    (self.zonal_stats_df['polygon_id'] == 'Global') &
                    (self.zonal_stats_df['source_band'] == band_name_display)
                ].copy()

                # MEJORA DE VALIDACIÓN: Si no hay datos, borramos cualquier archivo viejo y saltamos
                if plot_data.empty or (plot_data['sum'].fillna(0).abs() < 1e-9).all():
                    if os.path.exists(chart_path_zonal):
                        os.remove(chart_path_zonal) # Elimina basura de ejecuciones previas
                    logger.warning(f"Omitiendo gráfica vacía: {band_name_display}")
                    continue

                # Solo si pasa el filtro anterior, se ejecuta el código de plt.figure() y plt.savefig()
                fig, ax = plt.subplots(figsize=(10, 6))
                
                pivot_df_perc_sum = plot_data.pivot_table(
                    index='LST Zone',
                    columns='Scenario',
                    values='percentage_of_sum'
                ).reindex(LST_ZONE_ORDER)

                for col in plot_column_order:
                    if col not in pivot_df_perc_sum.columns:
                        pivot_df_perc_sum[col] = np.nan
                pivot_df_perc_sum = pivot_df_perc_sum[plot_column_order]

                # Final check on pivoted data to ensure it's not all-NaN or all-zero before plotting
                if pivot_df_perc_sum.isnull().all().all() or (pivot_df_perc_sum.fillna(0).abs() < 1e-9).all().all():
                    logger.warning(f"       Pivoted data for Global, Source Band '{band_name_display}' is empty or all zero. Skipping plot.")
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                plot = pivot_df_perc_sum.plot(kind='bar', ax=ax, color=[custom_scenario_colors[col] for col in pivot_df_perc_sum.columns])
                ax.set_title(f'Global: Porcentaje de {band_name_display} por Zona LST', fontsize=14)
                ax.set_ylabel('Porcentaje', fontsize=12)
                ax.set_xlabel('Clasificación de Zonas', fontsize=12)
                ax.tick_params(axis='x', rotation=0)
                ax.legend(title='Escenario')

                for container in plot.containers:
                    for patch in container:
                        if not np.isnan(patch.get_height()) and patch.get_height() > 0:
                            ax.annotate(f'{patch.get_height():.1f}%',
                                         (patch.get_x() + patch.get_width() / 2,
                                          patch.get_height()), ha='center', va='bottom', fontsize=9, color='black', weight='bold')
                plt.tight_layout()
                output_zonal_stats_chart_filename = os.path.join(self.report_inputs_dir, chart_filename) # Use chart_filename
                plt.savefig(chart_path_zonal, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"       Plot saved: {output_zonal_stats_chart_filename}")

            # --- Generate Per-Polygon Zonal Statistics Plots ---
            logger.info("     Generating Per-Polygon Zonal Statistics Plots...")
            unique_polygon_ids = self.zonal_stats_df[self.zonal_stats_df['polygon_id'] != 'Global']['polygon_id'].unique()

            for polygon_id in unique_polygon_ids:
                for band_name_display in self.social_variable_chart_order_display_names: # Iterate using the ordered list
                    plot_data = self.zonal_stats_df[
                        (self.zonal_stats_df['polygon_id'] == polygon_id) &
                        (self.zonal_stats_df['source_band'] == band_name_display)
                    ].copy()

                    # Check if the underlying sum data is empty or all zero before attempting to plot percentages.
                    if plot_data.empty or (plot_data['sum'].fillna(0).abs() < 1e-9).all():
                        logger.warning(f"       Sum data for Polygon {polygon_id}, Source Band '{band_name_display}' is empty or all zero. Skipping plot.")
                        continue

                    pivot_df_perc_sum = plot_data.pivot_table(
                        index='LST Zone',
                        columns='Scenario',
                        values='percentage_of_sum'
                    ).reindex(LST_ZONE_ORDER)

                    for col in plot_column_order:
                        if col not in pivot_df_perc_sum.columns:
                            pivot_df_perc_sum[col] = np.nan
                    pivot_df_perc_sum = pivot_df_perc_sum[plot_column_order]

                    # Final check on pivoted data to ensure it's not all-NaN or all-zero before plotting
                    if pivot_df_perc_sum.isnull().all().all() or (pivot_df_perc_sum.fillna(0).abs() < 1e-9).all().all():
                        logger.warning(f"       Pivoted data for Polygon {polygon_id}, Source Band '{band_name_display}' is empty or all zero. Skipping plot.")
                        continue

                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot = pivot_df_perc_sum.plot(kind='bar', ax=ax, color=[custom_scenario_colors[col] for col in pivot_df_perc_sum.columns])
                    ax.set_title(f'Polígono {polygon_id}: Porcentaje de {band_name_display} por Zona LST', fontsize=14)
                    ax.set_ylabel('Porcentaje', fontsize=12)
                    ax.set_xlabel('Clasificación de Zonas', fontsize=12)
                    ax.tick_params(axis='x', rotation=0)
                    ax.legend(title='Escenario')

                    for container in plot.containers:
                        for patch in container:
                            if not np.isnan(patch.get_height()) and patch.get_height() > 0:
                                ax.annotate(f'{patch.get_height():.1f}%',
                                             (patch.get_x() + patch.get_width() / 2,
                                              patch.get_height()), ha='center', va='bottom', fontsize=9, color='black', weight='bold')
                    plt.tight_layout()

                    safe_source_band_name = self._sanitize_band_name(band_name_display)
                    output_zonal_stats_chart_filename = os.path.join(self.report_inputs_dir, f"per_polygon_zonal_stats_polygon_{polygon_id}_{safe_source_band_name}.png")
                    plt.savefig(output_zonal_stats_chart_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"       Plot saved: {output_zonal_stats_chart_filename}")
                    self.generated_per_polygon_chart_details.append({'polygon_id': str(polygon_id), 'chart_type': 'zonal_stats', 'source_band': band_name_display, 'path': output_zonal_stats_chart_filename, 'title': f'Porcentaje de {band_name_display} por Zona LST'})

            logger.info("   All Zonal Statistics Plots Generated and Saved.")
        else:
            logger.warning("   No zonal statistics DataFrame found or it is empty. Skipping plotting.")

        logger.info("   Zonal Statistics charts generated.")



    def _generate_pdf_report(self):
        """Compiles a summary PDF."""
        doc = SimpleDocTemplate(self.paths["report_pdf"], pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom Styles must be defined at the beginning and unconditionally
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='Heading2Custom', parent=styles['Heading2'], spaceAfter=6, spaceBefore=12))
        styles.add(ParagraphStyle(
            name='IntroJustify',
            parent=styles['Normal'],
            alignment=TA_JUSTIFY,
            fontSize=10,
            leading=16,
            spaceAfter=12,
            allowMarkup=1 # Enable HTML markup
        ))
        styles.add(ParagraphStyle(
            name='Footnote',
            parent=styles['Normal'],
            fontSize=7,
            leading=10,
            textColor=colors.gray,
            leftIndent=0,
            spaceBefore=5,
            allowMarkup=1 # Enable HTML markup
        ))
        styles.add(ParagraphStyle(name='CodeStyle', parent=styles['Normal'], fontSize=7, leading=14, borderPadding=5, backColor=colors.white, minHeight=10))

        # Define style_center unconditionally
        style_center = ParagraphStyle(
            name='Centro',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            spaceAfter=12
        )
        styles.add(style_center)

        # Modify the existing 'Bullet' style or add if it somehow doesn't exist
        if 'Bullet' in styles:
            bullet_style = styles['Bullet']
            bullet_style.firstLineIndent = 0
            bullet_style.leftIndent = 20
            bullet_style.bulletIndent = 10
            bullet_style.spaceBefore = 3
            bullet_style.spaceAfter = 3
            bullet_style.parent = styles['Normal'] # Ensure parent is set if modifying
            bullet_style.allowMarkup = 1 # Enable HTML markup
        else:
            # Fallback for adding Bullet style if not found, also with allowMarkup
            styles.add(ParagraphStyle(
                  name='Bullet',
                  parent=styles['Normal'],
                  firstLineIndent=0,
                  leftIndent=20,
                  bulletIndent=10,
                  spaceBefore=3,
                  spaceAfter=3,
                  allowMarkup=1 # Enable HTML markup
              ))

        story = []

        # 1. Format the date string in Spanish first
        fecha_texto = format_date(datetime.now(), format="d 'de' MMMM 'de' yyyy", locale='es_MX')

        # 2. RETRIEVE MUNICIPALITY NAME AND STATE NAME
        municipality_name = self.municipality_name # Use directly from self
        state_name = self.state_name # Use directly from self

        # NEW: Logo Paths (using local paths)
        LOCAL_LOGO_DIR = self.local_logo_dir
        LOGO_CAPSUS_PATH = os.path.join(LOCAL_LOGO_DIR, 'CAPSUS_logo.png')
        LOGO_CAME_PATH = os.path.join(LOCAL_LOGO_DIR, 'CAMe_logo.jpg')
        LOGO_CEURE_PATH = os.path.join(LOCAL_LOGO_DIR, 'CEURE_logo.jpg')

        # --- DEFINE HEADER & FOOTER (Helper functions for ReportLab) ---
        def header_footer(canvas, doc):
            canvas.saveState()
            # Header
            header_text = f"Evaluación de acciones y proyectos para reducir la ICU - {municipality_name}"
            canvas.setFont('Helvetica-Bold', 10)
            canvas.drawString(inch, letter[1] - 0.5 * inch, header_text)
            canvas.line(inch, letter[1] - 0.6 * inch, letter[0] - inch, letter[1] - 0.6 * inch)

            # Footer
            footer_text = f"Proyecto - {self.project_name}"
            canvas.setFont('Helvetica', 9)
            canvas.drawString(inch, 0.6 * inch, footer_text)

            # Page Number
            page_num = canvas.getPageNumber()
            text = "Página %d" % page_num
            canvas.drawRightString(letter[0] - inch, 0.6 * inch, text)
            canvas.line(inch, 0.85 * inch, letter[0] - inch, 0.85 * inch)

            canvas.restoreState()

        # Helper function to load and resize images for ReportLab
        def get_image_for_reportlab(image_path, max_width, max_height=None):
            if not os.path.exists(image_path):
                logger.warning(f"Advertencia: Imagen no encontrada en {image_path}. No se cargará.")
                return None
            try:
                pil_img = PILImage.open(image_path)
                original_width_px, original_height_px = pil_img.size
                
                if original_width_px <= 0 or original_height_px <= 0:
                    logger.warning(f"Advertencia: Dimensiones de imagen inválidas en {image_path} ({original_width_px}x{original_height_px}).")
                    return None

                scale_factor = max_width / original_width_px
                final_width = original_width_px * scale_factor
                final_height = original_height_px * scale_factor

                if max_height is not None and final_height > max_height:
                    scale_factor = max_height / original_height_px
                    final_width = original_width_px * scale_factor
                    final_height = original_height_px * scale_factor
                
                # Safety check for absurdly large heights (e.g. > page height) which cause layout crashes
                if final_height > 10 * inch:
                    logger.warning(f"Advertencia: Altura de imagen excesiva ({final_height}) en {image_path}. Ajustando a límite seguro.")
                    ratio = (10 * inch) / final_height
                    final_height = 10 * inch
                    final_width = final_width * ratio

                return Image(image_path, width=final_width, height=final_height)
            except Exception as e:
                logger.error(f"Error al cargar o procesar la imagen {image_path}: {e}")
                return None


        # Initial global figure counter
        current_global_fig_num = 1

        # Max width for images in PDF, maintaining aspect ratio
        max_pdf_img_width = 6 * inch

        # --- TITLE PAGE ---
        story.append(Spacer(1, 1 * inch))
        story.append(Paragraph("Reporte técnico de simulación de acciones y proyectos para reducir la Isla de Calor Urbana (ICU)", styles['CenterTitle']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"{self.project_name}", styles['CenterTitle']))
        story.append(Spacer(1, 4.5 * inch))
        story.append(Paragraph(f"Fecha: {fecha_texto}", style_center))
        story.append(Paragraph("Reporte Automatizado", style_center))
        story.append(Spacer(1, 0.5 * inch))

        # Logos table for the bottom
        logos_for_table = []
        MAX_LOGO_DISPLAY_WIDTH_POINTS = 1.5 * inch
        MAX_LOGO_DISPLAY_HEIGHT_POINTS = 0.8 * inch

        logo_paths = [LOGO_CAME_PATH, LOGO_CAPSUS_PATH, LOGO_CEURE_PATH]
        for logo_path in logo_paths:
            logo_image = get_image_for_reportlab(logo_path, MAX_LOGO_DISPLAY_WIDTH_POINTS, MAX_LOGO_DISPLAY_HEIGHT_POINTS)
            if logo_image:
                logos_for_table.append(logo_image)
            else:
                logos_for_table.append(Paragraph(f"Logo faltante ({os.path.basename(logo_path)})", styles['Normal']))

        if logos_for_table:
            num_logos = len(logos_for_table)
            col_widths = [doc.width / num_logos for _ in range(num_logos)]

            logo_table = Table([logos_for_table], colWidths=col_widths)
            logo_table.hAlign = 'CENTER'
            logo_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(logo_table)

        story.append(PageBreak()) # Force new page after title

        # --- RESUMEN EJECUTIVO (NEW) ---
        # 1. Extraer variables clave dinámicas de la simulación
        ozone_impr = self.result.kpis.get('ozone_percentage_improvement', 0)
        ozone_str = f"{ozone_impr:.2f}%" if ozone_impr is not None and not np.isnan(ozone_impr) else "N/D"

        energy_twin = self.result.kpis.get('total_energy_consumption_twin_MWh_year', 0)
        energy_savings = self.result.kpis.get('energy_savings_MWh', 0)
        
        if energy_twin and not np.isnan(energy_twin) and energy_twin > 0:
            energy_perc = (energy_savings / energy_twin) * 100
            energy_perc_str = f"{energy_perc:.2f}%"
        else:
            energy_perc_str = "N/D"
        
        energy_savings_str = f"{energy_savings:.0f} MWh" if energy_savings is not None and not np.isnan(energy_savings) else "N/D"

        # 2. Construir la narrativa del Resumen Ejecutivo en el PDF
        story.append(Paragraph("Resumen Ejecutivo", styles['Heading1']))
        #story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Estimación de los posibles impactos de las acciones de prevención, adaptación y/o mitigación (a través de la Herramienta de evaluación de proyectos de mitigación de las ICU)</b>", styles['IntroJustify']))
        #story.append(Spacer(1, 12))
        
        story.append(Paragraph("El Reto del Calor Urbano", styles['Heading2Custom']))
        story.append(Paragraph("En la actualidad, el asfalto, el concreto y la falta de áreas verdes en nuestras ciudades absorben y retienen el calor del sol, creando lo que se conoce como <b>Islas de Calor Urbana (ICU)</b>. Este fenómeno provoca que ciertas zonas de la ciudad experimenten temperaturas mucho más altas que sus alrededores, afectando nuestra salud, aumentando el gasto en electricidad y empeorando la calidad del aire que respiramos.", styles['IntroJustify']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Para enfrentar este desafío en el municipio de <b>{municipality_name}</b>, se utilizó tecnología avanzada de simulación (un 'gemelo digital') para probar distintas soluciones antes de construirlas en la vida real. El objetivo es responder a una pregunta clave: <i>¿Qué pasaría si transformamos nuestros espacios urbanos con infraestructura más fresca y natural?</i>", styles['IntroJustify']))
        #story.append(Spacer(1, 12))

        story.append(Paragraph("Las Acciones Evaluadas", styles['Heading2Custom']))
        story.append(Paragraph("En este análisis, simulamos la implementación de medidas estratégicas en zonas específicas de la ciudad. Estas acciones incluyeron:", styles['IntroJustify']))
        story.append(Paragraph("<b>Arborización y vegetación:</b> Integración de nuevas áreas verdes con follaje escaso y moderado.", styles['Bullet']))
        story.append(Paragraph("<b>Techos fríos:</b> Aplicación de recubrimientos reflectantes en las azoteas de las construcciones para rebotar la luz solar.", styles['Bullet']))
        story.append(Paragraph("<b>Pavimentos frescos:</b> Sustitución de superficies viales tradicionales por materiales que absorben menos calor.", styles['Bullet']))
        #story.append(Spacer(1, 12))

        story.append(Paragraph("Impactos y Beneficios Esperados", styles['Heading2Custom']))
        story.append(Paragraph("Al comparar las condiciones actuales de la ciudad ('Escenario Base') contra nuestra ciudad mejorada ('Escenario de Intervención'), los resultados demuestran que estas acciones generan beneficios inmediatos y profundos en tres áreas fundamentales para nuestra calidad de vida:", styles['IntroJustify']))
        story.append(Spacer(1, 6))
        
        story.append(Paragraph("<b>1. Protección a las Personas y Enfriamiento de la Ciudad</b>", styles['Normal']))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Las intervenciones logran una reducción significativa de la temperatura en las calles y hogares.", styles['IntroJustify']))
        story.append(Paragraph("<b>Menos personas en riesgo:</b> La simulación muestra un desplazamiento masivo de habitantes que dejarían de vivir en 'Zonas de Impacto' (áreas de alto estrés térmico) para pasar a zonas de confort ('Fondo Térmico Urbano').", styles['Bullet']))
        story.append(Paragraph("<b>Cuidado a los más vulnerables:</b> Este enfriamiento beneficia directamente a la primera infancia y a los adultos mayores, reduciendo drásticamente su vulnerabilidad ante golpes de calor y deshidratación.", styles['Bullet']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>2. Ahorro Económico y Energético</b>", styles['Normal']))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Al tener hogares y calles más frescas, la necesidad de utilizar ventiladores o aire acondicionado disminuye notablemente.", styles['IntroJustify']))
        story.append(Paragraph(f"<b>Reducción en el recibo de luz:</b> Se estima que estas acciones generarían un <b>ahorro del {energy_perc_str}</b> en el consumo total residencial de electricidad en las áreas analizadas.", styles['Bullet']))
        story.append(Paragraph(f"<b>Impacto global:</b> Esto equivale a dejar de consumir aproximadamente <b>{energy_savings_str} al año</b>, aliviando no solo el bolsillo de las familias, sino también la presión sobre la red eléctrica municipal.", styles['Bullet']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>3. Aire Más Limpio y Sano</b>", styles['Normal']))
        story.append(Spacer(1, 4))
        story.append(Paragraph("El calor extremo funciona como un 'motor' que acelera la creación de contaminación, especialmente del ozono, un gas que irrita las vías respiratorias.", styles['IntroJustify']))
        story.append(Paragraph(f"<b>Menos contaminación:</b> Al bajar la temperatura de la ciudad, la simulación proyecta una <b>reducción del {ozone_str} en la formación de ozono troposférico</b>.", styles['Bullet']))
        story.append(Paragraph("<b>Mejor salud pública:</b> Respirar aire más limpio se traduce en menos enfermedades respiratorias y cardiovasculares para la población.", styles['Bullet']))
        
        # --- RESUMEN GRÁFICO (GRID TIPO FACET_WRAP) ---
        story.append(Spacer(1, 12))
        story.append(Paragraph("Resumen Visual de Impactos Globales", styles['Heading2Custom']))
        story.append(Spacer(1, 10))

        # --- 1. Grid de 2 celdas para Ahorro y Ozono ---
        # Ancho de columna para el grid de 2 celdas
        grid_col_width = doc.width / 2.0
        
        # Recopilamos las imágenes, ajustando su tamaño
        energy_chart_path = os.path.join(self.report_inputs_dir, 'total_energy_consumption_bar_chart_global.png')
        energy_chart_img = get_image_for_reportlab(energy_chart_path, grid_col_width - 10)
        ozone_chart_path = os.path.join(self.report_inputs_dir, 'ozone_improvement_donut_chart_global.png')
        ozone_chart_img = get_image_for_reportlab(ozone_chart_path, grid_col_width - 10)
        
        # Solo armamos el grid si las imágenes existen
        if energy_chart_img and ozone_chart_img:
            # Reducir el tamaño de la imagen de ozono para que coincida visualmente
            if ozone_chart_img:
                ozone_chart_img.drawHeight *= 0.75
                ozone_chart_img.drawWidth *= 0.75

            # Contenido para cada celda
            ahorro_content = [
                Paragraph("<b>Ahorro de Energía</b>", styles['Centro']),
                Spacer(1, 4),
                energy_chart_img
            ]
            
            reduccion_content = [
                Paragraph("<b>Reducción Ozono</b>", styles['Centro']),
                Spacer(1, 4),
                ozone_chart_img
            ]

            # Estructura de la tabla 1x2
            data = [[ahorro_content, reduccion_content]]
            
            grid_table = Table(data, colWidths=[grid_col_width, grid_col_width])
            grid_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
            ]))
            
            story.append(KeepTogether([grid_table]))
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

        # --- 2. Imagen de Enfriamiento LST (ancho completo) ---
        # El ancho máximo de la imagen será el ancho del documento menos un pequeño margen
        max_img_width = doc.width - 20 
        impact_map_img = get_image_for_reportlab(self.paths.get("map_impact"), max_img_width)
        
        if impact_map_img:
            enfriamiento_content = KeepTogether([
                Paragraph("<b>Enfriamiento LST</b>", styles['Centro']),
                Spacer(1, 4),
                impact_map_img
            ])
            story.append(enfriamiento_content)
            #story.append(PageBreak()) 

        # --- RESUMEN GRÁFICO (GRID TIPO FACET_WRAP) ---
        story.append(Paragraph("Resumen Visual de Impactos Globales", styles['Heading2Custom']))
        
        # Calculamos el ancho disponible dividiendo la página en 3 columnas (dejando un margen)
        grid_col_width = (doc.width / 3.0) - 6 
        
        # Recopilamos las imágenes globales ya generadas
        impact_map_img = get_image_for_reportlab(self.paths.get("map_impact"), grid_col_width)
        energy_chart_path = os.path.join(self.report_inputs_dir, 'total_energy_consumption_bar_chart_global.png')
        energy_chart_img = get_image_for_reportlab(energy_chart_path, grid_col_width)
        ozone_chart_path = os.path.join(self.report_inputs_dir, 'ozone_improvement_donut_chart_global.png')
        ozone_chart_img = get_image_for_reportlab(ozone_chart_path, grid_col_width)
        
        # Solo armamos el grid si las imágenes existen
        if impact_map_img and energy_chart_img and ozone_chart_img:
            grid_titles = [
                Paragraph("<b>Enfriamiento LST</b>", styles['Centro']),
                Paragraph("<b>Ahorro de Energía</b>", styles['Centro']),
                Paragraph("<b>Reducción Ozono</b>", styles['Centro'])
            ]
            grid_images = [impact_map_img, energy_chart_img, ozone_chart_img]
            
            # Matriz de la tabla (Fila 1: Títulos, Fila 2: Imágenes)
            grid_table = Table([grid_titles, grid_images], colWidths=[grid_col_width]*3)
            grid_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 2),
                ('RIGHTPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
            ]))
            
            story.append(KeepTogether([grid_table])) # Evita que se divida en dos páginas
            story.append(Spacer(1, 12))

        story.append(Paragraph("Conclusión", styles['Heading2Custom']))
        story.append(Paragraph(f"Los datos revelan que combatir las Islas de Calor no es solo un tema de confort, sino una inversión directa en salud pública, economía familiar y sostenibilidad ambiental. Implementar infraestructura verde, techos y pavimentos fríos en {municipality_name} es una estrategia altamente efectiva para proteger a la ciudadanía y construir una ciudad más resiliente ante el cambio climático.", styles['IntroJustify']))
        
        story.append(PageBreak()) # Force new page after executive summary


        # --- SECTION 1: INTRODUCTION ---

        story.append(Paragraph("1. Introducción", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"En respuesta a los retos que plantean las islas de calor urbana (ICU) en la megalópolis, la Comisión Ambiental de la Megalópolis (CAMe) dentro del 'Proyecto Piloto de identificación y evaluación de acciones y proyectos estratégicos para atención de las Islas de Calor Urbanas en la Megalópolis', ha priorizado determinar y evaluar acciones y proyectos puntuales de prevención, adaptación y mitigación que protejan a la población y el medio ambiente. Esta iniciativa es clave para abordar el desafío de las ICU en 10 demarcaciones territoriales de la Megalópolis, seleccionadas a partir de estudios previos de la UNAM financiados por CONAHCyT y SECTEI, que ya identificaron la presencia y evolución de las ICU a nivel municipal.", styles['IntroJustify']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>El objetivo de este reporte técnico es mostrar el impacto de diversas intervenciones urbanas (infraestructura verde, techos reflectivos, etc.) para mitigar el efecto de isla de calor en el municipio de {municipality_name}</b>. Este reporte presenta un análisis del impacto de diversas intervenciones urbanas en la Temperatura Superficial Terrestre (LST) así como sus posibles efectos en el consumo de energía y la calidad del aire. Para ello, se comparan dos escenarios: un escenario base, construído con información actual, y un escenario alternativo, que asume la implementación de intervenciones capaces de modificar la LST. La herramienta permite la simulación de intervenciones dentro de los límites de la demarcación territorial, sin embargo como parte del proyecto 'Proyecto Piloto de identificación y evaluación de acciones y proyectos estratégicos para atención de las Islas de Calor Urbanas en la Megalópolis' se espera que las intervenciones se propongan dentro de las ICUS priorizadas por cada demarcación territorial.", styles['IntroJustify']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Los resultados de ambos escenarios se estiman mediante modelos de aprendizaje automático.</b> Específicamente, a través de un modelo de regresión de refuerzo de gradiente basado en histograma (HistGBR por sus siglas en inglés), la herramienta simula las acciones definidas por el usuario y genera predicciones precisas sobre su impacto potencial en la LST así como algunos beneficios ambientales y socioeconómicos. El informe se divide en dos secciones: la primera presenta los resultados globales del municipio {municipality_name}, mientras que la segunda detalla el impacto en cada polígono de intervención establecido por la persona usuaria.", styles['IntroJustify']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>El propósito final de este reporte y de la herramienta es facilitar la toma de decisiones informadas, identificar las acciones más prometedoras y optimizar las estrategias de intervención para mitigar las causas y efectos de las ICU en la Megalópolis</b>, particularmente en el municipio de {municipality_name}, permitiendo así una gestión urbana más resiliente y sostenible a nivel municipal.", styles['IntroJustify']))
        story.append(PageBreak())


        # --- SECTION 2: GLOBAL GEOGRAPHIC ANALYSIS OF LST SCENARIOS ---
        story.append(Paragraph("2. Análisis Geográfico Global de Escenarios de LST", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"A continuación se presenta la cartografía de la Temperatura Superficial Terrestre (LST) estimada, junto con su clasificación térmica para el <b>escenario Base y escenario Intervención</b> (alternativo). La modelación de estas capas se realiza a partir de múltiples variables predictoras <sup>1</sup> <sup>2</sup> <sup>3</sup>, las cuales se detallan posteriormente. Metodológicamente, este proceso conllevó la ejecución de flujos de trabajo en Google Earth Engine (GEE) para la adquisición, procesamiento y cálculo de los siguientes insumos geoespaciales.", styles['IntroJustify']))
        story.append(Paragraph(f"<b>1) Temperatura de la Superficie Terrestre (LST)</b>: Representación digital de la temperatura de la superficie terrestre en grados celsius. Esta capa se calcula como la mediana de las imágenes de Landsat 8 dentro del período de análisis<sup>4</sup>. Específicamente, después de filtrar las imágenes por límites espaciales, fechas y remover nubosidad, se obtiene un compuesto mediano de todas las imágenes conseguidas en el período de análisis.", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>2) Albedo</b>: Reflectividad superficial, calculada de manera similar a la LST. Esta capa de Albedo se calcula como la mediana de los valores de Albedo para cada píxel dentro de tu período de análisis<sup>5</sup>. ", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>3) Índice de Vegetación de Diferencia Normalizada (NDVI):</b> Evalúa la cobertura y calidad de la cobertura vegetal. Para cada imagen individual de Landsat 8, se calculó el NDVI utilizando la diferencia normalizada de las bandas infrarrojo cercano (SR_B5) y rojo (SR_B4)<sup>6</sup>. Después de calcular el NDVI para cada imagen en el período de análisis, se tomó la mediana de estos valores para cada píxel. Esto significa que el ráster final de NDVI que se exporta representa el valor mediano de NDVI observado en el período especificado.", styles['Bullet']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
        "<sup>1</sup> Toscan, P. C., Seong, K., Jiao, J., Ribeiro, C. A. L. R., Carvalho, F. A. C., Oliveira, M. L. S., & Pereira, E. B. (2025). Impact of nature-based solutions (NBS) on urban surface temperatures and land cover changes using remote sensing and machine learning. Remote Sensing Applications: Society and Environment, 39, 101721. https://doi.org/10.1016/j.rsase.2025.1016/j.rsase.2025.101721",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>2</sup> Velasco, E. (2025, agosto 21). Calentamiento urbano e islas de calor: Causas y efectos. Seminario Virtual Sobre Islas De Calor Urbanas En La Megalópolis, Molina Center for Energy and the Environment (MCE2).",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>3</sup> Bai, Y., Wang, M., Yan, Y., & Wang, H. (2025). Exploring the impact of 2D/3D urban morphology on land surface temperature within the diurnal cycle in Tianjin. Scientific Reports, 15(1), 39740. https://doi.org/10.1038/s41598-025-17849-7",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>4</sup> El período de análisis considera imágenes del <b>1 de enero de 2025</b> al <b>30 de septiembre de 2025</b>. Debido al alto porcentaje de nubosidad en algunas imágenes satelitales, fue necesario excluir meses específicos del análisis. Las excepciones fueron Jiutepec (17011) y Cuautlancingo (21041), cuyos conjuntos de datos permitieron un análisis ininterrumpido. Para los municipios de Gustavo A. Madero (09005), Tláhuac (09011) y Zacatelco (29044) se excluyó el mes de julio. En Iztapalapa (09007) y Querétaro (22014) se excluyeron los meses de junio, julio y septiembre. En el municipio de Nezahualcóyotl (15058) se excluyó el mes de septiembre. En el municipio de Toluca (15106) se excluyeron los meses de enero y julio. En el municipio de Pachuca (13048) se excluyeron los meses de abril y mayo.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>5</sup> Datos base de Landsat 8, Nivel 2, Colección 2, Nivel 1 utilizada en Google Earth Engine es USGS. (2020). Landsat 8 Collection 2 Tier 1 Surface Reflectance. Google Earth Engine. La capa de albedo se calcula con el método de Tasumi, M., Allen, R. G., & Trezza, R. (2008). At-Surface Reflectance and Albedo from Satellite for Operational Calculation of Land Surface Energy Balance. Journal of Hydrologic Engineering, 13(2), 51–63. https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51).",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>6</sup> NDVI = (SR_B5 - SR_B4) / (SR_B5 + SR_B4) de acuerdo con Kaya, Z., & Dervisoglu, A. (2023). Determination of Urban Areas Using Google Earth Engine and Spectral Indices; Esenyurt Case Study. International Journal of Environment and Geoinformatics, 10(1), 1–8. https://doi.org/10.30897/ijegeo.1214001.",
        styles['Footnote']))
        story.append(PageBreak())


        story.append(Paragraph(f"<b>4) Índice de Agua de Diferencia Normalizada Modificado (MNDWI):</b> Evalúa las masas de agua. Sigue un proceso similar al NDVI y Albedo. Para cada imagen individual de Landsat 8, se calcula el MNDWI (Normalized Difference Modified Water Index) utilizando la diferencia normalizada de las bandas verde (SR_B3) e infrarrojo medio (SR_B6)<sup>7</sup>. Luego de calcular el MNDWI para cada imagen en el período de análisis, se toma la mediana de estos valores para cada píxel.", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>5) Índice de Suelo Descubierto (BSI):</b> Identifica las áreas de suelo descubierto. Para cada imagen individual de Landsat 8, se calcula el BSI utilizando una combinación de bandas SWIR1 (SR_B6), rojo (SR_B4), NIR (SR_B5) y azul (SR_B2)<sup>8</sup>. Después de calcular el BSI para cada imagen en el período de análisis, se toma la mediana de estos valores para cada píxel. ", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>6) Altura de construcciones:</b> Dimensión vertical de los edificios, estructuras e infraestructura que sobresalen del terreno. Esta capa proviene del conjunto de datos estáticos de GHSL<sup>9</sup>, por lo que representa un valor o estado fijo en un momento dado. Por lo tanto, no son un promedio ni una mediana calculada sobre el período de análisis. Se utilizan como capas de entrada individuales que se recortan de acuerdo al área de estudio sin una aggregación temporal.", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>7) Luces de noche:</b> Captura la luz emitida desde la superficie terrestre durante la noche. Similar a las variables de Landsat 8, la colección de imágenes de luces nocturnas (VIIRS<sup>10</sup>) se filtra por el período de análisis y luego se agrega calculando la mediana dentro del período de tiempo especificado.", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>8) Modelo Digital de Elevación:</b> Representación digital del relieve de la superficie terrestre. Esta capa proviene del conjunto de datos estáticos de SRTM<sup>11</sup>, por lo que representa un valor o estado fijo en un momento dado. Por lo tanto, no son un promedio ni una mediana calculada sobre el período de análisis. Se utilizan como capas de entrada individuales que se recortan de acuerdo al área de estudio sin una aggregación temporal.", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>9) Superficie construida:</b> Distribución de la superficie construida dentro del área de estudio. Esta capa proviene del conjunto de datos estáticos de GHSL<sup>12</sup>, por lo que representa un valor o estado fijo en un momento dado. Por lo tanto, no son un promedio ni una mediana calculada sobre el período de análisis. Se utilizan como capas de entrada individuales que se recortan de acuerdo al área de estudio sin una aggregación temporal.", styles['Bullet']))
        story.append(Spacer(1, 20))
        
        # --- FOOTER ---
        story.append(Paragraph(
        "<sup>7</sup> MNDWI = (SR_B3 - SR_B6) / (SR_B3 + SR_B6) de acuerdo con Kaya, Z., & Dervisoglu, A. (2023). Determination of Urban Areas Using Google Earth Engine and Spectral Indices; Esenyurt Case Study. International Journal of Environment and Geoinformatics, 10(1), 1–8. https://doi.org/10.30897/ijegeo.1214001.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>8</sup>  BSI = ((SWIR1 + SR_B4) - (NIR + SR_B2)) / ((SWIR1 + SR_B4) + (NIR + SR_B2)) de acuerdo con Kaya, Z., & Dervisoglu, A. (2023). Determination of Urban Areas Using Google Earth Engine and Spectral Indices; Esenyurt Case Study. International Journal of Environment and Geoinformatics, 10(1), 1–8. https://doi.org/10.30897/ijegeo.1214001.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>9</sup>  GHS-BUILT-S R2023A tomado de Pesaresi M., Politis P. (2023): GHS-BUILT-S R2023A - GHS built-up surface grid, derived from Sentinel2 composite and Landsat, multitemporal (1975-2030)European Commission, Joint Research Centre (JRC). PID: Joint Research Centre Data Catalogue - GHS-BUILT-S R2023A - GHS built-up surface grid, European Commission doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>10</sup>  NOAA National Geophysical Data Center. (2018). VIIRS Day/Night Band (DNB) Monthly Composites, Version 1. Google Earth Engine 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>11</sup> Farr, T. G., et al. (2007), The Shuttle Radar Topography Mission (SRTM), Rev. Geophys., 45, RG2004, doi:10.1029/2005RG000183.",
        styles['Footnote']))
        story.append(Paragraph(
        "<sup>12</sup> GHS-BUILT-S R2023A tomado de Pesaresi M., Politis P. (2023): GHS-BUILT-S R2023A - GHS built-up surface grid, derived from Sentinel2 composite and Landsat, multitemporal (1975-2030)European Commission, Joint Research Centre (JRC). PID: Joint Research Centre Data Catalogue - GHS-BUILT-S R2023A - GHS built-up surface grid, European Commission doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA.",
        styles['Footnote']))
        story.append(PageBreak())


        story.append(Paragraph(f"La delimitación de isotermas para el escenario Base y escenario Intervención (alternativo) consideró los siguientes intervalos térmicos de referencia, establecidos por la metodología de Centro EURE, aplicados sobre la capa de LST:", styles['IntroJustify']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>1) Zona núcleo:</b> Temperatura \u2265 +3 \u00b0C respecto al promedio urbano de la demarcación territorial.", styles['Bullet']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>2) Zona de impacto:</b> Temperatura de +2\u00b0C a +3\u00b0C respecto al promedio urbano de la demarcación territorial.", styles['Bullet']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>3) Zona de transición:</b> Temperatura de +1\u00b0C a +2\u00b0C respecto al promedio urbano de la demarcación territorial.", styles['Bullet']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>4) Fondo térmico urbano:</b> Temperatura \u2264 +1\u00b0C respecto al promedio urbano de la demarcación territorial.", styles['Bullet']))
        story.append(Spacer(1, 4))
        story.append(Paragraph("2.1 Intervenciones modeladas", styles['Heading2Custom']))
        story.append(Paragraph(f"<b>La herramienta permite a los usuarios asignar intervenciones específicas dentro de los polígonos de intervención, con el fin de modelar su efecto sobre la capa de LST dentro de las ICU identificadas</b>. Las intervenciones del cátalogo son predefinidas por la CAMe y están diseñadas para alterar parámetros del balance energético urbano, facilitando la evaluación tanto de estrategias de mitigación como de escenarios de impacto adverso. Entre las principales estrategias de mitigación, orientadas a reducir la absorción de calor y promover el enfriamiento evaporativo, se encuentran las siguientes:", styles['IntroJustify']))
        story.append(Paragraph(f"-Implementación de techos fríos", styles['Bullet']))
        story.append(Paragraph(f"-Uso de pavimentos fríos (alta reflectividad)", styles['Bullet']))
        story.append(Paragraph(f"-Arborización y creación de infraestructura verde", styles['Bullet']))
        story.append(Paragraph(f"-Recuperación o desarrollo de cuerpos de agua", styles['Bullet']))
        story.append(Paragraph(f"-Incremento de la cobertura vegetal", styles['Bullet']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>La modelación de estas intervenciones se realiza para cada tipo de cobertura de suelo dentro del polígono establecido</b>. El usuario es responsable de seleccionar la intervención propuesta para cada categoría de cobertura de suelo: superficie vial, superficie construida, cuerpos de agua (estanques poco profundos o profundos), áreas verdes (vegetación escasa, moderada o densa) y suelo descubierto. En consecuencia, las medidas relacionadas con pavimentos reflejantes o fríos se aplican exclusivamente en las áreas identificadas como superficies viales.", styles['IntroJustify']))
        story.append(Paragraph(f"<b>En este ejercicio, las intervenciones modeladas en la simulación fueron las siguientes:</b>", styles['IntroJustify']))
    

        # Attempt to decode intervention codes using CSV lookup tables in data/csv
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            csv_dir = os.path.join(project_root, 'data', 'csv')

            csv_map_files = {
                'Street': 'intervenciones_vialidades.csv',
                'Builtup': 'intervenciones_construido.csv',
                'Shallow_Water': 'intervenciones_agua.csv',
                'Deep_Water': 'intervenciones_agua.csv',
                'Sparse_Green': 'intervenciones_areas_verdes.csv',
                'Moderate_Green': 'intervenciones_areas_verdes.csv',
                'Dense_Green': 'intervenciones_areas_verdes.csv',
                'Bareland': 'intervenciones_suelo_descubierto.csv'
            }

            # Load CSV lookup tables into memory (1-based index -> intervention name)
            lookup_tables = {}
            for key, fname in csv_map_files.items():
                path = os.path.join(csv_dir, fname)
                if os.path.exists(path):
                    try:
                        df_lookup = pd.read_csv(path)
                        if 'intervencion' in df_lookup.columns:
                            lookup_tables[fname] = df_lookup['intervencion'].astype(str).tolist()
                        else:
                            lookup_tables[fname] = []
                    except Exception:
                        lookup_tables[fname] = []
                else:
                    lookup_tables[fname] = []

            # Generate a per-polygon list of interventions with human-readable surface labels
            prop_to_surface_label = {
                'Street': 'Superficie vial',
                'Builtup': 'Superficie construida',
                'Shallow_Water': 'Superficie agua',
                'Deep_Water': 'Superficie agua',
                'Sparse_Green': 'Superficie verde',
                'Moderate_Green': 'Superficie verde',
                'Dense_Green': 'Superficie verde',
                'Bareland': 'Superficie suelo descubierto'
            }

            any_found = False
            if hasattr(self, 'gdf_interventions') and not self.gdf_interventions.empty:
                for idx, feat in self.gdf_interventions.iterrows():
                    try:
                        prop_dict = feat.to_dict()
                    except Exception:
                        prop_dict = {}

                    # Determine polygon id
                    polygon_id = prop_dict.get('id', None)
                    polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id) if polygon_id is not None else f"{idx}"

                    # For each relevant property, if value>0 then map to intervention name and append
                    for prop_name, csv_fname in csv_map_files.items():
                        if prop_name in prop_dict:
                            try:
                                val = prop_dict.get(prop_name)
                                if val is None or (isinstance(val, float) and np.isnan(val)):
                                    continue
                                val_int = int(val)
                            except Exception:
                                continue
                            if val_int > 0:
                                lookup_list = lookup_tables.get(csv_fname, [])
                                # The value from the GeoJSON is a 1-based index. Convert to 0-based for list access.
                                csv_index = val_int - 1

                                if lookup_list and 0 <= csv_index < len(lookup_list):
                                    interv_name = lookup_list[csv_index]
                                else:
                                    # Fallback if index is out of bounds or lookup list is empty
                                    interv_name = f"Intervención {val_int} ({prop_name})"

                                surface_label = prop_to_surface_label.get(prop_name, prop_name)
                                story.append(Paragraph(f"- Polígono ID {polygon_id_str}: Se realizó la intervención \"{interv_name}\" en \"{surface_label}\".", styles['Bullet']))
                                any_found = True

            if not any_found:
                story.append(Paragraph("<i>No se detectaron intervenciones en el archivo proporcionado.</i>", styles['Normal']))
        except Exception as e:
            logger.warning(f"No se pudo cargar o decodificar las intervenciones: {e}")
            story.append(Paragraph("<i>No fue posible listar las intervenciones modeladas (error de decodificación).</i>", styles['Normal']))

        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>La estimación del impacto térmico se realizó mediante modelos de aprendizaje automático (machine learning) calibrados específicamente para cada demarcación.</b> Una vez definidos los polígonos y las estrategias a implementar, el algoritmo procesa estas variables de entrada y simula la modificación de las propiedades físicas de las superficies (albedo, NDVI, MNDWI, etc.). Este enfoque analítico permite proyectar la variación de la LST contrastando dos escenarios: el escenario base o gemelo (sin intervención) y el escenario proyectado (con intervención). Aunque se evaluaron diversos algoritmos de aprendizaje automático<sup>13</sup>, para la estimación de la LST se seleccionó un modelo de regresión de gradiente basado en histogramas (Histogram-based Gradient Boosting Regressor - Hist Gradient Boosting). Este algoritmo proyecta los cambios en la temperatura superficial a partir de variables clave (como albedo, NDVI, MNDWI, entre otras), habiendo demostrado consistentemente una mayor precisión y robustez analítica frente a las demás alternativas evaluadas.", styles['IntroJustify']))
        
        story.append(Spacer(1, 350))
        story.append(Paragraph(
        "<sup>13</sup> La herramienta evaluó la implementación de los siguientes modelos de aprendizaje automático: Árboles Aleatorios (Random Forest), Refuerzo de Gradientes Extremo (eXtreme Gradient Boosting - XGBoost) y  Refuerzo de Gradientes basado en histogramas (HistogramGradient BoostingRegressor  HGRBoost). Para mayor información sobre la evaluación de modelos, ver 5. Anexos.",
        styles['Footnote']))
        story.append(PageBreak())

        # Add LST Twin Map
        img_twin_lst = get_image_for_reportlab(self.paths["map_base"], max_pdf_img_width)
        if img_twin_lst:
            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: Temperatura de la superficie terrestre - Escenario Base ({municipality_name})</b>", styles['Normal']),
                Spacer(1, 6),
                img_twin_lst
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1
        
        # NOTE: La figura de diferencia se añadirá tras el mapa de intervención

        # Add LST Intervention Map
        img_intervention_lst = get_image_for_reportlab(self.paths["map_intervention"], max_pdf_img_width)
        if img_intervention_lst:
            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: Temperatura de la superficie terrestre - Escenario con intervención ({municipality_name})</b>", styles['Normal']),
                Spacer(1, 6),
                img_intervention_lst
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1

        story.append(PageBreak()) # Force new page

        # Add LST Classification Twin Map
        lst_class_base_map_path = os.path.join(self.report_inputs_dir, "lst_classification_twin_map.png")
        img_class_twin_lst = get_image_for_reportlab(lst_class_base_map_path, max_pdf_img_width)
        if img_class_twin_lst:
            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: Clasificación de LST - Escenario Base ({municipality_name})</b>", styles['Normal']),
                Spacer(1, 6),
                img_class_twin_lst
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1

        story.append(PageBreak()) # Force new page

        # Add LST Classification Intervention Map
        lst_class_intervention_map_path = os.path.join(self.report_inputs_dir, "lst_classification_intervention_map.png")
        img_class_intervention_lst = get_image_for_reportlab(lst_class_intervention_map_path, max_pdf_img_width)
        if img_class_intervention_lst:
            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: Clasificación de LST - Escenario con intervención ({municipality_name})</b>", styles['Normal']),
                Spacer(1, 6),
                img_class_intervention_lst
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1

        # Add Difference LST Map (Figura 5) - Moved here per request
        heatmap_impact_path = self.paths.get("map_impact")
        img_heatmap_impact = get_image_for_reportlab(heatmap_impact_path, max_pdf_img_width)
        if img_heatmap_impact:
            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: Diferencia LST (Intervención - Base) ({municipality_name})</b>", styles['Normal']),
                Spacer(1, 6),
                img_heatmap_impact
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1

        story.append(PageBreak())

        # --- SECTION 3: GLOBAL METRICS AND CHARTS ---
        story.append(Paragraph("3. Métricas Globales", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"En esta sección se realiza un desglose detallado de indicadores clave de rendimiento generados a partir del análisis global en todo el municipio de {municipality_name}. A continuación, se detallan las variables específicas agrupadas por dimensión:", styles['IntroJustify']))
        story.append(Paragraph("<b>1. Población y vivienda </b>", styles['IntroJustify']))
        story.append(Paragraph(f"  -Primera Infancia: Población de 0 a 5 años.", styles['Bullet']))
        story.append(Paragraph(f"  -Adultos Mayores: Población de 65 años y más.", styles['Bullet']))
        story.append(Paragraph(f"  -Discapacidad: Población con alguna limitación física o mental.", styles['Bullet']))
        story.append(Paragraph(f"  -Rezago Habitacional: Viviendas vulnerables (carencia de agua, electricidad o en situación de hacinamiento).", styles['Bullet']))
        story.append(Paragraph(f"  -Acceso a la Salud: Población sin afiliación a servicios médicos.", styles['Bullet']))
        story.append(Paragraph(f"  -Perspectiva de Género: Hogares con jefatura femenina. Mujeres en edad productiva (15 a 60 años).", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>2. Consumo de energía eléctrica </b>", styles['IntroJustify']))
        story.append(Paragraph(f"  -Estimación del consumo total residencial de electricidad (kWh).", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>3. Calidad del aire</b>", styles['IntroJustify']))
        story.append(Paragraph(f"  -Concentración potencial de Ozono troposférico (O\u00b3 ppb).", styles['Bullet']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>En la sección de Población y Vivienda, se presentan gráficas comparativas que evalúan el impacto de las estrategias de mitigación</b>. Al contrastar el Escenario Base con el Escenario de Intervención, se evidencia el desplazamiento de habitantes y viviendas entre áreas de alto estrés térmico ('Zona de Impacto') y zonas de confort ('Fondo Térmico Urbano'). La estimación del impacto térmico sobre la población se realiza mediante un análisis de estadística zonal. Este geoproceso superpone las capas de información demográfica sobre el mapa de clasificación térmica (LST), permitiendo calcular y extraer métricas agregadas (suma, promedio, etc.) específicamente para los habitantes ubicados dentro de cada categoría térmica. Bajo esta premisa, la prioridad es garantizar que el grueso de la viviendas y población resida en áreas de confort térmico. Este cambio representa una disminución en la vulnerabilidad climática y un incremento en el bienestar general de la comunidad.", styles['IntroJustify']))
        story.append(Paragraph("<b>En la sección de Consumo de Energía Eléctrica, se presenta una comparativa gráfica que cuantifica el impacto de las estrategias de mitigación sobre la demanda residencial de electricidad</b>. Este análisis permite dimensionar el ahorro energético derivado de la reducción de la temperatura superficial a partir de las intervenciones propuestas. El objetivo central de esta métrica es demostrar una disminución neta en el consumo residencial total de electricidad bajo el Escenario Intervención, validando así los beneficios económicos y ambientales de las intervenciones propuestas.", styles['IntroJustify']))
        story.append(Paragraph("<b>En la sección de Calidad del Aire, se modela el impacto de las estrategias de mitigación sobre la formación potencial de ozono troposférico</b>. Dado que la temperatura actúa como catalizador en las reacciones fotoquímicas, una disminución en la concentración de este contaminante valida la efectividad ambiental de las intervenciones. En este contexto, un resultado óptimo se define por una reducción cuantificable en las partes por billón (ppb) estimadas, lo que contribuye directamente a la salud pública.", styles['IntroJustify']))

        story.append(Paragraph("<b>3.1 Población y vivienda </b>", styles['Heading2Custom']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Esta sección presenta los resultados estadísticos obtenidos para el municipio de {municipality_name} a partir de la clasificación y el análisis detallado de isotermas. Este análisis integra datos socio-demográficos y de las características de la vivienda, proporcionados por el Instituto Nacional de Estadística y Geografía<sup>14</sup> (INEGI). El propósito central es proporcionar una visión general de los patrones de temperatura (isotermas) observados con el contexto de la población, facilitando una comprensión del impacto de las Islas de Calor Urbanas (ICUs) y sus efectos en los habitantes.", styles['IntroJustify']))
        story.append(Paragraph("<sup>14</sup>  Instituto Nacional de Estadística y Geografía (INEGI), Censo de Población y Vivienda 2020. Principales resultados por manzana urbana.",styles['Footnote']))
        story.append(Spacer(1, 30))

         # --- Other Global Zonal Stats Charts (Social Variables) ---
        for band_name_display in self.social_variable_chart_order_display_names:
            safe_source_band_name = self._sanitize_band_name(band_name_display)
            chart_filename = f"zonal_stats_percentage_Global_{safe_source_band_name}.png"
            chart_path_zonal = os.path.join(self.report_inputs_dir, chart_filename)

            img_chart = get_image_for_reportlab(chart_path_zonal, max_pdf_img_width)
            
            if img_chart:
                title_text = self.chart_titles.get(chart_filename, f"{band_name_display} por Zona LST (Global)")

                story.append(KeepTogether([
                    Paragraph(f"<b>Figura {current_global_fig_num}: {title_text}</b>", styles['Normal']),
                    Spacer(1, 6),
                    img_chart
                ]))
                story.append(Spacer(1, 12))
                current_global_fig_num += 1
            else:
                # Si la gráfica no existe, se inserta la leyenda de error
                story.append(Paragraph(f"No se encontró información para variable {band_name_display}", styles['Normal']))
                story.append(Spacer(1, 12))

        # 1. Obtener la lista de resúmenes de energía del DTO
        energy_summaries = getattr(self.result, 'energy_summaries', [])

        # 2. Extraer el texto del resumen global
        global_energy_summary_text = None
        if isinstance(energy_summaries, list):
            global_summary = next((s for s in energy_summaries if s.get('polygon_identifier') == 'Global'), None)
            if global_summary:
                global_energy_summary_text = global_summary.get('summary_text')

        # --- Global Energy Consumption Summary (Text) ---

        if global_energy_summary_text:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>3.2 Consumo de energía eléctrica </b>", styles['Heading2Custom']))
            story.append(Paragraph(f"El análisis de consumo residencial de electricidad Global para {municipality_name} estima el impacto potencial de las intervenciones implementadas en el consumo de energía. El cálculo toma en cuenta el consumo de electricidad anual per cápita a nivel municipal<sup>15</sup>, el umbral de temperatura confortable para el municipio de {municipality_name}, así como la tasa de consumo de energía por grado de aumento estimada para el estado de {state_name}<sup>16</sup>.", styles['IntroJustify']))
            story.append(Paragraph("<sup>15</sup> Información tomada de la Plataforma Nacional de Energía Ambiente y Sociedad (PLANEAS) del Consejo Nacional de Humanidades, Ciencies y Tecnologías (CONAHCyT) Ecosistema Nacional Informático de Energía y Cambio Climático con base en Comisión Federal Electricidad (CFE) (2018). Usuarios y consumo y de electricidad por municipio (A partir de 2018). Datos abiertos. Consultado en 26 marzo 2020. Disponible en: https://datos.gob.mx/busca/dataset/usuarios-y-consumo-de-electricidad-por-municipio-a-partir-de-2018.",styles['Footnote']))
            story.append(Paragraph("<sup>16</sup> Botzen, W.J.W.; Nees, T.; Estrada, F. Temperature Effects on Electricity and Gas Consumption: Empirical Evidence from Mexico and Projections under Future Climate Conditions. Sustainability 2021, 13, 305. https://doi.org/10.3390/su13010305",styles['Footnote']))
            story.append(Spacer(1, 10))
            for line in global_energy_summary_text.split('\n'):
                processed_line = line.strip()
                if not processed_line:
                    continue

                is_bolded_ansi = processed_line.startswith('\x1b[1m') and processed_line.endswith('\x1b[0m')
                if is_bolded_ansi:
                    processed_line = processed_line.replace('\x1b[1m', '').replace('\x1b[0m', '').strip()

                if processed_line.startswith('---') and processed_line.endswith('---'):
                    processed_line = processed_line.strip(' -')

                if is_bolded_ansi:
                    processed_line = f"<b>{processed_line}</b>"

                story.append(Paragraph(processed_line, styles['CodeStyle']))
            story.append(Spacer(1,12))
        else:
            story.append(Paragraph("<i>No se encontró el resumen global de consumo de energía.</i>", styles['Normal']))

        story.append(Spacer(1, 12))
        total_energy_consumption_bar_chart_path = os.path.join(self.report_inputs_dir, 'total_energy_consumption_bar_chart_global.png')
        img_chart = get_image_for_reportlab(total_energy_consumption_bar_chart_path, max_pdf_img_width)
        if img_chart:
            title_text = self.chart_titles.get(os.path.basename(total_energy_consumption_bar_chart_path), 'Consumo Residencial Total de Electricidad (Global)')

            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: {title_text}</b>", styles['Normal']),
                Spacer(1, 6),
                img_chart
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1
        else:
            story.append(Paragraph(f"No se encontró información para variable Consumo Residencial Total de Electricidad", styles['Normal']))
        
        story.append(PageBreak())
        story.append(Paragraph("<b>3.3 Calidad del Aire </b>", styles['Heading2Custom']))
        story.append(Paragraph(f"En esta sección se presenta el porcentaje de cambio en la concentración de ozono troposférico al comparar los resultados del escenario base y el escenario de intervención. La calidad del aire, específicamente la formación de ozono troposférico, está fuertemente condicionada por las variaciones de la Temperatura Superficial Terrestre (LST)<sup>17</sup>. La gráfica adjunta ilustra la mejora porcentual en la potencial concentración de ozono troposférico en el municipio de {municipality_name}, como resultado de la disminución o incremento de la LST. El modelo ocupado para este indicador cuantifica la sensibilidad del ozono a la temperatura y no puede se considerado para proporcionar una predicción exacta y determinista completa. Estos hallazgos validan la correlación positiva entre la mitigación térmica y la reducción de precursores de ozono troposférico, confirmando la efectividad de las estrategias implementadas.", styles['IntroJustify']))
        story.append(Spacer(1, 12))
        # --- Global Ozone Donut Chart ---
        ozone_donut_chart_path = os.path.join(self.report_inputs_dir, 'ozone_improvement_donut_chart_global.png')
        img_chart = get_image_for_reportlab(ozone_donut_chart_path, max_pdf_img_width * 0.8) # Donut charts usually fit better if slightly smaller
        if img_chart:
            title_text = self.chart_titles.get(os.path.basename(ozone_donut_chart_path), 'Porcentaje de cambio en la concentración de ozono (O3) Global')

            story.append(KeepTogether([
                Paragraph(f"<b>Figura {current_global_fig_num}: {title_text}</b>", styles['Normal']),
                Spacer(1, 6),
                img_chart
            ]))
            story.append(Spacer(1, 12))
            current_global_fig_num += 1
        else:
            story.append(Paragraph(f"No se encontró información para variable Porcentaje de cambio en la concentración de ozono (O3)", styles['Normal']))
            story.append(Spacer(1, 12))
        story.append(Paragraph("<sup>17</sup> Castro, T., Peralta, O., Sánchez-Vargas, A., & Salcido, A. (2025). Evolution of Tropospheric Ozone and Surface Temperature in Mexico City from 2000 to 2021. Atmosphere, 16(12), 1379. https://doi.org/10.3390/atmos16121379.",styles['Footnote']))


        story.append(PageBreak())

        # --- SECTION 4: PER-POLYGON DEEP DIVE ANALYSIS ---
        story.append(Paragraph("4. Análisis Detallado por Polígono de Intervención", styles['Heading1']))
        story.append(Paragraph("Esta sección ofrece un estudio a detalle para cada polígono de intervención, replicando el marco metodológico utilizado en el análisis global. Los resultados se muestran en dos secciones. La primera sección presenta la cartografía de Temperatura Superficial Terrestre (LST) estimada, junto con su clasificación térmica para el escenario Base y escenario Intervención (alternativo) siguiendo la misma lógica y metodología que en la sección global (2. Análisis Geográfico Global de Escenarios de LST).", styles['IntroJustify']))
        story.append(Spacer(1, 6))
        story.append(Paragraph("De igual forma, en la segunda parte se realiza un desglose detallado de indicadores clave de rendimiento para las dimensiones de <i>Población y Vivienda</i>, <i>Consumo de Energía</i> y <i>Calidad del Aire</i> estimados para cada polígono de intervención definido.", styles['IntroJustify']))

        if not self.gdf_interventions.empty:
            # Define the order of chart types for per-polygon display
            per_polygon_chart_types_ordered = [
                {'chart_type': 'lst_twin_polygon', 'title_prefix': "LST Escenario Base"},
                {'chart_type': 'lst_intervention_polygon', 'title_prefix': "LST Escenario Intervención"},
                {'chart_type': 'lst_classification_twin_polygon', 'title_prefix': "Clasificación LST Escenario Base"},
                {'chart_type': 'lst_classification_intervention_polygon', 'title_prefix': "Clasificación LST Escenario Intervención"},
                {'chart_type': 'heatmap_impact', 'title_prefix': "Diferencia LST (Intervención - Base)"},
            ]

            # Iterate through each polygon
            for index, feature in self.gdf_interventions.iterrows():
                polygon_id = feature['id'] if 'id' in feature and not pd.isna(feature['id']) else f"Unnamed_Polygon_{index}"
                polygon_id_str = str(int(polygon_id)) if isinstance(polygon_id, (int, float)) else str(polygon_id)

                story.append(Paragraph(f"4.{index+1} Polígono ID: {polygon_id_str}", styles['Heading2Custom']))
                story.append(Spacer(1, 6))

                # Filter all generated charts for the current polygon_id
                charts_for_current_polygon = [c for c in self.generated_per_polygon_chart_details if c['polygon_id'] == polygon_id_str]

                # --- Add Per-Polygon Maps (LST and LST Classification) ---
                # Collect per-polygon map flowables and place them in a 2-column table (reduced size)
                per_polygon_cells = []
                # Calculate width dynamically for 2 columns (two maps per row)
                try:
                    usable_width = doc.width - 12  # small horizontal padding
                    per_polygon_img_width = (usable_width / 2) - 6  # account for cell padding
                except Exception:
                    per_polygon_img_width = (6 * inch) / 2

                # Calculate max height per image so two rows fit on a page
                try:
                    per_polygon_img_max_height = (doc.height / 2) - 24  # subtract small vertical padding
                except Exception:
                    per_polygon_img_max_height = 4 * inch

                # Reduce map dimensions by 25% (scale factor 0.75)
                SCALE_REDUCTION = 1
                per_polygon_img_width = per_polygon_img_width * SCALE_REDUCTION
                per_polygon_img_max_height = per_polygon_img_max_height * SCALE_REDUCTION

                for chart_spec in per_polygon_chart_types_ordered:
                    chart_type_to_find = chart_spec['chart_type']
                    found_chart = next((c for c in charts_for_current_polygon if c['chart_type'] == chart_type_to_find), None)

                    if found_chart and found_chart['path']:
                        # Special case: render per-polygon difference heatmap as full-page figure
                        if chart_type_to_find == 'heatmap_impact':
                            full_img_max_width = doc.width - (1 * inch)
                            full_img_max_height = doc.height - (1 * inch)
                            img_full = get_image_for_reportlab(found_chart['path'], full_img_max_width, full_img_max_height)
                            caption = Paragraph(f"<b>Figura {current_global_fig_num}: {found_chart['title']} - Polígono ID {polygon_id_str}</b>", styles['Normal'])
                            if img_full:
                                story.append(Spacer(1, 12))
                                story.append(caption)
                                story.append(Spacer(1, 6))
                                story.append(img_full)
                                story.append(PageBreak())
                                current_global_fig_num += 1
                                # Skip adding this chart to the 2x2 grid
                                continue
                            else:
                                # Fallback to note missing image
                                story.append(Paragraph(f"No se encontró información para variable {found_chart['title']}", styles['Normal']))
                                story.append(Spacer(1, 12))
                                current_global_fig_num += 1
                                continue

                        # Default behavior: Load image constrained by width and max height to avoid huge cells
                        img_chart = get_image_for_reportlab(found_chart['path'], per_polygon_img_width, per_polygon_img_max_height)
                        caption = Paragraph(f"<b>Figura {current_global_fig_num}: {found_chart['title']} - Polígono ID {polygon_id_str}</b>", styles['Normal'])
                        if img_chart:
                            # Use an inner table to keep caption and image together but allow ReportLab to paginate rows
                            inner = [[caption], [img_chart]]
                        else:
                            inner = [[caption], [Paragraph(f"No se encontró información para variable {found_chart['title']}", styles['Normal'])]]

                        inner_table = Table(inner, colWidths=[per_polygon_img_width])
                        inner_table.setStyle(TableStyle([
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('LEFTPADDING', (0,0), (-1,-1), 0),
                            ('RIGHTPADDING', (0,0), (-1,-1), 0),
                            ('TOPPADDING', (0,0), (-1,-1), 0),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                        ]))
                        per_polygon_cells.append(inner_table)
                        current_global_fig_num += 1
                    else:
                        display_title_base = chart_spec.get('title_prefix', chart_type_to_find)
                        per_polygon_cells.append(Paragraph(f"No se encontró información para variable {display_title_base}", styles['Normal']))

                # Arrange cells into 2x2 grid per page (2 columns x 2 rows)
                if per_polygon_cells:
                    col_widths = [ (doc.width / 2) - 6, (doc.width / 2) - 6 ]
                    # iterate groups of 4 cells
                    for i in range(0, len(per_polygon_cells), 4):
                        c0 = per_polygon_cells[i]
                        c1 = per_polygon_cells[i+1] if i+1 < len(per_polygon_cells) else ''
                        c2 = per_polygon_cells[i+2] if i+2 < len(per_polygon_cells) else ''
                        c3 = per_polygon_cells[i+3] if i+3 < len(per_polygon_cells) else ''

                        maps_table = Table([[c0, c1], [c2, c3]], colWidths=col_widths)
                        maps_table.setStyle(TableStyle([
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('LEFTPADDING', (0,0), (-1,-1), 6),
                            ('RIGHTPADDING', (0,0), (-1,-1), 6),
                            ('TOPPADDING', (0,0), (-1,-1), 6),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                        ]))
                        story.append(maps_table)
                        story.append(Spacer(1, 12))
                        # Force a page break so each 2x2 block occupies one page
                        story.append(PageBreak())

                # --- Add Per-Polygon Zonal Stats Charts (Social Variables) in ORDER ---
                for band_name_display in self.social_variable_chart_order_display_names:
                    # Find the specific zonal stats chart for this polygon and source band
                    found_chart = next((c for c in charts_for_current_polygon if c['chart_type'] == 'zonal_stats' and c['source_band'] == band_name_display), None)
                    if found_chart and found_chart['path']:
                        img_chart = get_image_for_reportlab(found_chart['path'], max_pdf_img_width)
                        if img_chart:
                            # Get the global title and remove " (Global)" for per-polygon context
                            global_title = self.chart_titles.get(f'zonal_stats_percentage_Global_{self._sanitize_band_name(band_name_display)}.png', band_name_display)
                            per_polygon_title = global_title.replace(' (Global)', '')

                            story.append(KeepTogether([
                                Paragraph(f"<b>Figura {current_global_fig_num}: {per_polygon_title} - Polígono ID {polygon_id_str}</b>", styles['Normal']),
                                Spacer(1, 6),
                                img_chart
                            ]))
                            story.append(Spacer(1, 12))
                            current_global_fig_num += 1
                        else:
                            story.append(Paragraph(f"No se encontró información para variable {band_name_display}", styles['Normal']))
                            story.append(Spacer(1, 12))
                    else:
                        story.append(Paragraph(f"No se encontró información para variable Porcentaje de {band_name_display} por Zona LST", styles['Normal']))
                        story.append(Spacer(1, 12))

                # Add Per-Polygon Energy Consumption Summary (text) from energy_summaries
                polygon_energy_summary_text = None
                polygon_summary = next((s for s in getattr(self.result, 'energy_summaries', []) if s.get('polygon_identifier') == f'Polígono ID {polygon_id_str}'), None)
                if polygon_summary:
                    polygon_energy_summary_text = polygon_summary.get('summary_text')

                # 2. Renderizado en el PDF (Lógica de Story)
                if polygon_energy_summary_text:
                    story.append(Paragraph(f"<b>Resumen de Resultados del Análisis de Energía para el Polígono {polygon_id_str}</b>", styles['Normal']))
                    story.append(Spacer(1, 12))
                    for line in polygon_energy_summary_text.split('\n'):
                        processed_line = line.strip()
                        if not processed_line:
                            continue
                        # This logic is to handle the text formatting from the engine's summary
                        is_bolded_ansi = processed_line.startswith('\x1b[1m') and processed_line.endswith('\x1b[0m')
                        if is_bolded_ansi:
                            processed_line = processed_line.replace('\x1b[1m', '').replace('\x1b[0m', '').strip()
                        if processed_line.startswith('---') and processed_line.endswith('---'):
                            processed_line = processed_line.strip(' -')
                        if is_bolded_ansi:
                            processed_line = f"<b>{processed_line}</b>"
                        story.append(Paragraph(processed_line, styles['CodeStyle']))
                    story.append(Spacer(1, 12))
                else:
                    # Mensaje de fallback si el polígono no tiene datos asociados
                    story.append(Paragraph(f"<i>No se encontraron datos de impacto energético específicos para el Polígono ID {polygon_id_str}.</i>", styles['Normal']))
                    story.append(Spacer(1, 12))

                # Add Per-Polygon Total Energy Consumption Bar Chart
                energy_bar_chart_filename = f'total_energy_consumption_bar_chart_Polígono_ID_{polygon_id_str}.png'
                chart_path_energy = os.path.join(self.report_inputs_dir, energy_bar_chart_filename)
                img_chart = get_image_for_reportlab(chart_path_energy, max_pdf_img_width)
                if img_chart:
                    story.append(KeepTogether([
                        Paragraph(f"<b>Figura {current_global_fig_num}: Consumo Residencial Total de Electricidad - Polígono ID {polygon_id_str}</b>", styles['Normal']),
                        Spacer(1, 6),
                        img_chart
                    ]))
                    story.append(Spacer(1, 12))
                    current_global_fig_num += 1
                else:
                    story.append(Paragraph(f"No se encontró información para variable Consumo Residencial Total de Electricidad", styles['Normal']))
                    story.append(Spacer(1, 12))

                # Add Per-Polygon Ozone Donut Chart
                ozone_donut_chart_filename = f'ozone_improvement_donut_chart_polygon_{polygon_id_str}.png'
                chart_path_ozone = os.path.join(self.report_inputs_dir, ozone_donut_chart_filename)
                img_chart = get_image_for_reportlab(chart_path_ozone, max_pdf_img_width * 0.8)
                if img_chart:
                    story.append(KeepTogether([
                        Paragraph(f"<b>Figura {current_global_fig_num}: Porcentaje de cambio en la concentración de ozono (O3) - Polígono ID {polygon_id_str}</b>", styles['Normal']),
                        Spacer(1, 6),
                        img_chart
                    ]))
                    story.append(Spacer(1, 12))
                    current_global_fig_num += 1
                else:
                    story.append(Paragraph(f"No se encontró información para variable Porcentaje de cambio en la concentración de ozono (O3)", styles['Normal']))
                    story.append(Spacer(1, 12))

                story.append(PageBreak()) # New page for next polygon or next section

            logger.info("   All per-polygon charts added to story.")

        else:
            story.append(Paragraph("<i>No se encontraron polígonos de intervención en el GeoPackage proporcionado.</i>", styles['Normal']))
            story.append(Spacer(1, 12))

        story.append(Paragraph("5. Anexo", styles['Heading1']))
        story.append(Spacer(1, 12))

        # Insert models evaluation table (if available) BEFORE Figura 1
        try:
            # Look for evaluation JSON in multiple likely locations: report_inputs, models/, workspace models/
            candidates = [
                os.path.join(self.report_inputs_dir, 'all_models_evaluation_results.json'),
                os.path.join(os.getcwd(), 'models', 'all_models_evaluation_results.json'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'all_models_evaluation_results.json')
            ]
            eval_json_path = None
            for c in candidates:
                if os.path.exists(c):
                    eval_json_path = c
                    break

            if eval_json_path:
                with open(eval_json_path, 'r', encoding='utf-8') as jf:
                    eval_data = json.load(jf)

                story.append(Paragraph('Evaluación de Modelos', styles['Heading2Custom']))
                story.append(Spacer(1, 6))

                # Normalize into table rows
                table_data = []
                if isinstance(eval_data, dict):
                    # If contains list of models under a key, try to detect
                    list_candidate = None
                    for k, v in eval_data.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            list_candidate = v
                            break

                    if list_candidate is not None:
                        # Convert to DataFrame for sorting and formatting
                        df = pd.DataFrame(list_candidate)
                        all_cols = df.columns.tolist()

                        # --- Sorting ---
                        # Find actual column names for sorting using aliases
                        sort_by_cols = []
                        municipio_aliases = ['municipality_id', 'municipalityId', 'municipio', 'municipality', 'Municipio']
                        modelo_aliases = ['Model_name', 'model_name', 'ModelName', 'modelName', 'Modelo']

                        # Find Municipio column
                        for alias in municipio_aliases:
                            if alias in all_cols:
                                sort_by_cols.append(alias)
                                break
                        # Find Modelo column
                        for alias in modelo_aliases:
                            if alias in all_cols:
                                sort_by_cols.append(alias)
                                break

                        if sort_by_cols:
                            df.sort_values(by=sort_by_cols, inplace=True)

                        # --- Formatting and Table construction ---
                        # Preserve original column order from the source file
                        ordered_cols = list(list_candidate[0].keys())
                        table_data.append(ordered_cols)

                        # Identify columns to round using aliases
                        cols_to_round = set()
                        rounding_aliases = {
                            'R2': ['R2_Score', 'R2', 'r2', 'R_squared', 'r_squared'],
                            'RMSE': ['RMSE', 'rmse', 'Rmse'],
                            'MAE': ['MAE', 'mae']
                        }
                        for key, aliases in rounding_aliases.items():
                            for alias in aliases:
                                if alias in all_cols:
                                    cols_to_round.add(alias)
                                    break

                        # Create rows from the sorted and formatted DataFrame
                        for _, item_series in df.iterrows():
                            row = []
                            for col_name in ordered_cols:
                                val = item_series.get(col_name, '')
                                if col_name in cols_to_round:
                                    try:
                                        val = f"{float(val):.3f}"
                                    except (ValueError, TypeError):
                                        val = str(val)
                                else:
                                    val = str(val)
                                row.append(val)
                            table_data.append(row)
                    else:
                        # render simple key-value pairs
                        table_data.append(['Métrica', 'Valor'])
                        for k, v in eval_data.items():
                            table_data.append([str(k), str(v)])

                elif isinstance(eval_data, list):
                    if eval_data and isinstance(eval_data[0], dict):
                        # Convert to DataFrame for sorting and formatting
                        df = pd.DataFrame(eval_data)
                        all_cols = df.columns.tolist()

                        # --- Sorting ---
                        # Find actual column names for sorting using aliases
                        sort_by_cols = []
                        municipio_aliases = ['municipality_id', 'municipalityId', 'municipio', 'municipality', 'Municipio']
                        modelo_aliases = ['Model_name', 'model_name', 'ModelName', 'modelName', 'Modelo']

                        # Find Municipio column
                        for alias in municipio_aliases:
                            if alias in all_cols:
                                sort_by_cols.append(alias)
                                break
                        # Find Modelo column
                        for alias in modelo_aliases:
                            if alias in all_cols:
                                sort_by_cols.append(alias)
                                break

                        if sort_by_cols:
                            df.sort_values(by=sort_by_cols, inplace=True)

                        # --- Formatting and Table construction ---
                        # Preserve original column order from the source file
                        ordered_cols = list(eval_data[0].keys())
                        table_data.append(ordered_cols)

                        # Identify columns to round using aliases
                        cols_to_round = set()
                        rounding_aliases = {
                            'R2': ['R2_Score', 'R2', 'r2', 'R_squared', 'r_squared'],
                            'RMSE': ['RMSE', 'rmse', 'Rmse'],
                            'MAE': ['MAE', 'mae']
                        }
                        for key, aliases in rounding_aliases.items():
                            for alias in aliases:
                                if alias in all_cols:
                                    cols_to_round.add(alias)
                                    break

                        # Create rows from the sorted and formatted DataFrame
                        for _, item_series in df.iterrows():
                            row = []
                            for col_name in ordered_cols:
                                val = item_series.get(col_name, '')
                                if col_name in cols_to_round:
                                    try:
                                        val = f"{float(val):.3f}"
                                    except (ValueError, TypeError):
                                        val = str(val)
                                else:
                                    val = str(val)
                                row.append(val)
                            table_data.append(row)
                    else:
                        table_data = [[str(x)] for x in eval_data]

                else:
                    table_data = [[str(eval_data)]]

                # Create ReportLab table
                if table_data:
                    # Calculate column widths to fit page
                    available_width = doc.width
                    num_cols = len(table_data[0]) if table_data else 0
                    col_widths = None

                    if num_cols > 0:
                        col_max_lens = [0] * num_cols
                        for row in table_data:
                            for i, cell in enumerate(row):
                                if i < num_cols:
                                    col_max_lens[i] = max(col_max_lens[i], len(str(cell)))
                        total_len = sum(col_max_lens)
                        if total_len > 0:
                            col_widths = [(l / total_len) * available_width for l in col_max_lens]

                    tbl = Table(table_data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('LEFTPADDING', (0,0), (-1,-1), 4),
                        ('RIGHTPADDING', (0,0), (-1,-1), 4),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 12))
        except Exception as e:
            logger.warning(f"No se pudo cargar la tabla de evaluación de modelos: {e}")

        story.append(Paragraph("<b>R<sup>2</sup>_Score (Coeficiente de Determinación)</b>: Representa la proporción de la varianza en la LST que es predecible a partir de las variables independientes (como NDVI, Albedo, etc.). En términos más simples, muestra qué tan bien el modelo explica la variabilidad de la LST. Un valor más cercano a 1 indica que el modelo explica una gran parte de la varianza de la LST y, por lo tanto, es un buen ajuste a los datos. ",styles['Footnote']))
        story.append(Paragraph("<b>RMSE (Root Mean Squared Error - Raíz del Error Cuadrático Medio)</b>: Mide la magnitud promedio de los errores del modelo. La raíz cuadrada se aplica para que la unidad del error sea la misma que la unidad de la variable predicha (grados Celsius). Un valor de RMSE más bajo indica un mejor rendimiento del modelo.",styles['Footnote']))
        story.append(Paragraph("<b>MAE (Mean Absolute Error - Error Absoluto Medio)</b>: Mide la magnitud promedio de los errores de un conjunto de predicciones. Calcula el promedio de las diferencias absolutas entre las predicciones y los valores reales. Un valor de MAE más bajo indica un mejor rendimiento del modelo.",styles['Footnote']))
        


        try:
            doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
            logger.info(f"   ✅ PDF Report generated at {self.paths['report_pdf']}")
        except Exception as e:
            logger.error(f"   ❌ PDF Generation failed: {e}")
