import sys
import os
import time
import pandas as pd
import geopandas as gpd
import logging
#from IPython.display import display, Image
import numpy as np # Import numpy
import rasterio # Import rasterio for rasterize

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add the 'proyecto' directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'proyecto'))

# Importar Clases Refactorizadas
from app.core.loaders import GlobalDataLoader
from app.core.engine import SimulationEngine
from app.core.processor import ResultProcessor

def run_comparison():
    
    logger.info("🧪 TEST")
    

    # 1. CONFIGURACIÓN (Ajusta esto según tu prueba original)
    MUNICIPIO_CLAVE = "13048" # IZTAPALAPA, CDMX
    SAMPLING_FACTOR = 6       # Usa 6 para igualar al prototipo (o 1 para máxima precisión)

    # Rutas de datos (asumiendo que ya ejecutaste el setup de carpetas)
    DATA_DIR = "./data" # Changed to project data path
    MODEL_DIR = "./models"
    GPKG_PATH = "./data/interventions/intervenciones.geojson" # Changed to project data path
    LOCAL_LOGO_DIR = "./data/logos" # Explicitly define the local logo directory

    # ------------------------------------------------------
    # FASE 1: INFRAESTRUCTURA (GlobalDataLoader)
    # ------------------------------------------------------
    logger.info("[1/3] Cargando Datos (GlobalDataLoader)...")
    start_time = time.time()

    loader = GlobalDataLoader(data_dir=DATA_DIR, model_path=os.path.join(MODEL_DIR, "heat_island_v1.joblib"))

    # Obtener contexto
    # Usamos bbox ficticio para el testing minimalista. En un caso real, esto vendría del frontend o de un config
    # OJO: La bbox debe ser compatible con el CRS de los rasters. EPSG:4326 (lat/lon) es común.
    context = loader.get_project_context(bbox=[-99.68, 19.10, -99.45, 19.39], municipio_clave=MUNICIPIO_CLAVE)
    context['ml_model'] = loader.get_model() # Inyección explicita
    context['municipio_clave'] = MUNICIPIO_CLAVE # Add municipio_clave to context for engine to pass to result

    logger.info(f"✅ Datos cargados en {time.time() - start_time:.2f}s")

    # ------------------------------------------------------
    # FASE 2: MOTOR DE SIMULACIÓN (SimulationEngine)
    # ------------------------------------------------------
    logger.info(f"[2/3] Ejecutando Simulación (Engine) | Sampling: {SAMPLING_FACTOR}x...")

    # Inicializar Motor
    engine = SimulationEngine(
        context=context,
        simulation_sampling_factor=SAMPLING_FACTOR
    )

    # Cargar Geometría de Intervención
    if os.path.exists(GPKG_PATH):
        logger.info(f"   📂 Usando archivo de intervenciones: {GPKG_PATH}")
        gdf_interventions = gpd.read_file(GPKG_PATH)
    else:
        logger.info("   ⚠️ No se encontró 'intervenciones.geojson'.")
        logger.info("   CREANDO POLÍGONO DUMMY CENTRADO para la prueba.")
        from shapely.geometry import box
        profile = context['profile']
        h, w = profile['height'], profile['width']
        t = profile['transform']
        cx, cy = t * (w/2, h/2)
        # Crear un cuadrado de ~500m
        dummy_poly = box(cx, cy, cx + 0.005, cy + 0.005)
        gdf_interventions = gpd.GeoDataFrame(
            {'geometry': [dummy_poly], 'id': [1], 'Builtup': [1], 'energy_efficiency': [0]},
            crs=profile['crs']
        )
        # Inyectar regla dummy si no hay CSVs
        if not engine.all_intervention_scenarios:
             engine.all_intervention_scenarios = {
                 'Builtup': pd.DataFrame([{'Albedo': 0.9, 'NDVI': 0.5}])
             }

    # EJECUTAR CÁLCULO
    start_time = time.time()
    result = engine.run_simulation(gdf_interventions)
    logger.info(f"✅ Simulación completada en {time.time() - start_time:.2f}s")

    # ------------------------------------------------------
    # FASE 3: RESULTADOS (ResultProcessor)
    # ------------------------------------------------------
    logger.info("[3/3] Procesando Salidas (ResultProcessor)...")
    output_dir = "./results"
    # Ensure the output directory is created
    os.makedirs(output_dir, exist_ok=True)
    # Update filename to avoid 'dummy' confusion
    energy_summary_path = os.path.join(output_dir, "report_inputs", "energy_consumption_summary.json")
    zonal_stats_path = os.path.join(output_dir, "report_inputs", "zonal_statistics.csv")

    processor = ResultProcessor(
        result=result,
        output_base_dir=output_dir,
        project_name= f"TestProject_{MUNICIPIO_CLAVE}",
        municipio_clave=MUNICIPIO_CLAVE,
        gdf_interventions=gdf_interventions,
        local_population_raster_path=loader.raster_paths.get('Population'),
        energy_summary_path=energy_summary_path,
        zonal_stats_path=zonal_stats_path,
        local_logo_dir=LOCAL_LOGO_DIR # Pass the correct local logo directory
    )

    # Modificado para usar los nuevos métodos públicos
    web_data = processor.get_web_payload()
    pdf_path = processor.generate_pdf_report()
    geospatial_zip_path = processor.export_geospatial_pkg()

    # Update paths dictionary to reflect new return values
    paths = {
        'metrics': os.path.join(output_dir, "report_inputs", "metrics.json"),
        'report_pdf': pdf_path,
        'geospatial_zip': geospatial_zip_path
    }

    # ------------------------------------------------------
    # REPORTE DE COMPARACIÓN
    # ------------------------------------------------------
    logger.info("   📊 RESULTADOS PARA VALIDACIÓN ")

    kpis = result.kpis
    
    logger.info(f"📂 Archivos generados en: {output_dir}")
    logger.info(f"   - PDF: {paths.get('report_pdf', 'No generado')}")
    logger.info(f"   - JSON de Métricas: {paths.get('metrics', 'No generado')}")
    logger.info(f"   - ZIP Geospatial: {paths.get('geospatial_zip', 'No generado')}")
    logger.info("=========================================================")

    # Verify the existence of the generated PDF
    if os.path.exists(paths.get('report_pdf', '')):
        size = os.path.getsize(paths['report_pdf']) / 1024 / 1024 # in MB
        logger.info(f"✅ Verification: PDF report generated successfully ({size:.2f} MB).")
    else:
        logger.info(f"   ❌ Verification: PDF report NOT found at {paths.get('report_pdf')}.")

if __name__ == "__main__":
    run_comparison()
