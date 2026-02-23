# **Simulador de Mitigación de Isla de Calor Urbana (ICU)**

## **Introducción**

Esta herramienta es un motor de simulación científica diseñado para ayudar a los gobiernos locales a enfrentar el cambio climático en entornos urbanos. Su objetivo principal es permitir a las autoridades municipales **identificar y modelar el impacto de diversas intervenciones urbanas** —como la instalación de techos frescos, la creación de áreas verdes o el uso de pavimentos reflectantes— antes de su implementación real.

A través de modelos de aprendizaje automático (Machine Learning), la herramienta predice cómo cambiaría la Temperatura Superficial Terrestre (TST) en zonas específicas y cuantifica beneficios adicionales muy importantes para la toma de decisiones:

1. **Ahorro energético:** Estima cuánto disminuiría el consumo de electricidad residencial al reducir la temperatura.  
2. **Calidad del aire:** Modela la reducción potencial en la concentración de ozono troposférico.  
3. **Impacto social:** Cruza los datos térmicos con información demográfica para identificar cómo se protege a poblaciones vulnerables (niños, adultos mayores, etc.).

## **Características Principales**

* **Predicción Basada en ML:** Utiliza un modelo HistGradientBoostingRegressor para estimar cambios en la temperatura superficial basados en variables biofísicas (albedo, NDVI, MNDWI, etc.).  
* **Escenarios de Intervención:** Permite definir polígonos geográficos y aplicarles "tratamientos" específicos (vegetación densa, cuerpos de agua, techos fríos, entre otros).  
* **Análisis de Impacto Energético:** Cálculo del ahorro en MWh/año basado en umbrales de confort térmico y sensibilidad de consumo por municipio.  
* **Modelado de Ozono:** Estimación de la formación de ![][image1] basada en la relación térmica superficie-atmósfera.  
* **Generación de Reportes Automáticos:** Exporta automáticamente mapas en PNG, archivos geoespaciales en GeoTIFF y un **informe técnico completo en formato PDF** con gráficas comparativas.

## **Requisitos del Sistema**

El sistema requiere Python 3.9+ y las dependencias listadas en requirements\_core.txt.

### **Dependencias Clave:**

* **NumPy & Pandas:** Procesamiento de datos matriciales y tabulares.  
* **Rasterio & GeoPandas:** Manejo de imágenes satelitales (TIF) y vectores (GeoJSON/GPKG).  
* **Scikit-Learn:** Motor de inferencia para el modelo de calor.  
* **ReportLab:** Generación del documento PDF.  
* **Contextily:** Integración de mapas base (Esri/OpenStreetMap) en los resultados visuales.

## **Estructura del Repositorio**

icu-main/  
├── app/  
│   └── core/  
│       ├── dto.py          \# Objetos de transferencia de datos (Resultados)  
│       ├── engine.py       \# Motor lógico de la simulación y cálculos físicos  
│       ├── loaders.py      \# Cargador global de rasters, modelos y CSVs  
│       ├── processor.py    \# Generador de activos visuales y reporte PDF  
│       └── exceptions.py   \# Gestión de errores específicos del core  
├── data/  
│   ├── city\_rasters/       \# Capas base (LST, NDVI, Albedo, Población, etc.)  
│   ├── csv/                \# Tablas de consumo eléctrico y reglas de intervención  
│   ├── interventions/      \# GeoJSON con los polígonos definidos por el usuario  
│   └── logos/              \# Logos institucionales para el reporte  
├── models/                 \# Modelos de Machine Learning entrenados (.joblib)  
├── results/                \# Salidas generadas (Mapas, GeoTIFFs, Informe PDF)  
└── validation\_run.py       \# Script principal para ejecutar una simulación de prueba

## **Guía de Uso Rápido**

### **1\. Preparación de Datos**

Asegúrese de contar con los archivos ráster necesarios en data/city\_rasters/ y el modelo entrenado en models/heat\_island\_v1.joblib.

### **2\. Definir Intervenciones**

Cree un archivo intervenciones.geojson en la carpeta correspondiente. Cada polígono debe tener atributos que indiquen el tipo de intervención (ej. Builtup: 1 para techos fríos).

### **3\. Ejecutar la Simulación**

Puede utilizar el script de validación incluido para procesar un proyecto:

python validation\_run.py

### **4\. Revisar Resultados**

Tras la ejecución, la carpeta results/ contendrá:

* **informe\_TestProject.pdf:** El reporte técnico detallado.  
* **simulations/:** Archivos GeoTIFF con la diferencia térmica para uso en SIG (QGIS/ArcGIS).  
* **report\_inputs/:** Gráficas individuales y métricas en formato JSON.

## **Metodología Científica**

La herramienta utiliza un enfoque de **Gemelo Digital Térmico**. Compara un "Escenario Base" (condiciones actuales capturadas por sensores Landsat 8\) contra un "Escenario de Intervención" donde las propiedades físicas de la superficie (albedo, vegetación) son modificadas matemáticamente.

El modelo de aprendizaje automático ha sido entrenado para entender la compleja relación entre la morfología urbana (altura de edificios, luces nocturnas, elevación) y la retención de calor, lo que permite una precisión superior a la de los modelos lineales tradicionales.

## **Licencia**

Este proyecto está bajo la licencia **MIT**, Comisión Ambiental de la Megalópolis (CAMe, 2026).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAXCAYAAADgKtSgAAABbUlEQVR4Xu2UvS8EURTFN1ugkhBRTGbmzVckpqLSSCQUOpvtRSfR6JQqf4GEQij0/AOiotVQi2ylUm1CIcFy7ro7npN57C7l/pKXee+ec++8d+ejUhnQK3EcmzAMD4wx+1mWjbLeF1EU7aLgOwqvyTpJkhDrB4xn9vZCVYtesiBAe8VocbwrpDBGg+MdoC3qzZdY+xEk3Usix4n2yTBOWXCCHi9o0gVrNtjxmPqarDmB+UWScJMR1mxQfFWL37DmRBN+a4mc8FY3scFaKWmaTnZbnH15ng+Zz1f0EWPP9hZo0hvHbaDfqW/aisl6S+fS2uuvDEVNzp1Dm1XPEWsdVK9zXB7Uuooxa+hvJBouJ6wJyN2E3oC+zFoBxHMpEgTBioaqSDzTwnPfzCXA14JvhuMFvu+Pw7QN0xWuO/inTLHHBTZyLBvheF94njehp2p/G5gf/ltxAcWerLm0pWbrfwbtmJdW4n8/zNqAUj4A1plnNJd11hAAAAAASUVORK5CYII=>