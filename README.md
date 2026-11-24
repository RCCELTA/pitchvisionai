# PitchVision AI

**Proyecto PitchVision AI – Real Club Celta de Vigo, S.A.D.**

Este código forma parte del proyecto PitchVision AI (expediente TSI-100130-2024-074), financiado por el Ministerio para la Transformación Digital y de la Función Pública y por la Unión Europea – NextGenerationEU, en el marco del Plan de Recuperación, Transformación y Resiliencia.

Sistema de análisis de datos deportivos para evaluación de rendimiento físico y predicción de riesgo de lesiones en futbolistas, utilizando técnicas de machine learning y análisis de datos con PySpark.
 
## Descripción del Proyecto
 
PitchVision AI es una plataforma de análisis avanzado que procesa datos de rendimiento deportivo de futbolistas, incluyendo:
- Datos antropométricos
- Pruebas físicas (CMJ, Nordic, Isoprone, SLJ)
- Datos GPS de entrenamientos y partidos
- Perfiles competitivos
- Historial de lesiones
 
El proyecto utiliza técnicas de clustering, análisis exploratorio de datos (EDA) y modelos predictivos para identificar patrones de rendimiento y evaluar riesgos de lesión.
 
## Estructura del Proyecto
 
### `exploratory_data_analisys/`
Contiene scripts de análisis exploratorio de datos para diferentes fuentes de información:
 
- **`aggregated_data.py`**: Análisis de datos agregados de múltiples fuentes (antropometría, pruebas físicas, GPS)
- **`injury_risk_data.py`**: Análisis de datos relacionados con riesgo de lesiones
- **`v_anthropometry_analisys.py`**: Análisis de datos antropométricos (altura, peso, IMC, etc.)
- **`v_competitive_profile_analisys.py`**: Análisis de perfiles competitivos de jugadores
- **`v_gps_data_analisys.py`**: Análisis de datos GPS (distancia, velocidad, aceleraciones)
- **`v_injury_analisys.py`**: Análisis de historial de lesiones
- **`v_injury_analisys_for_predictive_model.ipynb`**: Notebook de análisis de lesiones para modelo predictivo
- **`v_physical_test_cmj_analisys.py`**: Análisis de pruebas Counter Movement Jump
- **`v_physical_test_isoprone_analisys.py`**: Análisis de pruebas Isoprone
- **`v_physical_test_nordic_analisys.py`**: Análisis de pruebas Nordic Hamstring
- **`v_physical_test_slj_analisys.py`**: Análisis de pruebas Standing Long Jump
 
### `clustering/`
Implementación de algoritmos de clustering (K-Means con PCA) para segmentación de jugadores:
 
#### `aggregated_data/`
- **`clustering_aggregated_data.py`**: Clustering sobre datos agregados completos
- **`clustering_aggregated_fisicas_anthropometry.py`**: Clustering combinando pruebas físicas y antropometría
- **`clustering_aggregated_nordic_isoprone_gps.py`**: Clustering específico para Nordic, Isoprone y GPS
- **`clustering_aggregated_todo.py`**: Clustering con todas las variables disponibles
 
#### `v_anthropometry/`
- **`clustering_anthropometry.py`**: Clustering basado en datos antropométricos
- **`clustering_anthropometry_13_15.py`**: Clustering para jugadores de 13-15 años
- **`clustering_excluyendo_menores.py`**: Clustering excluyendo jugadores menores
 
#### Clustering por tipo de prueba física:
- **`v_gps_data/clustering_v_gps_data.py`**: Clustering datos GPS
- **`v_physical_test_cmj/clustering_physical_test_cmj.py`**: Clustering CMJ
- **`v_physical_test_isoprone/clustering_isoprone.py`**: Clustering Isoprone
- **`v_physical_test_nordic/clustering_nordic.py`**: Clustering Nordic
- **`v_physical_test_slj/clustering_slj.py`**: Clustering SLJ
 
#### Visualización:
- **`visualizacionResultados.py`**: Visualización y análisis de resultados de clustering, incluyendo matrices de transición entre clusters
 
### `predictive_model/`
Modelos predictivos para evaluación de riesgo de lesiones:
 
- **`predictive_model.py`**: Implementación del modelo predictivo usando Logistic Regression
- **`predictive_model_crossval.ipynb`**: Notebook con validación cruzada del modelo predictivo
 
## Tecnologías Utilizadas
 
- **PySpark**: Procesamiento distribuido de datos
- **Databricks**: Plataforma de ejecución
- **Python**: Lenguaje principal
- **Bibliotecas de ML**:
  - `pyspark.ml` (KMeans, PCA, LogisticRegression)
  - `pyspark.ml.feature` (VectorAssembler, StandardScaler)
- **Visualización**:
  - `matplotlib`
  - `seaborn`
  - `pandas`
- **NumPy**: Operaciones numéricas
 
## Fuentes de Datos
 
El proyecto trabaja con las siguientes tablas de Databricks:
 
### Silver Layer (datos limpios):
- `dev_silver.pitchvision.v_anthropometry`
- `dev_silver.pitchvision.v_competitive_profile`
- `dev_silver.pitchvision.v_gps_data`
- `dev_silver.pitchvision.v_physical_test_cmj`
- `dev_silver.pitchvision.v_physical_test_isoprone`
- `dev_silver.pitchvision.v_physical_test_nordic`
- `dev_silver.pitchvision.v_physical_test_slj`
 
### Gold Layer (datos agregados y procesados):
- `dev_gold.pitchvision.agregado_physical_tests`
- `dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering`
- `dev_gold.pitchvision.injury_risk_input_filtered`
- `dev_gold.pitchvision.clustersresults_*`
 
## Metodología
 
### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones
- Detección de outliers
- Análisis de correlaciones
- Estadísticas descriptivas por temporada y edad
 
### 2. Clustering
- Preprocesamiento: escalado y normalización
- Reducción de dimensionalidad con PCA
- Algoritmo K-Means para segmentación
- Evaluación con Silhouette Score
- Visualización de clusters
 
### 3. Modelado Predictivo
- Feature engineering
- Entrenamiento de modelos de clasificación
- Validación cruzada
- Evaluación de métricas (precisión, recall, F1-score)
 
## Columnas Clave
 
- **`id_unico_deportista`**: Identificador único del jugador
- **`test_date`**: Fecha de la prueba/medición
- **`season`**: Temporada deportiva
- **`age_years`**: Edad del jugador en años
 
## Uso
 
Los scripts están diseñados para ejecutarse en entorno Databricks con acceso a las tablas mencionadas. Cada script puede ejecutarse de forma independiente según el análisis requerido.
 
### Ejemplo de flujo de trabajo:
1. Ejecutar scripts de EDA para entender los datos
2. Aplicar clustering para segmentar jugadores
3. Visualizar resultados de clustering
4. Entrenar modelo predictivo de lesiones
5. Evaluar y validar el modelo
 
## Resultados
 
El proyecto genera:
- Perfiles de jugadores basados en clustering
- Matrices de transición entre clusters
- Predicciones de riesgo de lesión
- Visualizaciones y reportes analíticos
- Tablas procesadas en Gold Layer
