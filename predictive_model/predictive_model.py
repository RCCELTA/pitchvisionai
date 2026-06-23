# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
import sys
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
import pandas as pd
from pyspark.sql.window import Window
import seaborn as sns
from pyspark.sql.functions import col, udf
from pyspark.ml.classification import LogisticRegression
import numpy as np
import matplotlib.patches as mpatches

# COMMAND ----------


TABLE_NAME = "dev_gold.pitchvision.injury_risk_input_filtered"

df = spark.table(TABLE_NAME)

print(f"EDA DE TABLA: {TABLE_NAME}")

num_rows = df.count()
num_cols = len(df.columns)

print("\n1. DIMENSIONES")
print(f"Filas: {num_rows:,}")
print(f"Columnas: {num_cols:,}")

print("\n2. ESQUEMA")
df.printSchema()

print("\n3. MUESTRA DE DATOS")
display(df.limit(10))


numeric_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
]

string_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, StringType)
]

date_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, (DateType, TimestampType))
]

other_cols = [
    field.name for field in df.schema.fields
    if field.name not in numeric_cols + string_cols + date_cols
]

print("\n4. TIPOS DE COLUMNAS")
print(f"Columnas numéricas ({len(numeric_cols)}): {numeric_cols}")
print(f"Columnas texto/categóricas ({len(string_cols)}): {string_cols}")
print(f"Columnas fecha/timestamp ({len(date_cols)}): {date_cols}")
print(f"Otras columnas ({len(other_cols)}): {other_cols}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas acumuladas

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import sum as spark_sum

# Definir ventanas para la última semana y el último mes por deportista
window_7d = Window.partitionBy("id_unico_deportista").orderBy("activity_date").rangeBetween(-7*86400, 0)
window_30d = Window.partitionBy("id_unico_deportista").orderBy("activity_date").rangeBetween(-30*86400, 0)

# Convertir activity_date a timestamp si es necesario
df = df.withColumn("activity_date_ts", F.col("activity_date").cast("timestamp"))

# Crear columnas de acumulado
df = df.withColumn(
    "participation_time_minutes_7d_sum",
    F.sum("participation_time_minutes").over(
        Window.partitionBy("id_unico_deportista")
        .orderBy(F.col("activity_date_ts").cast("long"))
        .rangeBetween(-7*86400, 0)
    )
).withColumn(
    "participation_time_minutes_30d_sum",
    F.sum("participation_time_minutes").over(
        Window.partitionBy("id_unico_deportista")
        .orderBy(F.col("activity_date_ts").cast("long"))
        .rangeBetween(-30*86400, 0)
    )
).withColumn(
    "total_distance_7d_sum",
    F.sum("total_distance").over(
        Window.partitionBy("id_unico_deportista")
        .orderBy(F.col("activity_date_ts").cast("long"))
        .rangeBetween(-7*86400, 0)
    )
).withColumn(
    "total_distance_30d_sum",
    F.sum("total_distance").over(
        Window.partitionBy("id_unico_deportista")
        .orderBy(F.col("activity_date_ts").cast("long"))
        .rangeBetween(-30*86400, 0)
    )
)

display(df.select(
    "id_unico_deportista", "activity_date", 
    "participation_time_minutes", "total_distance",
    "participation_time_minutes_7d_sum", "participation_time_minutes_30d_sum",
    "total_distance_7d_sum", "total_distance_30d_sum"
).limit(10))

# COMMAND ----------

cols_to_max = [
    "total_distance",
    "max_velocity",
    "max_acceleration",
    "max_deceleration",
    "acc_dcc_distance",
    "acc_dcc_count",
    "acc_count",
    "dcc_count",
    "HSR",
    "sprint_distance",
    "sprint_count"
]

# Calcular solo la desviación respecto al máximo por jugador para cada columna (sin añadir el máximo)
for col_name in cols_to_max:
    max_col_name = f"{col_name}_player_max_tmp"
    dev_col_name = f"{col_name}_dev_player_max"
    window = Window.partitionBy("id_unico_deportista")
    df = df.withColumn(
        max_col_name, F.max(col_name).over(window)
    ).withColumn(
        dev_col_name,
        F.when(
            F.col(max_col_name).isNotNull() & (F.col(max_col_name) != 0),
            (F.col(col_name) - F.col(max_col_name)) / F.col(max_col_name)
        ).otherwise(None)
    ).drop(max_col_name)

display(df.select("id_unico_deportista", *cols_to_max, *[f"{col}_dev_player_max" for col in cols_to_max]).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Borrado de columnas no informativas

# COMMAND ----------

columnas_a_eliminar = [
    "session_week",
    "iso_left_repetitions",
    "gps_activity_team_id",
    "session_time_minutes",
    "iso_right_repetitions",
    "nor_left_repetitions",
    "comp_activity_team_id"
]

df = df.drop(*columnas_a_eliminar)

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparación de datos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuraciónde variables independientes
# MAGIC Identíficamos qué columns de nuestra tabla limpia van a servir para predecir el riesgo de lesión. Excluimos la variables objetivo (`injury_risk`) y los datos de identidad del jugador y fecha (`id_unico_deportista`, `activity_date`) para que el modelo no se confunda con ellos.
# MAGIC nota: adecuar nombre de la columna variable objetivo y nombre de la tabla

# COMMAND ----------

columna_objetivo = "injury_risk"
columnas_excluidas = [columna_objetivo, "id_unico_deportista", "activity_date"]
columnas_features = [c for c in df.columns if c not in columnas_excluidas]
print(f"Número total de variables que procesará el modelo: {len(columnas_features)}")
print("Variables seleccionadas:", columnas_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminar columnas con moda >95%

# COMMAND ----------

from pyspark.sql import DataFrame

def eliminar_columnas_moda(df: DataFrame, columnas: list, umbral: float = 0.95):
    cols_a_eliminar = []
    total_filas = df.count()
    for col_name in columnas:
        moda = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
        if moda and moda['count'] / total_filas > umbral:
            cols_a_eliminar.append(col_name)
    print(f"Columnas eliminadas por moda > {umbral*100:.1f}%: {cols_a_eliminar}")
    columnas_restantes = [c for c in columnas if c not in cols_a_eliminar]
    return df.drop(*cols_a_eliminar), columnas_restantes

df_limpio_1, columnas_features = eliminar_columnas_moda(df, columnas_features)
display(df_limpio_1.limit(10))

# COMMAND ----------

df_limpio_1 = df_limpio_1.drop("max_deceleration_dev_player_max")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminar columnas con correlación más del 95%

# COMMAND ----------

# DBTITLE 1,Cell 15
def eliminar_columnas_covarianza(df: DataFrame, columnas: list, umbral: float = 0.95) -> DataFrame:
    # Solo columnas numéricas
    df_cols = set(df.columns)
    numeric_cols = [c for c in columnas if c in df_cols and isinstance(df.schema[c].dataType, NumericType)]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_cov", handleInvalid="skip")
    df_vec = assembler.transform(df.select(numeric_cols))
    corr_matrix = Correlation.corr(df_vec, "features_cov", "pearson").head()[0].toArray()
    cols_a_eliminar = set()
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix[i, j]) > umbral:
                # No eliminar columnas que contienen '_dev_'
                if not numeric_cols[j].find('_dev_') != -1:
                    cols_a_eliminar.add(numeric_cols[j])
    print(f"Columnas eliminadas por covarianza > {umbral*100:.1f}%: {list(cols_a_eliminar)}")
    return df.drop(*cols_a_eliminar), [c for c in columnas if c not in cols_a_eliminar]

df_limpio_2, columnas_features = eliminar_columnas_covarianza(df_limpio_1, columnas_features, umbral=0.95)
display(df_limpio_2.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cálculo de pesos

# COMMAND ----------

print("Calculando pesos de balanceo para la variable objetivo...")

# Contamos totales de la base de datos actual
total_registros = df_limpio_2.count()
total_positivos = df_limpio_2.filter(F.col(columna_objetivo) == 1).count()
total_negativos = total_registros - total_positivos

# Calculamos los factores de peso
peso_0 = total_registros / (2.0 * total_negativos)
peso_1 = total_registros / (2.0 * total_positivos)

print(f"-> Peso asignado a los días Sanos (0): {peso_0:.4f}")
print(f"-> Peso asignado a los días de Riesgo (1): {peso_1:.4f} (Factor x{peso_1/peso_0:.1f})")

# Añadimos la columna de pesos al DataFrame
df_con_pesos = df_limpio_2.withColumn(
    "weight", 
    F.when(F.col(columna_objetivo) == 1, peso_1).otherwise(peso_0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ensamblado de vectores
# MAGIC PySpark requiere que todas las variables independientes estén agrupadas en una única columna de tipo vector. Utilizamos `VectorAssembler` para consolidar todas las columnas seleccionadas en una nueva columna temporal llamada `features_sin_escalar`.

# COMMAND ----------

# DBTITLE 1,Cell 19
valid_features = [c for c in columnas_features if c in df_con_pesos.columns and df_con_pesos.schema[c].dataType.typeName() != 'timestamp']

assembler = VectorAssembler(
    inputCols=valid_features, 
    outputCol="features_sin_escalar",
    handleInvalid="keep"
)

df_ensamblado = assembler.transform(df_con_pesos)

display(df_ensamblado.select("id_unico_deportista", "activity_date", "features_sin_escalar", "weight",  columna_objetivo).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Escalado de Variables
# MAGIC Como la Regresión Logística utiliza cálculos basados en magnitudes, necesitamos igualar el terreno de juego para que variables con números grandes (ej. Distancia Total en metros) no eclipsen a variables con números pequeños (ej. Ratio ACWR). Ponemos todos los datos en la misma escala (media 0 y desviación estándar 1). Utilizamos `StandardScaler`.

# COMMAND ----------

scaler = StandardScaler(
    inputCol="features_sin_escalar", 
    outputCol="features", 
    withStd=True, 
    withMean=True
)

scaler_model = scaler.fit(df_ensamblado)
df_preparado = scaler_model.transform(df_ensamblado)

df_entrenamiento = df_preparado.select("id_unico_deportista", "activity_date", "features", "weight", columna_objetivo)

display(df_entrenamiento.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuración y entrenamiento del modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuración de la Regresión Logística
# MAGIC Configuramos el algoritmo de Regresión Logística de PySpark. Para proteger el modelo contra el sobreentrenamiento (overfitting) al usar las dos temporadas completas, aplicamos una regularización ligera (`regParam`). Además, especificamos explícitamente el nombre de la columna de entrada de datos (`features`) y la variable objetivo (`{variable objetivo}`).

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol=columna_objetivo,
    predictionCol="prediccion_dura",        # Clasificación estándar 0 o 1 (umbral 50%)
    probabilityCol="probabilidad_vector",   # Vector que contiene la probabilidad de [Sano, Riesgo]
    weightCol="weight",
    regParam=0.01                           # Penalización suave para evitar overfitting
)

print("Algoritmo de Regresión Logística configurado y listo para el entrenamiento.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entrenamiento del Modelo con las Dos Temporadas
# MAGIC Ejecutamos el entrenamiento (.fit) utilizando la totalidad de los datos disponibles. Al finalizar, el modelo habrá aprendido los patrones de fatiga y riesgo, asignando un peso matemático a cada variable del club.

# COMMAND ----------

modelo_entrenado = lr.fit(df_entrenamiento, {lr.labelCol: columna_objetivo})

print("¡Entrenamiento completado con éxito sobre las dos temporadas!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generación de Probabilidades de Riesgo
# MAGIC Pasamos los datos por el modelo entrenado para generar las predicciones. Por defecto, PySpark entrega la probabilidad en una columna de tipo vector llamada `probabilidad_vector` que contiene dos números: [Probabilidad de ser 0, Probabilidad de ser 1]. Como nuestro semáforo se guiará por el riesgo de lesión (ser 1), creamos una celda para extraer exclusivamente ese segundo valor como un número decimal continuo (de 0.0 a 1.0).
# MAGIC
# MAGIC nota: poner nombre variable objetivo

# COMMAND ----------

df_predicciones_raw = modelo_entrenado.transform(df_entrenamiento) # Aplicamos el modelo para obtener las columnas de predicción

extraer_prob_riesgo = udf(lambda v: float(v[1]), DoubleType())     # Función para extraer el segundo elemento del vector de probabilidad (riesgo de lesión)

df_predicciones_con_riesgo = df_predicciones_raw.withColumn(       # Creamos la columna definitiva 'probabilidad_riesgo'
    "probabilidad_riesgo", 
    extraer_prob_riesgo("probabilidad_vector")
)

df_historico_riesgo = df_predicciones_con_riesgo.select(           # Limpiamos el DataFrame para quedarnos solo con lo necesario para la fase de negocio
    "id_unico_deportista",
    "activity_date",
    "probabilidad_riesgo",
    columna_objetivo
)

display(df_historico_riesgo.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualización de datos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mapeo de Colores para el Gráfico Comparativo
# MAGIC Transformamos las predicciones a Pandas para poder graficarlas. Aplicamos reglas visuales:
# MAGIC - **Histórico:** 1 (Lesión) = Negro | 0 (Sano) = Gris.
# MAGIC - **Predicción (Semáforo):** < 0.4 = Verde | 0.4 a 0.6 = Naranja | > 0.6 = Rojo.
# MAGIC Asignamos también una coordenada Y a cada jugador para separar sus líneas.

# COMMAND ----------

# 1. Pasamos el DataFrame de PySpark a Pandas
pdf = df_historico_riesgo.toPandas()

# 2. Aseguramos que la fecha tenga el formato correcto para el eje X
pdf['activity_date'] = pd.to_datetime(pdf['activity_date'])

# 3. Mapeo del histórico de lesiones (1 = Negro, 0 = Gris)
pdf['color_historico'] = np.where(pdf['injury_risk'] == 1, 'black', '#cccccc')

# 4. Mapeo del semáforo de probabilidad
condiciones_prob = [
    pdf['probabilidad_riesgo'] < 0.4,
    (pdf['probabilidad_riesgo'] >= 0.4) & (pdf['probabilidad_riesgo'] <= 0.6),
    pdf['probabilidad_riesgo'] > 0.6
]
colores_prob = ['green', 'orange', 'red']
pdf['color_prediccion'] = np.select(condiciones_prob, colores_prob, default='green')

# 5. Crear identificadores numéricos para el eje Y por cada jugador
lista_jugadores = sorted(pdf['id_unico_deportista'].unique())
mapeo_y = {jugador: i for i, jugador in enumerate(lista_jugadores)}
pdf['y_base'] = pdf['id_unico_deportista'].map(mapeo_y)

print(f"Datos preparados para {len(lista_jugadores)} jugadores. Listos para graficar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Renderizado del Gráfico
# MAGIC Dibujamos el lienzo con fondo blanco. Por cada jugador, creamos dos líneas paralelas usando marcadores cuadrados:
# MAGIC - La línea superior (desplazada +0.15) muestra la realidad (Histórico).
# MAGIC - La línea inferior (desplazada -0.15) muestra la IA (Predicción).
# MAGIC Añadimos una leyenda explicativa personalizada para facilitar la lectura al cuerpo técnico.

# COMMAND ----------

# Configuramos el tamaño del gráfico dinámicamente según la cantidad de jugadores
plt.figure(figsize=(16, len(lista_jugadores) * 1.5))
ax = plt.gca()

# Fondo blanco limpio y profesional
ax.set_facecolor('white')

# Linea superior: Histórico de Lesiones (desplazado +0.15)
plt.scatter(
    pdf['activity_date'], 
    pdf['y_base'] + 0.15, 
    c=pdf['color_historico'], 
    marker='s', 
    s=100, 
    edgecolors='none'
)

# Linea inferior: Probabilidad Predicha / Semáforo (desplazado -0.15)
plt.scatter(
    pdf['activity_date'], 
    pdf['y_base'] - 0.15, 
    c=pdf['color_prediccion'], 
    marker='s', 
    s=100, 
    edgecolors='none'
)

# Configuración de los ejes y estética
plt.yticks(range(len(lista_jugadores)), lista_jugadores, fontsize=12)
plt.xlabel("Fechas", fontsize=14, labelpad=10)
plt.ylabel("Jugadores", fontsize=14, labelpad=10)
plt.title("Auditoría de IA: Histórico Real vs Predicción de Riesgo", fontsize=16, pad=20)

# Cuadrícula sutil para guiar el ojo sin ensuciar el fondo blanco
plt.grid(True, color='#f0f0f0', linestyle='-', linewidth=1)

# Creación de la leyenda personalizada para la reunión
leyenda_elementos = [
    mpatches.Patch(color='black', label='Histórico: Lesión Real (1)'),
    mpatches.Patch(color='#cccccc', label='Histórico: Sano (0)'),
    mpatches.Patch(color='green', label='Predicción: Riesgo Bajo (< 0.3)'),
    mpatches.Patch(color='orange', label='Predicción: Riesgo Medio (0.3 - 0.6)'),
    mpatches.Patch(color='red', label='Predicción: Riesgo Alto (> 0.6)')
]

# Colocamos la leyenda fuera del gráfico para que no tape los datos
plt.legend(handles=leyenda_elementos, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)

# Ajuste automático de márgenes y visualización en Databricks
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Cell 34
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Recuperamos los coeficientes del modelo entrenado
coeficientes = modelo_entrenado.coefficients.toArray()

# 2. Creamos un DataFrame de Pandas emparejando el nombre de la columna con su peso matemático
df_importancia = pd.DataFrame({
    'Variable': valid_features,
    'Impacto': coeficientes
})

# 3. Calculamos el impacto absoluto para poder ordenar las variables de mayor a menor relevancia global
df_importancia['Impacto_Absoluto'] = df_importancia['Impacto'].abs()
df_importancia = df_importancia.sort_values(by='Impacto_Absoluto', ascending=False)

# 4. Mostramos el Top 10 por pantalla en texto
print("TOP VARIABLES QUE MÁS AFECTAN AL SEMÁFORO DE RIESGO:")
print("-" * 65)
print(df_importancia[['Variable', 'Impacto']].head(15).to_string(index=False))
print("-" * 65)

# 5. Diseñamos un gráfico de barras para presentar los resultados en el cuerpo técnico
plt.figure(figsize=(12, len(columnas_features) * 0.4 if len(columnas_features) > 10 else 6))

# Creamos un código de colores: Rojo para variables que aumentan riesgo, Verde para las que protegen
colores = ['#d9534f' if x > 0 else '#5cb85c' for x in df_importancia['Impacto']]

sns.barplot(
    x='Impacto', 
    y='Variable', 
    data=df_importancia, 
    palette=colores
)

plt.title("Impacto de las Variables en el Semáforo de Riesgo (Coeficientes del Modelo)", fontsize=14, pad=15)
plt.xlabel("Dirección e Intensidad del Impacto (Coeficiente)", fontsize=12)
plt.ylabel("Variables evaluadas", fontsize=12)
plt.grid(True, axis='x', color='#f0f0f0', linestyle='--')

# Añadimos una leyenda explicativa para la reunión con los preparadores
import matplotlib.patches as mpatches
leyenda = [
    mpatches.Patch(color='#d9534f', label='Aumenta el Riesgo (Favorece el Rojo/Naranja)'),
    mpatches.Patch(color='#5cb85c', label='Disminuye el Riesgo (Favorece el Verde)')
]
plt.legend(handles=leyenda, loc='lower right')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,RF - Separador
# MAGIC %md
# MAGIC # Random Forest — Comparativa con Regresión Logística
# MAGIC Entrenamos un Random Forest sobre exactamente los mismos datos preparados (`df_entrenamiento`) para comparar rendimiento y distribución de riesgo frente al modelo de Regresión Logística.

# COMMAND ----------

# DBTITLE 1,RF - Configuración y Entrenamiento
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=columna_objetivo,
    predictionCol="prediccion_dura_rf",
    probabilityCol="probabilidad_vector_rf",
    weightCol="weight",
    numTrees=100,
    maxDepth=5,
    seed=42
)

modelo_rf = rf.fit(df_entrenamiento)
print("¡Entrenamiento del Random Forest completado con éxito!")

# COMMAND ----------

# DBTITLE 1,RF - Generación de Probabilidades de Riesgo
df_predicciones_raw_rf = modelo_rf.transform(df_entrenamiento)

extraer_prob_rf = udf(lambda v: float(v[1]), DoubleType())

df_predicciones_rf = df_predicciones_raw_rf.withColumn(
    "probabilidad_riesgo",
    extraer_prob_rf("probabilidad_vector_rf")
)

df_historico_riesgo_rf = df_predicciones_rf.select(
    "id_unico_deportista",
    "activity_date",
    "probabilidad_riesgo",
    columna_objetivo
)

display(df_historico_riesgo_rf.limit(5))

# COMMAND ----------

# DBTITLE 1,RF - Mapeo de Colores para el Gráfico
pdf_rf = df_historico_riesgo_rf.toPandas()
pdf_rf['activity_date'] = pd.to_datetime(pdf_rf['activity_date'])

# Histórico: 1 (Lesión) = Negro | 0 (Sano) = Gris
pdf_rf['color_historico'] = np.where(pdf_rf[columna_objetivo] == 1, 'black', '#cccccc')

# Semáforo de probabilidad
condiciones_rf = [
    pdf_rf['probabilidad_riesgo'] < 0.4,
    (pdf_rf['probabilidad_riesgo'] >= 0.4) & (pdf_rf['probabilidad_riesgo'] <= 0.6),
    pdf_rf['probabilidad_riesgo'] > 0.6
]
pdf_rf['color_prediccion'] = np.select(condiciones_rf, ['green', 'orange', 'red'], default='green')

lista_jugadores_rf = sorted(pdf_rf['id_unico_deportista'].unique())
mapeo_y_rf = {j: i for i, j in enumerate(lista_jugadores_rf)}
pdf_rf['y_base'] = pdf_rf['id_unico_deportista'].map(mapeo_y_rf)

print(f"Datos preparados para {len(lista_jugadores_rf)} jugadores. Listos para graficar.")

# COMMAND ----------

# DBTITLE 1,RF - Timeline Histórico vs Predicción
plt.figure(figsize=(16, len(lista_jugadores_rf) * 1.5))
ax = plt.gca()
ax.set_facecolor('white')

# Línea superior: Histórico real
plt.scatter(
    pdf_rf['activity_date'], pdf_rf['y_base'] + 0.15,
    c=pdf_rf['color_historico'], marker='s', s=100, edgecolors='none'
)

# Línea inferior: Semáforo RF
plt.scatter(
    pdf_rf['activity_date'], pdf_rf['y_base'] - 0.15,
    c=pdf_rf['color_prediccion'], marker='s', s=100, edgecolors='none'
)

plt.yticks(range(len(lista_jugadores_rf)), lista_jugadores_rf, fontsize=12)
plt.xlabel("Fechas", fontsize=14, labelpad=10)
plt.ylabel("Jugadores", fontsize=14, labelpad=10)
plt.title("Random Forest: Auditoría — Histórico Real vs Predicción de Riesgo", fontsize=16, pad=20)
plt.grid(True, color='#f0f0f0', linestyle='-', linewidth=1)

leyenda_rf = [
    mpatches.Patch(color='black',   label='Histórico: Lesión Real (1)'),
    mpatches.Patch(color='#cccccc', label='Histórico: Sano (0)'),
    mpatches.Patch(color='green',   label='Predicción: Riesgo Bajo (< 0.4)'),
    mpatches.Patch(color='orange',  label='Predicción: Riesgo Medio (0.4 – 0.6)'),
    mpatches.Patch(color='red',     label='Predicción: Riesgo Alto (> 0.6)')
]
plt.legend(handles=leyenda_rf, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,RF - Importancia de Features
importancias_rf = modelo_rf.featureImportances.toArray()

df_importancia_rf = pd.DataFrame({
    'Variable': valid_features,
    'Importancia': importancias_rf
}).sort_values(by='Importancia', ascending=False)

print("TOP VARIABLES — RANDOM FOREST (Importancia por reducción de impureza):")
print("-" * 65)
print(df_importancia_rf.head(15).to_string(index=False))
print("-" * 65)

plt.figure(figsize=(12, len(valid_features) * 0.4 if len(valid_features) > 10 else 6))

sns.barplot(
    x='Importancia',
    y='Variable',
    data=df_importancia_rf,
    color='#5b9bd5'
)

plt.title("Importancia de Variables — Random Forest", fontsize=14, pad=15)
plt.xlabel("Importancia (reducción media de impureza de Gini)", fontsize=12)
plt.ylabel("Variables evaluadas", fontsize=12)
plt.grid(True, axis='x', color='#f0f0f0', linestyle='--')
plt.tight_layout()
plt.show()
