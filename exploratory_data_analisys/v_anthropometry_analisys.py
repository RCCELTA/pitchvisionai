# Databricks notebook source
# MAGIC %pip install matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
import pandas as pd
from pyspark.sql.window import Window
import seaborn as sns

# COMMAND ----------

# ============================================================
# KEEP ALIVE LIGERO PARA DATABRICKS
# Mantiene el cluster activo durante un tiempo limitado
# ============================================================

import time
from datetime import datetime

# Configuración
DURATION_MINUTES = 60   # tiempo total que quieres mantenerlo activo
PING_EVERY_MINUTES = 2  # cada cuántos minutos hace una operación mínima

num_pings = DURATION_MINUTES // PING_EVERY_MINUTES

print(f"Iniciando keep alive durante {DURATION_MINUTES} minutos...")
print(f"Ping cada {PING_EVERY_MINUTES} minutos.")

for i in range(int(num_pings)):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Ping {i + 1}/{int(num_pings)}")

    # Operación Spark mínima
    spark.range(1).count()

    time.sleep(PING_EVERY_MINUTES * 60)

print("Keep alive terminado.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Tabla v_anthropometry

# COMMAND ----------

TABLE_NAME = "dev_silver.pitchvision.v_anthropometry"  

df = spark.table(TABLE_NAME)

print(f"EDA DE TABLA: {TABLE_NAME}")

num_rows = df.count()
num_cols = len(df.columns)

print("\n1. DIMENSIONES")
print("-" * 100)
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
# MAGIC Solo hay dos columnas categóricas: id  y la temporada que realmente no nos dan información para este estudio de clusters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conteo de nulos

# COMMAND ----------

#  Nulos por columna

print("\n6. NULOS POR COLUMNA")
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]

nulls_df = df.agg(*null_exprs)
#da la vuelta a la tabla para que quede en horizontal y sea más legible
nulls_long = nulls_df.select(
    F.explode(
        F.array(*[
            F.struct(
                F.lit(c).alias("columna"),
                F.col(c).alias("nulos")
            )
            for c in df.columns
        ])
    ).alias("x")
).select("x.*")

nulls_report = (
    nulls_long
    .withColumn("filas_totales", F.lit(num_rows))
    .withColumn(
        "porcentaje_nulos",
        F.round((F.col("nulos") / F.col("filas_totales")) * 100, 2)
    )
    .orderBy(F.col("porcentaje_nulos").desc())
)

display(nulls_report)

# COMMAND ----------

# MAGIC %md
# MAGIC Hay muchas columnas con más de un 50% de nulos, habrá que considerar que hacer con estos nulos, si eliminar esas columnas imputar con alguna estrategia o simplemente tratarlos como nulos.

# COMMAND ----------

df.select("id_unico_deportista").distinct().count()

# COMMAND ----------

df = spark.table("dev_silver.pitchvision.v_anthropometry")
df.agg(
    F.min("age_years").alias("edad_minima"),
    F.max("age_years").alias("edad_maxima")
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Imputación de nulos
# MAGIC 1. Mantiene el valor original si existe.
# MAGIC 2. Si es nulo, usa el valor anterior del mismo jugador.
# MAGIC 3. Si no hay anterior, usa el posterior del mismo jugador.
# MAGIC 4. Si sigue nulo, usa la mediana del grupo de edad.
# MAGIC 5. Elimina columnas técnicas.

# COMMAND ----------

ID_COL = "id_unico_deportista"
DATE_COL = "test_date"
AGE_COL = "age_years"

EXCLUDE_COLS = [
    ID_COL,
    AGE_COL,
    "is_restricted"
]

numeric_cols_to_impute = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in EXCLUDE_COLS
]

print(f"Columnas numéricas a imputar: {len(numeric_cols_to_impute)}")

# 1. FORWARD FILL + BACKWARD FILL POR JUGADOR

w_forward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(Window.unboundedPreceding, 0)
)

w_backward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(0, Window.unboundedFollowing)
)

df_imputed = df

for c in numeric_cols_to_impute:
    last_value = F.last(F.col(c), ignorenulls=True).over(w_forward)
    next_value = F.first(F.col(c), ignorenulls=True).over(w_backward)

    df_imputed = df_imputed.withColumn(
        c,
        F.coalesce(
            F.col(c),
            last_value,
            next_value
        )
    )

print("Imputación por histórico del jugador definida.")

# 2. CREAR GRUPOS DE EDAD

df_imputed = (
    df_imputed
    .withColumn(
        "_age_group_imputation",
        F.when((F.col(AGE_COL) >= 13) & (F.col(AGE_COL) <= 15), F.lit("13-15"))
         .when((F.col(AGE_COL) >= 16) & (F.col(AGE_COL) <= 18), F.lit("16-18"))
         .when((F.col(AGE_COL) >= 19) & (F.col(AGE_COL) <= 22), F.lit("19-22"))
         .when((F.col(AGE_COL) >= 23) & (F.col(AGE_COL) <= 28), F.lit("23-28"))
         .when((F.col(AGE_COL) >= 29) & (F.col(AGE_COL) <= 33), F.lit("29-33"))
         .when((F.col(AGE_COL) >= 34) & (F.col(AGE_COL) <= 39), F.lit("34-39"))
         .otherwise(F.lit(None))
    )
)

# 3. CALCULAR MEDIANAS POR GRUPO DE EDAD

median_exprs = [
    F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}__age_group_median")
    for c in numeric_cols_to_impute
]

age_group_medians = (
    df_imputed
    .filter(F.col("_age_group_imputation").isNotNull())
    .groupBy("_age_group_imputation")
    .agg(*median_exprs)
)

# 4. UNIR MEDIANAS Y RELLENAR NULOS RESTANTES

df_imputed = df_imputed.join(
    age_group_medians,
    on="_age_group_imputation",
    how="left"
)

for c in numeric_cols_to_impute:
    median_col = f"{c}__age_group_median"

    df_imputed = (
        df_imputed
        .withColumn(
            c,
            F.coalesce(
                F.col(c),
                F.col(median_col)
            )
        )
        .drop(median_col)
    )

# 5. ELIMINAR COLUMNA TÉCNICA

df_imputed = df_imputed.drop("_age_group_imputation")

# Ejecutamos una sola acción al final
df_imputed = df_imputed.cache()

print(f"Filas finales: {df_imputed.count()}")
display(df_imputed.limit(10))

# COMMAND ----------

df = df_imputed

# COMMAND ----------

#cacheado para operaciones más rápidas
df = df.cache()
df.count()

# COMMAND ----------

print("\nNULOS POR COLUMNA DESPUÉS DE LA IMPUTACIÓN")
null_exprs_post = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]

nulls_df_post = df.agg(*null_exprs_post)
nulls_long_post = nulls_df_post.select(
    F.explode(
        F.array(*[
            F.struct(
                F.lit(c).alias("columna"),
                F.col(c).alias("nulos")
            )
            for c in df.columns
        ])
    ).alias("x")
).select("x.*")

nulls_report_post = (
    nulls_long_post
    .withColumn("filas_totales", F.lit(df.count()))
    .withColumn(
        "porcentaje_nulos",
        F.round((F.col("nulos") / F.col("filas_totales")) * 100, 2)
    )
    .orderBy(F.col("porcentaje_nulos").desc())
)

display(nulls_report_post)

# COMMAND ----------

# MAGIC %md
# MAGIC Ultimo paso imputar nulos con la mediana global.

# COMMAND ----------

numeric_cols_final = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
]

# Calculamos la mediana global de cada columna numérica
global_medians = df.agg(
    *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_global_median") for c in numeric_cols_final]
).collect()[0]

df_final = df
for c in numeric_cols_final:
    median_value = global_medians[f"{c}_global_median"]
    df_final = df_final.withColumn(
        c,
        F.coalesce(F.col(c), F.lit(median_value))
    )

df_final = df_final.cache()

print("Imputación por mediana global completada.")
display(df_final.limit(10))

# COMMAND ----------

print(df.is_cached)
print(df_final.is_cached)
print(df.count())

# COMMAND ----------

df = df_final

print("\n8. PERCENTILES DE COLUMNAS NUMÉRICAS:")

# Calculamos todos los percentiles en una sola acción Spark
percentile_exprs = []
for c in numeric_cols:
    percentile_exprs.append(
        F.expr(f"percentile_approx(`{c}`, array(0.05, 0.25, 0.50, 0.75, 0.95), 10000)").alias(c)
    )

percentiles_raw = df.agg(*percentile_exprs).collect()[0]

# Convertimos a filas para crear el DataFrame de percentiles
percentile_rows = []
for c in numeric_cols:
    p = percentiles_raw[c]
    if p is not None and len(p) == 5:
        percentile_rows.append((c, float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

if percentile_rows:
    percentiles_df = spark.createDataFrame(
        percentile_rows,
        ["columna", "p05", "p25", "p50_mediana", "p75", "p95"]
    )
    display(percentiles_df)
else:
    print("No hay percentiles disponibles.")

# COMMAND ----------

# MAGIC %md
# MAGIC Estadísticas descriptivas de las variables numéricas.
# MAGIC Haría falta hacer un estudio de outliers para que no haya problemas con los algoritmos de clustering como KMeans, no sé como de limpios están los datos o si puede haber alguna medición erronea con algun valor extremo que nos pueda afectar negativamente a la hora de usar algoritmos basados en distancias.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Detección de outliers

# COMMAND ----------

# DBTITLE 1,Detección outliers percentiles
# Convertimos percentiles_df a diccionario local
percentiles_dict = {
    row["columna"]: {
        "p05": row["p05"],
        "p95": row["p95"]
    }
    for row in percentiles_df.collect()
}

total_rows = df.count()

# Calculamos valores_bajos y valores_altos para TODAS las columnas en UNA sola pasada
agg_exprs = []
for c in numeric_cols:
    if c not in percentiles_dict:
        continue
    p05 = percentiles_dict[c]["p05"]
    p95 = percentiles_dict[c]["p95"]
    agg_exprs.append(F.sum(F.when(F.col(c) < F.lit(p05), 1).otherwise(0)).alias(f"{c}__bajos"))
    agg_exprs.append(F.sum(F.when(F.col(c) > F.lit(p95), 1).otherwise(0)).alias(f"{c}__altos"))

print(f"Calculando outliers para {len(numeric_cols)} columnas en UNA sola pasada...")
results = df.agg(*agg_exprs).collect()[0]

# Construimos el DataFrame de resultados
outlier_rows = []
for c in numeric_cols:
    if c not in percentiles_dict:
        continue
    p05 = percentiles_dict[c]["p05"]
    p95 = percentiles_dict[c]["p95"]
    valores_bajos = results[f"{c}__bajos"]
    valores_altos = results[f"{c}__altos"]
    total_fuera = valores_bajos + valores_altos

    outlier_rows.append((
        c, p05, p95,
        valores_bajos, valores_altos, total_fuera,
        round(total_fuera / total_rows * 100, 2)
    ))

outliers_percentiles_df = spark.createDataFrame(
    outlier_rows,
    [
        "columna", "p05", "p95",
        "valores_menores_p05", "valores_mayores_p95",
        "total_fuera_p05_p95", "porcentaje_fuera_p05_p95"
    ]
).orderBy(F.col("total_fuera_p05_p95").desc())

display(outliers_percentiles_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a ver que variables analizar en más detalle calculando aquellas que tienen extremos más alejados de los valores de percentil 0% y 95%.

# COMMAND ----------

# DBTITLE 1,Severidad extremos una sola pasada
ID_COL = "id_unico_deportista"

# Convertimos percentiles_df a diccionario local
percentiles_dict = {
    row["columna"]: {
        "p05": row["p05"],
        "p25": row["p25"],
        "p50": row["p50_mediana"],
        "p75": row["p75"],
        "p95": row["p95"]
    }
    for row in percentiles_df.collect()
}

# Construimos todas las expresiones de agregación en una sola pasada
agg_exprs = []
cols_with_iqr = []

for c in numeric_cols:
    if c not in percentiles_dict:
        continue
    p05 = percentiles_dict[c]["p05"]
    p25 = percentiles_dict[c]["p25"]
    p75 = percentiles_dict[c]["p75"]
    p95 = percentiles_dict[c]["p95"]
    iqr = p75 - p25

    if iqr == 0 or iqr is None:
        continue

    cols_with_iqr.append((c, p05, p25, p75, p95, iqr))

    # Expresión del exceso normalizado
    excess_expr = (
        F.when(F.col(c) < F.lit(p05), (F.lit(p05) - F.col(c)) / F.lit(iqr))
         .when(F.col(c) > F.lit(p95), (F.col(c) - F.lit(p95)) / F.lit(iqr))
         .otherwise(F.lit(0.0))
    )

    agg_exprs.append(F.count(F.when(F.col(c).isNotNull(), True)).alias(f"{c}__n_no_nulos"))
    agg_exprs.append(F.sum(F.when((F.col(c) < F.lit(p05)) | (F.col(c) > F.lit(p95)), 1).otherwise(0)).alias(f"{c}__n_fuera"))
    agg_exprs.append(F.avg(excess_expr).alias(f"{c}__media_exceso"))
    agg_exprs.append(F.max(excess_expr).alias(f"{c}__max_exceso"))
    agg_exprs.append(F.expr(f"percentile_approx(CASE WHEN `{c}` < {p05} THEN ({p05} - `{c}`) / {iqr} WHEN `{c}` > {p95} THEN (`{c}` - {p95}) / {iqr} ELSE 0.0 END, 0.95, 10000)").alias(f"{c}__p95_exceso"))

print(f"Calculando severidad de extremos para {len(cols_with_iqr)} columnas en UNA sola pasada...")
results = df.agg(*agg_exprs).collect()[0]

# Construimos el DataFrame de resultados
extreme_severity_rows = []
for c, p05, p25, p75, p95, iqr in cols_with_iqr:
    n_no_nulos = results[f"{c}__n_no_nulos"]
    n_fuera = results[f"{c}__n_fuera"]
    media_exceso = results[f"{c}__media_exceso"]
    max_exceso = results[f"{c}__max_exceso"]
    p95_exceso = results[f"{c}__p95_exceso"]

    extreme_severity_rows.append((
        c, p05, p25, p75, p95, iqr,
        n_no_nulos, n_fuera,
        float(media_exceso) if media_exceso is not None else None,
        float(p95_exceso) if p95_exceso is not None else None,
        float(max_exceso) if max_exceso is not None else None
    ))

extreme_severity_df = spark.createDataFrame(
    extreme_severity_rows,
    [
        "columna", "p05", "p25", "p75", "p95", "iqr",
        "n_no_nulos", "n_fuera_p05_p95",
        "media_exceso_normalizado", "p95_exceso_normalizado", "max_exceso_normalizado"
    ]
)

display(
    extreme_severity_df
    .orderBy(F.col("max_exceso_normalizado").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC Para evaluar valores extremos se calculó el `max_exceso_normalizado`, que mide cuánto se aleja el valor más extremo de una variable respecto a sus límites p05-p95, normalizado por el rango intercuartílico de la propia variable (`IQR = p75 - p25`). Si un valor está dentro del rango p05-p95, su exceso es 0. Si está por debajo de p05 o por encima de p95, se calcula la distancia hasta ese límite y se divide entre el IQR. De esta forma, el indicador permite comparar variables con unidades distintas. Por ejemplo, un `max_exceso_normalizado = 3` significa que el valor más extremo está a una distancia equivalente a 3 IQRs más allá del percentil p05 o p95. Valores altos no implican necesariamente error, pero sí indican variables o registros que deben revisarse manualmente.

# COMMAND ----------

# PINTAR DISTRIBUCIONES DE VARIABLES CON max_exceso_normalizado > 5
THRESHOLD_EXCESO = 5
MAX_POINTS = 10000

# Seleccionamos todas las variables con max_exceso_normalizado > 5
extreme_cols_rows = (
    extreme_severity_df
    .filter(F.col("max_exceso_normalizado") > THRESHOLD_EXCESO)
    .orderBy(F.col("max_exceso_normalizado").desc())
    .collect()
)

top_extreme_cols = [row["columna"] for row in extreme_cols_rows]

print(f"Variables con max_exceso_normalizado > {THRESHOLD_EXCESO}:")
print(f"Total variables seleccionadas: {len(top_extreme_cols)}")
print(top_extreme_cols)

if len(top_extreme_cols) == 0:
    print(f"No hay variables con max_exceso_normalizado > {THRESHOLD_EXCESO}")

else:
    for c in top_extreme_cols:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]
        iqr = p75 - p25

        plot_df = (
            df
            .select(c)
            .where(F.col(c).isNotNull())
            .limit(MAX_POINTS)
            .toPandas()
        )

        if plot_df.empty:
            print(f"No hay datos para plotear en columna: {c}")
            continue

        min_val = plot_df[c].min()
        max_val = plot_df[c].max()

        lower_excess = max(0, p05 - min_val) / iqr if iqr != 0 else 0
        upper_excess = max(0, max_val - p95) / iqr if iqr != 0 else 0

        max_exceso = max(lower_excess, upper_excess)

        plt.figure(figsize=(8, 5))

        plt.hist(
            plot_df[c],
            bins=50,
            alpha=0.8
        )

        plt.axvline(
            p05,
            linestyle="--",
            linewidth=2,
            label=f"p05 = {p05:.2f}"
        )

        plt.axvline(
            p95,
            linestyle="--",
            linewidth=2,
            label=f"p95 = {p95:.2f}"
        )

        plt.xlabel(c)
        plt.ylabel("Frecuencia")
        plt.title(
            f"Distribución de {c}\n"
            f"max_exceso_normalizado ≈ {max_exceso:.2f} IQR | "
            f"Exceso inferior: {lower_excess:.2f} IQR | "
            f"Exceso superior: {upper_excess:.2f} IQR"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos claramente que muchas variables presentan outliers claros. Hay que tomar una decisión de que hacer con ellos.

# COMMAND ----------

print("\n11. RANGO DE COLUMNA FECHA: test_date")

display(
    df.agg(
        F.min("test_date").alias("fecha_minima"),
        F.max("test_date").alias("fecha_maxima")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Los test han sido realizados a lo largo de algo más de dos años.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estudio de correlación de variables

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a rellenar los nulos con la mediana aproximada, habría que estudiar si es lo mejor para cada variable.

# COMMAND ----------

df_cleaned = df
# Vectorizamos columnas numéricas
assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features_corr",
    handleInvalid="keep"
)

df_vector_corr = assembler.transform(df_cleaned.select(*numeric_cols))

# Calculamos matriz de correlación
corr_matrix = Correlation.corr(
    df_vector_corr,
    "features_corr",
    method="pearson"
).head()[0]

# Convertimos a pandas para visualizar
corr_array = corr_matrix.toArray()

corr_pd = pd.DataFrame(
    corr_array,
    index=numeric_cols,
    columns=numeric_cols
)

display(corr_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Difícil de interpretar, vamos a utilizar un umbral de alta correlación para filtrar:

# COMMAND ----------

CORR_THRESHOLD = 0.95
high_corr_pairs = []

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        corr_value = corr_pd.iloc[i, j]
        
        if abs(corr_value) >= CORR_THRESHOLD:
            high_corr_pairs.append({
                "variable_1": numeric_cols[i],
                "variable_2": numeric_cols[j],
                "correlacion": float(corr_value),
                "correlacion_abs": float(abs(corr_value))
            })

high_corr_pd = pd.DataFrame(high_corr_pairs)

high_corr_pd = high_corr_pd.sort_values("correlacion_abs", ascending=False)
display(high_corr_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos claramente que hay pares de variables extremadamente correlacionadas, se propone eliminar una de las dos en aquellas parejas que superen el umbral.
# MAGIC
# MAGIC Para cada pareja hay que decidir cuál de las dos eliminamos siguiendo un criterio enfocado en la fisiología:
# MAGIC
# MAGIC | Pareja correlacionada | Mantener | Eliminar | Criterio |
# MAGIC |---|---|---|---|
# MAGIC | `head_perimeter_cm` / `rocha_head_bone_mass_kg` | `head_perimeter_cm` | `rocha_head_bone_mass_kg` | El perímetro craneal es una medida directa. La masa ósea de cabeza estimada por Rocha es una variable derivada. |
# MAGIC | `ponderal_index` / `ectomorphy` | `ectomorphy` | `ponderal_index` | Para perfiles corporales/deportivos, `ectomorphy` es más interpretable dentro del somatotipo. |
# MAGIC | `thoracic_perimeter_cm` / `thoracic_perimeter_corrected_cm` | `thoracic_perimeter_corrected_cm` | `thoracic_perimeter_cm` | Variable corregida. |
# MAGIC | `carter_fat_pct` / `faulkner_fat_pct` | `faulkner_plus_carter_mean_fat_pct` | `carter_fat_pct`, `faulkner_fat_pct` | Si se mantiene la media Faulkner+Carter, no es necesario conservar ambas estimaciones individuales en una primera prueba. |
# MAGIC | `carter_fat_pct` / `faulkner_plus_carter_mean_fat_pct` | `faulkner_plus_carter_mean_fat_pct` | `carter_fat_pct` | La media combina métodos y puede ser más estable. |
# MAGIC | `faulkner_fat_pct` / `faulkner_plus_carter_mean_fat_pct` | `faulkner_plus_carter_mean_fat_pct` | `faulkner_fat_pct` | La media resume ambas estimaciones de porcentaje graso. |
# MAGIC | `skinfold_sum_6_mm` / `skinfold_sum_8_mm` | `skinfold_sum_8_mm` | `skinfold_sum_6_mm` | La suma de 8 pliegues contiene más información antropométrica que la de 6. |
# MAGIC | `diff_brut_estimated_mass_kg` / `diff_structural_brut_mass_pct` | `diff_structural_brut_mass_pct` | `diff_brut_estimated_mass_kg` | El porcentaje es más comparable entre deportistas de distinto tamaño corporal. |
# MAGIC | `arm_perimeter_corrected_cm` / `arm_contracted_perimeter_corrected_cm` | `arm_contracted_perimeter_corrected_cm` | `arm_perimeter_corrected_cm` | Para perfil muscular, el perímetro contraído corregido suele ser más informativo. |
# MAGIC | `arm_contracted_perimeter_cm` / `arm_contracted_perimeter_corrected_cm` | `arm_contracted_perimeter_corrected_cm` | `arm_contracted_perimeter_cm` | Es preferible la variable corregida frente a la bruta. |
# MAGIC | `kerr_fat_mass_pct` / `fat_muscle_index` | `fat_muscle_index` | `kerr_fat_mass_pct` | El índice grasa/músculo resume mejor la composición corporal relativa. |
# MAGIC | `body_surface_area` / `skin_mass_kg` | `body_surface_area` | `skin_mass_kg` | La superficie corporal es más interpretable y general; la masa de piel es una variable derivada. |
# MAGIC | `arm_relaxed_perimeter_cm` / `arm_perimeter_corrected_cm` | `arm_perimeter_corrected_cm` | `arm_relaxed_perimeter_cm` | La corregida reduce el efecto del tejido graso/subcutáneo respecto al perímetro relajado bruto. |
# MAGIC | `arm_relaxed_perimeter_cm` / `arm_contracted_perimeter_cm` | `arm_contracted_perimeter_corrected_cm` | `arm_relaxed_perimeter_cm` | Dentro de la familia de perímetros de brazo, el perímetro contraído corregido es el representante más informativo para esta primera clusterización. |
# MAGIC
# MAGIC Destacar que esto habría que consultarlo con algún experto, esto simplemente es una primera aproximación.
# MAGIC También considerar si aumentar o reducir el umbral.

# COMMAND ----------

cols_to_drop_corr_pairs = [
    "rocha_head_bone_mass_kg",
    "ponderal_index",
    "thoracic_perimeter_cm",
    "carter_fat_pct",
    "faulkner_fat_pct",
    "skinfold_sum_6_mm",
    "diff_brut_estimated_mass_kg",
    "arm_perimeter_corrected_cm",
    "arm_contracted_perimeter_cm",
    "kerr_fat_mass_pct",
    "skin_mass_kg",
    "arm_relaxed_perimeter_cm"
]


df_reduced = df_cleaned.drop(*cols_to_drop_corr_pairs)
print(f"Columnas restantes: {len(df_reduced.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prueba KNN
# MAGIC Primera prueba solo con variables numéricas

# COMMAND ----------


# Columna identificadora del deportista
ID_COL = "id_unico_deportista"

# Número de componentes PCA
PCA_K = 5

# Número de clusters a probar
K_VALUES = [2, 3, 5]

# Semilla para reproducibilidad
SEED = 42

# Máximo de puntos a pintar en cada gráfico
MAX_POINTS_PLOT = 10000

numeric_cols = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name != ID_COL
]

print("\nCOLUMNAS NUMÉRICAS USADAS COMO FEATURES:")
print(numeric_cols)
print(f"Número de features numéricas: {len(numeric_cols)}")


# IMPUTACIÓN BÁSICA DE NULOS CON MEDIANA
df_model = df

median_values = {}

for c in numeric_cols:
    try:
        median = df_model.approxQuantile(c, [0.5], 0.01)[0]
        median_values[c] = median
    except Exception:
        median_values[c] = 0

df_model = df_model.fillna(median_values)

# Seleccionamos ID + columnas numéricas
select_cols = [ID_COL] + numeric_cols

df_model = df_model.select(*select_cols)


# VECTOR ASSEMBLER
# Convierte columnas numéricas en un vector de features

assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features_raw",
    handleInvalid="keep"
)

df_vector = assembler.transform(df_model)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

pca = PCA(
    k=PCA_K,
    inputCol="features_scaled",
    outputCol="pca_features"
)

pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

print("\nVARIANZA EXPLICADA POR PCA:")

explained_variance = pca_model.explainedVariance.toArray()

for i, v in enumerate(explained_variance):
    print(f"PC{i+1}: {v:.4f} ({v*100:.2f}%)")

print(
    f"Varianza explicada acumulada: "
    f"{explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)"
)

# CONVERTIR VECTOR PCA A COLUMNAS PC1 Y PC2
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")

df_pca_view = (
    df_pca
    .withColumn("pca_array", to_array(F.col("pca_features")))
    .withColumn("PC1", F.col("pca_array")[0])
    .withColumn("PC2", F.col("pca_array")[1])
    .drop("pca_array")
)

print("\nMUESTRA PCA:")
display(df_pca_view.select(ID_COL, "PC1", "PC2").limit(20))

# KMEANS + PLOT DE CLUSTERS SOBRE PCA
results = []

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for k in K_VALUES:
    print("=" * 100)
    print(f"Entrenando KMeans con k={k}...")
    
    kmeans = KMeans(
        k=k,
        seed=SEED,
        featuresCol="features_scaled",
        predictionCol="prediction"
    )
    model = kmeans.fit(df_scaled)
    
    # Aplicamos el modelo sobre df_pca_view para conservar PC1 y PC2
    predictions = model.transform(df_pca_view)
    
    silhouette = evaluator.evaluate(predictions)
    cost = model.summary.trainingCost
    
    cluster_sizes = (
        predictions
        .groupBy("prediction")
        .count()
        .orderBy("prediction")
    )
    
    results.append((k, silhouette, cost))
    
    print(f"k={k} | silhouette={silhouette:.4f} | cost={cost:.2f}")
    
    print("\nTamaño de clusters:")
    display(cluster_sizes)

    plot_df = (
        predictions
        .select(ID_COL, "PC1", "PC2", "prediction")
        .limit(MAX_POINTS_PLOT)
        .toPandas()
    )

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["prediction"],
        alpha=0.6,
        s=20
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans k={k} | Silhouette={silhouette:.4f}")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster")
    plt.show()

results_df = spark.createDataFrame(
    results,
    ["k", "silhouette", "training_cost"]
).orderBy("k")

print("\nRESULTADOS COMPARATIVOS K-MEANS:")
display(results_df)

best_row = results_df.orderBy(F.col("silhouette").desc()).first()

print("\nMEJOR K SEGÚN SILHOUETTE:")
print(f"k = {best_row['k']}")
print(f"silhouette = {best_row['silhouette']:.4f}")
print(f"training_cost = {best_row['training_cost']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC PCA = 2 39,7% de varianza acumulada
