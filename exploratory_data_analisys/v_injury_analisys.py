# Databricks notebook source
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

# MAGIC %md
# MAGIC # Tabla v_injury

# COMMAND ----------

TABLE_NAME = "dev_silver.pitchvision.v_injury"  

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
# MAGIC Que diferencia hay entre Menor y Leve en la variable total_injury_dates, posible pregunta a cliente.

# COMMAND ----------

display(df.select("injury_type").distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC Muscotendinosas es lo mismo que musculotendinosa

# COMMAND ----------

df = df.withColumn(
    "injury_type",
    F.when(
        F.col("injury_type") == "Muscotendinosa", "Musculotendinosas"
    ).otherwise(F.col("injury_type"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Transformamos la columna severity a numérica ordinal, el orden de severidad es N/A (no estaba cubierta, porque se han olvidado)  orden:  menor< leve< moderada< grave 

# COMMAND ----------

severity_clean = F.trim(F.lower(F.col("severity")))

df = df.withColumn(
    "severity",
    F.when(severity_clean == "menor", 0)
     .when(severity_clean == "leve", 1)
     .when(severity_clean == "moderada", 2)
     .when(severity_clean == "grave", 3)
     .otherwise(None)
)

df = df.fillna({"severity": -1})

# COMMAND ----------

display(df.select("severity").distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC El cliente pide predicción en lesión de isquiotibial, vamos a analizar la tabla completa pero realmente para predicción so necesitaríamos as filas de lesión de isquios.

# COMMAND ----------

df[df["affected_zone"] == "Isquiotibial"].count()

# COMMAND ----------

# MAGIC %md
# MAGIC Tenemos tan solo 94 casos de lesión de isquios habrá que ver como entrenar el modelo de regresión con tan pocos datos.

# COMMAND ----------

display(df.select("return_competition_date").distinct())

# COMMAND ----------

df_isquios = df.filter(F.col("affected_zone") == "Isquiotibial")
display(df_isquios)

# COMMAND ----------

# DBTITLE 1,Header nulos Injury
# MAGIC %md
# MAGIC ### Conteo de nulos

# COMMAND ----------

# DBTITLE 1,Nulos por columna Injury
#  Nulos por columna

print("\n6. NULOS POR COLUMNA")
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]

nulls_df = df.agg(*null_exprs)

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
# MAGIC En este caso los nulos de la variable return_training_date no tienen sentido imputarlo ya que el hecho de que no haya esa fecha es información valiosa en sí.
# MAGIC La variable day_code representa el día en el que se produjo la lesión respecto al día del partido (MD = Match day, MD-1, MD +1....)

# COMMAND ----------

df = df.filter(F.col("id_unico_deportista").isNotNull())

# COMMAND ----------

df = df.drop("return_competition_date", "return_training_date")

# COMMAND ----------

# Excluir temporada 23-24 (agosto 2023 a julio 2024)
df = df.filter(
    ~((F.col("injury_date") >= "2023-08-01") & (F.col("injury_date") < "2024-08-01"))
)


# COMMAND ----------

#  Nulos por columna

print("\n6. NULOS POR COLUMNA")
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]

nulls_df = df.agg(*null_exprs)

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

# DBTITLE 1,Percentiles Injury
# Excluimos columnas ID que no aportan al análisis
numeric_cols = [c for c in numeric_cols if c not in ("injury_id", "team_id")]

print("\n8. PERCENTILES DE COLUMNAS NUMÉRICAS:")

percentile_rows = []

for c in numeric_cols:
    try:
        p = df.approxQuantile(c, [0.05, 0.25, 0.50, 0.75, 0.95], 0.01)
        percentile_rows.append((c, p[0], p[1], p[2], p[3], p[4]))
    except Exception as e:
        print(f"No se pudieron calcular percentiles para {c}: {e}")

if percentile_rows:
    percentiles_df = spark.createDataFrame(
        percentile_rows,
        ["columna", "p05", "p25", "p50_mediana", "p75", "p95"]
    )
    display(percentiles_df)
else:
    print("No hay percentiles disponibles.")

# COMMAND ----------

# DBTITLE 1,Header outliers Injury
# MAGIC %md
# MAGIC ### Detección de outliers

# COMMAND ----------

# DBTITLE 1,Detección outliers percentiles Injury
outlier_rows = []

# Convertimos percentiles_df a diccionario local
percentiles_dict = {
    row["columna"]: {
        "p05": row["p05"],
        "p95": row["p95"]
    }
    for row in percentiles_df.collect()
}

total_rows = df.count()

for c in numeric_cols:
    try:
        p05 = percentiles_dict[c]["p05"]
        p95 = percentiles_dict[c]["p95"]

        valores_bajos = df.filter(F.col(c) < F.lit(p05)).count()
        valores_altos = df.filter(F.col(c) > F.lit(p95)).count()
        total_fuera = valores_bajos + valores_altos

        outlier_rows.append((
            c,
            p05,
            p95,
            valores_bajos,
            valores_altos,
            total_fuera,
            round(total_fuera / total_rows * 100, 2)
        ))

    except Exception as e:
        print(f"No se pudo calcular fuera de percentiles para {c}: {e}")

outliers_percentiles_df = spark.createDataFrame(
    outlier_rows,
    [
        "columna",
        "p05",
        "p95",
        "valores_menores_p05",
        "valores_mayores_p95",
        "total_fuera_p05_p95",
        "porcentaje_fuera_p05_p95"
    ]
).orderBy(F.col("total_fuera_p05_p95").desc())

display(outliers_percentiles_df)

# COMMAND ----------

# DBTITLE 1,Severidad de extremos Injury
ID_COL = "id_unico_deportista"

# Convertimos percentiles_df a diccionario local con más percentiles
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

extreme_severity_rows = []

for c in numeric_cols:
    try:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]

        iqr = p75 - p25

        if iqr == 0 or iqr is None:
            continue

        tmp = (
            df
            .select(c)
            .where(F.col(c).isNotNull())
            .withColumn(
                "exceso_extremo",
                F.when(F.col(c) < F.lit(p05), F.lit(p05) - F.col(c))
                 .when(F.col(c) > F.lit(p95), F.col(c) - F.lit(p95))
                 .otherwise(F.lit(0.0))
            )
            .withColumn(
                "exceso_extremo_normalizado",
                F.col("exceso_extremo") / F.lit(iqr)
            )
        )

        stats = tmp.agg(
            F.count("*").alias("n_no_nulos"),
            F.sum(F.when(F.col("exceso_extremo") > 0, 1).otherwise(0)).alias("n_fuera_p05_p95"),
            F.avg("exceso_extremo_normalizado").alias("media_exceso_norm"),
            F.max("exceso_extremo_normalizado").alias("max_exceso_norm"),
            F.expr("percentile_approx(exceso_extremo_normalizado, 0.95)").alias("p95_exceso_norm")
        ).collect()[0]

        extreme_severity_rows.append((
            c,
            p05,
            p25,
            p75,
            p95,
            iqr,
            stats["n_no_nulos"],
            stats["n_fuera_p05_p95"],
            float(stats["media_exceso_norm"]) if stats["media_exceso_norm"] is not None else None,
            float(stats["p95_exceso_norm"]) if stats["p95_exceso_norm"] is not None else None,
            float(stats["max_exceso_norm"]) if stats["max_exceso_norm"] is not None else None
        ))

    except Exception as e:
        print(f"No se pudo calcular severidad para {c}: {e}")

extreme_severity_df = spark.createDataFrame(
    extreme_severity_rows,
    [
        "columna",
        "p05",
        "p25",
        "p75",
        "p95",
        "iqr",
        "n_no_nulos",
        "n_fuera_p05_p95",
        "media_exceso_normalizado",
        "p95_exceso_normalizado",
        "max_exceso_normalizado"
    ]
)

display(
    extreme_severity_df
    .orderBy(F.col("max_exceso_normalizado").desc())
)

# COMMAND ----------

# DBTITLE 1,Distribuciones extremos Injury
# PINTAR DISTRIBUCIONES DE VARIABLES CON max_exceso_normalizado > 5

THRESHOLD_EXCESO = 5
MAX_POINTS = 10000

# Seleccionamos todas las variables con max_exceso_normalizado > umbral
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

        plt.hist(plot_df[c], bins=50, alpha=0.8)
        plt.axvline(p05, linestyle="--", linewidth=2, label=f"p05 = {p05:.2f}")
        plt.axvline(p95, linestyle="--", linewidth=2, label=f"p95 = {p95:.2f}")

        plt.xlabel(c)
        plt.ylabel("Frecuencia")
        plt.title(
            f"Distribuci\u00f3n de {c}\n"
            f"max_exceso_normalizado \u2248 {max_exceso:.2f} IQR | "
            f"Exceso inferior: {lower_excess:.2f} IQR | "
            f"Exceso superior: {upper_excess:.2f} IQR"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC En este caso al tratarse de una varibale que determina el tiempo de recuperación de una lesión no tendría porque ser outliers, podría simplemnente deberse a lesiones poco comunes y graves como rotura en el cruzado o algún otro tipo de lesión que mantenga al jugador fuera del campo más de un año.
# MAGIC
# MAGIC Comproabr si estos "outliers" son lesiones graves.

# COMMAND ----------

# DBTITLE 1,Rango fechas Injury
print("\n11. RANGO DE COLUMNA FECHA: injury_date")

display(
    df.agg(
        F.min("injury_date").alias("fecha_minima"),
        F.max("injury_date").alias("fecha_maxima")
    )
)

# COMMAND ----------

# DBTITLE 1,Duplicados Injury
#  Duplicados
print("\n12. DUPLICADOS EXACTOS")
print("-" * 100)

distinct_rows = df.distinct().count()
duplicated_rows = num_rows - distinct_rows

print(f"Filas totales: {num_rows:,}")
print(f"Filas distintas: {distinct_rows:,}")
print(f"Filas duplicadas exactas: {duplicated_rows:,}")

# COMMAND ----------

df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Header correlación Injury
# MAGIC %md
# MAGIC ### Estudio de correlación de variables

# COMMAND ----------

# DBTITLE 1,Matriz correlación Injury
# Vectorizamos columnas numéricas
assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features_corr",
    handleInvalid="keep"
)

df_vector_corr = assembler.transform(df.select(*numeric_cols))

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

display(df)

# COMMAND ----------

categorical_cols = [
    "affected_side",
    "injury_type",
    "affected_zone",
    "injury_mecanism",
    "injury_mecanism_action",
    "injury_situation",
    "injury_activity_type",
    "day_code"
]

for c in categorical_cols:
    print("=" * 80)
    print(c)
    display(
        df
        .groupBy(c)
        .agg(F.count("*").alias("n"))
        .orderBy(F.desc("n"))
    )

# COMMAND ----------

df_injuries_player_base = (
    df
    .groupBy(ID_COL)
    .agg(
        F.count("*").alias("injury_n_total"),
        F.sum("total_injury_days").alias("injury_days_sum"),
        F.avg("total_injury_days").alias("injury_days_mean"),
        F.max("total_injury_days").alias("injury_days_max"),
        F.stddev("total_injury_days").alias("injury_days_std"),
        F.avg("severity").alias("injury_severity_mean"),
        F.max("severity").alias("injury_severity_max")
    )
    .fillna({
        "injury_days_std": 0
    })
)

display(df_injuries_player_base)

# COMMAND ----------

from functools import reduce

# Columnas categóricas que sí agregaría
categorical_cols_for_agg = [
    "affected_side",
    "injury_type",
    "affected_zone",
    "injury_mecanism",
    "injury_situation",
    "injury_activity_type",
    "day_code"
]

TOP_N_CATEGORIES = 10

df_category_features_list = []

for c in categorical_cols_for_agg:
    
    # Top categorías más frecuentes
    top_values = [
        row[c]
        for row in (
            df
            .filter(F.col(c).isNotNull())
            .groupBy(c)
            .agg(F.count("*").alias("n"))
            .orderBy(F.desc("n"))
            .limit(TOP_N_CATEGORIES)
            .collect()
        )
    ]
    
    print(f"{c}: {top_values}")
    
    agg_exprs = []
    
    for value in top_values:
        safe_value = (
            str(value)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        
        col_name = f"injury_n_{c}_{safe_value}"
        
        agg_exprs.append(
            F.sum(
                F.when(F.col(c) == value, 1).otherwise(0)
            ).alias(col_name)
        )
    
    df_cat = (
        df
        .groupBy(ID_COL)
        .agg(*agg_exprs)
    )
    
    df_category_features_list.append(df_cat)

# Unir todas las features categóricas
df_injuries_category_features = reduce(
    lambda left, right: left.join(right, on=ID_COL, how="outer"),
    df_category_features_list
).fillna(0)

display(df_injuries_category_features)

# COMMAND ----------

df_injuries_player_agg = (
    df_injuries_player_base
    .join(df_injuries_category_features, on=ID_COL, how="left")
    .fillna(0)
)

# Crear porcentajes para las columnas de conteos categóricos
count_category_cols = [
    c for c in df_injuries_player_agg.columns
    if c.startswith("injury_n_")
    and c != "injury_n_total"
]

for c in count_category_cols:
    df_injuries_player_agg = df_injuries_player_agg.withColumn(
        c.replace("injury_n_", "injury_pct_"),
        F.when(
            F.col("injury_n_total") > 0,
            F.col(c) / F.col("injury_n_total")
        ).otherwise(F.lit(0.0))
    )

display(df_injuries_player_agg)

# COMMAND ----------

cols_excluir_injury_clustering = [
    ID_COL,
    "first_injury_date",
    "last_injury_date",
    "injury_reference_date"
]

features_injury_cluster = [
    c for c in df_injuries_player_agg.columns
    if c not in cols_excluir_injury_clustering
]

print("Variables de lesiones para clustering:", len(features_injury_cluster))

for c in features_injury_cluster:
    print(c)

# COMMAND ----------

df_injuries_player_agg.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "dev_gold.pitchvision.injury_agregado_por_jugador"
)

# COMMAND ----------


