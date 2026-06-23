# Databricks notebook source
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
from functools import reduce

# COMMAND ----------

# DBTITLE 1,Header CMJ
# MAGIC %md
# MAGIC # Tabla v_physical_test_cmj

# COMMAND ----------

# DBTITLE 1,Carga tabla y EDA básico CMJ
TABLE_NAME = "dev_silver.pitchvision.v_physical_test_cmj"  

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

# DBTITLE 1,Header nulos CMJ
# MAGIC %md
# MAGIC ### Conteo de nulos

# COMMAND ----------

# DBTITLE 1,Nulos por columna CMJ
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

# DBTITLE 1,Deportistas únicos CMJ
df.select("id_unico_deportista").distinct().count()

# COMMAND ----------

# DBTITLE 1,Percentiles CMJ
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

# DBTITLE 1,Header outliers CMJ
# MAGIC %md
# MAGIC ### Detección de outliers

# COMMAND ----------

# DBTITLE 1,Detección outliers percentiles CMJ
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

# DBTITLE 1,Severidad de extremos CMJ
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

# DBTITLE 1,Distribuciones extremos CMJ
# ============================================================
# PINTAR DISTRIBUCIONES DE VARIABLES CON max_exceso_normalizado > 5
# ============================================================

import matplotlib.pyplot as plt

THRESHOLD_EXCESO = 5
MAX_POINTS = 10000

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

# Winsorización con límites a 3*IQR de los percentiles 05 y 95

winsorized_df = df

for c in numeric_cols:
    try:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]
        iqr = p75 - p25

        lower_limit = p05 - 3 * iqr
        upper_limit = p95 + 3 * iqr

        winsorized_df = winsorized_df.withColumn(
            c,
            F.when(F.col(c) < lower_limit, lower_limit)
             .when(F.col(c) > upper_limit, upper_limit)
             .otherwise(F.col(c))
        )
    except Exception as e:
        print(f"No se pudo winsorizar columna {c}: {e}")

print("Winsorización aplicada con límites a 3*IQR de p05 y p95.")
display(winsorized_df.limit(10))

# COMMAND ----------

df= winsorized_df

# COMMAND ----------

# Imputar nulos numéricos con la mediana del propio jugador en otras fechas

from pyspark.sql import Window

ID_COL = "id_unico_deportista"
DATE_COL = "test_date"

# Crear ventanas por jugador
w_player = Window.partitionBy(ID_COL)

# Calcular la mediana por jugador para cada columna numérica
player_medians = df.select(
    ID_COL,
    *[
        F.expr(f"percentile_approx({c}, 0.5)").over(w_player).alias(f"{c}__player_median")
        for c in numeric_cols
    ]
)

# Unir las medianas al dataframe original
df_with_medians = df.join(player_medians, on=ID_COL, how="left")

# Imputar nulos con la mediana del jugador
imputed_exprs = [
    F.when(
        F.col(c).isNull(), F.col(f"{c}__player_median")
    ).otherwise(F.col(c)).alias(c)
    if c in numeric_cols else F.col(c)
    for c in df.columns
]

df_imputed = df_with_medians.select(*imputed_exprs)

# Contar nulos restantes por columna
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_imputed.columns
]

nulls_after_impute = df_imputed.agg(*null_exprs)

nulls_long = nulls_after_impute.select(
    F.explode(
        F.array(*[
            F.struct(
                F.lit(c).alias("columna"),
                F.col(c).alias("nulos")
            )
            for c in df_imputed.columns
        ])
    ).alias("x")
).select("x.*")

nulls_report_after = (
    nulls_long
    .withColumn("filas_totales", F.lit(df_imputed.count()))
    .withColumn(
        "porcentaje_nulos",
        F.round((F.col("nulos") / F.col("filas_totales")) * 100, 2)
    )
    .orderBy(F.col("porcentaje_nulos").desc())
)

display(nulls_report_after)

# COMMAND ----------

# Imputar los nulos restantes con la mediana global de cada columna numérica

remaining_nulls = [
    row["columna"]
    for row in nulls_report_after.filter(F.col("nulos") > 0).collect()
]

global_medians = {}
for c in remaining_nulls:
    try:
        median = df_imputed.approxQuantile(c, [0.5], 0.01)[0]
        global_medians[c] = median
    except Exception:
        global_medians[c] = None

df_final = df_imputed.fillna(global_medians)

# Verificar nulos restantes
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_final.columns
]
nulls_final = df_final.agg(*null_exprs)
nulls_long_final = nulls_final.select(
    F.explode(
        F.array(*[
            F.struct(
                F.lit(c).alias("columna"),
                F.col(c).alias("nulos")
            )
            for c in df_final.columns
        ])
    ).alias("x")
).select("x.*")
nulls_report_final = (
    nulls_long_final
    .withColumn("filas_totales", F.lit(df_final.count()))
    .withColumn(
        "porcentaje_nulos",
        F.round((F.col("nulos") / F.col("filas_totales")) * 100, 2)
    )
    .orderBy(F.col("porcentaje_nulos").desc())
)

display(nulls_report_final)

# COMMAND ----------

# DBTITLE 1,Header correlación CMJ
# MAGIC %md
# MAGIC ### Estudio de correlación de variables

# COMMAND ----------

df_cleaned=df_final

# COMMAND ----------

# DBTITLE 1,Matriz correlación CMJ
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

# DBTITLE 1,Pares alta correlación CMJ
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

if not high_corr_pd.empty:
    high_corr_pd = high_corr_pd.sort_values("correlacion_abs", ascending=False)
    display(high_corr_pd)
else:
    print(f"No hay pares de variables con correlaci\u00f3n >= {CORR_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminamos columnas best frente a avg

# COMMAND ----------

# VARIABLES ELIMINADAS POR ALTA CORRELACIÓN

cols_to_drop_corr_pairs = [
    "concentric_impulse_ns_best",
    "peak_power_bm_w_kg_best",
    "concentric_mean_power_bm_w_kg_best",
    "rsi_modified_imp_mom_best",
    "jump_height_imp_mom_cm_best",
    "concentric_peak_force_bm_n_kg_best"
]

df_reduced = df_cleaned.drop(*cols_to_drop_corr_pairs)

print(f"Columnas eliminadas: {cols_to_drop_corr_pairs}")
print(f"Columnas restantes: {len(df_reduced.columns)}")

# COMMAND ----------

ultimas_temporadas = ["24-25", "25-26"]

total_jugadores = df_cleaned.select("id_unico_deportista").distinct().count()
jugadores_ultimas = (
    df_cleaned
    .filter(F.col("season").isin(ultimas_temporadas))
    .select("id_unico_deportista")
    .distinct()
    .count()
)
porcentaje_ultimas = round((jugadores_ultimas / total_jugadores) * 100, 2)
print(f"Porcentaje de jugadores con al menos una medición en las temporadas {ultimas_temporadas}: {porcentaje_ultimas}%")

# COMMAND ----------

# Conteo de jugadores por temporada y número de sesiones
from pyspark.sql import functions as F

sesiones_por_jugador = (
    df_cleaned.groupBy("id_unico_deportista", "season")
    .count()
    .withColumnRenamed("count", "num_sesiones")
)

# Filtrar solo la temporada 25-26
sesiones_ultima_temp = sesiones_por_jugador.filter(F.col("season") == "25-26")

resumen = (
    sesiones_ultima_temp.groupBy("season", "num_sesiones")
    .agg(F.count("id_unico_deportista").alias("num_jugadores"))
    .orderBy("season", "num_sesiones")
)

display(resumen)

# COMMAND ----------

df_cleaned = df_reduced

# COMMAND ----------

# Recalcular numeric_cols desde df_cleaned actual (sin columnas eliminadas por correlación)
numeric_cols = [
    field.name for field in df_cleaned.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in ["id_unico_deportista", "n_trials", "is_restricted"]
]

# Crear DataFrame base con sesiones por jugador y temporada
sesiones_por_jugador = (
    df_cleaned.groupBy("id_unico_deportista", "season")
    .count()
    .withColumnRenamed("count", "num_sesiones")
)

# 1. Jugadores con >=5 sesiones en 25-26: media de 25-26
jugadores_25_26 = (
    sesiones_por_jugador
    .filter((F.col("season") == "25-26") & (F.col("num_sesiones") >= 5))
    .select("id_unico_deportista")
    .distinct()
)

df_25_26 = (
    df_cleaned
    .filter(F.col("season") == "25-26")
    .join(jugadores_25_26, on="id_unico_deportista", how="inner")
    .groupBy("id_unico_deportista")
    .agg(*[F.mean(c).alias(c) for c in numeric_cols])
    .withColumn("temporada_fuente", F.lit("25-26"))
)

# 2. Jugadores con <5 sesiones en 25-26 pero >=5 en 24-25+25-26: media de ambas
jugadores_restantes = (
    df_cleaned.select("id_unico_deportista").distinct()
    .join(jugadores_25_26, on="id_unico_deportista", how="left_anti")
)

sesiones_ultimas = (
    sesiones_por_jugador
    .filter(F.col("season").isin(["24-25", "25-26"]))
    .groupBy("id_unico_deportista")
    .agg(F.sum("num_sesiones").alias("total_sesiones_ultimas"))
    .filter(F.col("total_sesiones_ultimas") >= 5)
)

jugadores_ultimas = (
    jugadores_restantes
    .join(sesiones_ultimas, on="id_unico_deportista", how="inner")
    .select("id_unico_deportista")
)

df_ultimas = (
    df_cleaned
    .filter(F.col("season").isin(["24-25", "25-26"]))
    .join(jugadores_ultimas, on="id_unico_deportista", how="inner")
    .groupBy("id_unico_deportista")
    .agg(*[F.mean(c).alias(c) for c in numeric_cols])
    .withColumn("temporada_fuente", F.lit("24-25+25-26"))
)

# 3. Jugadores restantes: media de todas sus mediciones (aunque no sean 5)
jugadores_sin_suficientes = (
    df_cleaned.select("id_unico_deportista").distinct()
    .join(jugadores_25_26, on="id_unico_deportista", how="left_anti")
    .join(jugadores_ultimas, on="id_unico_deportista", how="left_anti")
)

df_antiguas = (
    df_cleaned
    .join(jugadores_sin_suficientes, on="id_unico_deportista", how="inner")
    .groupBy("id_unico_deportista")
    .agg(*[F.mean(c).alias(c) for c in numeric_cols])
    .withColumn("temporada_fuente", F.lit("antiguas"))
)

# Unir todos los casos
df_final_para_Clustering = (
    df_25_26
    .unionByName(df_ultimas)
    .unionByName(df_antiguas)
)

display(df_final_para_Clustering)

# COMMAND ----------

# Comprobar nulos en df_final_para_Clustering
from pyspark.sql.functions import col, sum as sum_

null_counts = (
    df_final_para_Clustering
    .select([
        sum_(col(c).isNull().cast("int")).alias(c)
        for c in df_final_para_Clustering.columns
    ])
)

display(null_counts)

# COMMAND ----------

df_final_para_Clustering.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.cmj_agregado_por_jugador"
)

