# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
import sys
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
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

TABLE_NAME = "dev_silver.pitchvision.v_competitive_profile"  

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


display(df.select("activity_team_id").distinct())

# COMMAND ----------

from pyspark.sql.functions import mode

# Obtener el valor más frecuente de activity_team_id
most_frequent_teamid = df.agg(mode("activity_team_id", True).alias("mode_teamid")).collect()[0]["mode_teamid"]

# Llenar los nulos con el valor más frecuente
df_filled = df.fillna({"activity_team_id": most_frequent_teamid})

display(df_filled)

# COMMAND ----------

df=df_filled

# COMMAND ----------

df.select("id_unico_deportista").distinct().count()

# COMMAND ----------

#  Duplicados
print("\n12. DUPLICADOS EXACTOS")
print("-" * 100)

distinct_rows = df.distinct().count()
duplicated_rows = num_rows - distinct_rows

print(f"Filas totales: {num_rows:,}")
print(f"Filas distintas: {distinct_rows:,}")
print(f"Filas duplicadas exactas: {duplicated_rows:,}")

# COMMAND ----------

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

# PINTAR DISTRIBUCIONES DE VARIABLES CON max_exceso_normalizado > 5

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
# MAGIC ### Correlación de variables

# COMMAND ----------

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
    print(f"No hay pares de variables con correlación >= {CORR_THRESHOLD}")

# COMMAND ----------

# Eliminar columnas altamente correlacionadas (completar tras análisis)
cols_to_drop_corr_pairs = [
    "mean_acc_count_YTD",
    "mean_dcc_count_YTD",
    "mean_dcc_count_per_min_YTD"
    "mean_acc_count_per_min_YTD"
]

if cols_to_drop_corr_pairs:
    df_reduced = df.drop(*cols_to_drop_corr_pairs)
    print(f"Columnas eliminadas: {cols_to_drop_corr_pairs}")
    print(f"Columnas restantes: {len(df_reduced.columns)}")
else:
    df_reduced = df
    print("No se eliminaron columnas por correlación.")
    print(f"Columnas: {len(df_reduced.columns)}")

# COMMAND ----------

df=df_reduced

# COMMAND ----------

ID_COL = "id_unico_deportista"
SEASON_COL = "season"

df_base = df

MIN_TESTS_25_26 = 5
SEASON_RECENT = "25-26"
SEASONS_RECENT_2 = ["24-25", "25-26"]

jugadores_total = (
    df_base
    .select(ID_COL)
    .distinct()
)

tests_25_26 = (
    df_base
    .filter(F.col(SEASON_COL) == SEASON_RECENT)
    .groupBy(ID_COL)
    .agg(F.count("*").alias("n_tests_25_26"))
)

tests_24_25_25_26 = (
    df_base
    .filter(F.col(SEASON_COL).isin(SEASONS_RECENT_2))
    .groupBy(ID_COL)
    .agg(F.count("*").alias("n_tests_24_25_25_26"))
)

criterio_jugador = (
    jugadores_total
    .join(tests_25_26, on=ID_COL, how="left")
    .join(tests_24_25_25_26, on=ID_COL, how="left")
    .fillna({
        "n_tests_25_26": 0,
        "n_tests_24_25_25_26": 0
    })
    .withColumn(
        "temporada_fuente",
        F.when(
            F.col("n_tests_25_26") >= MIN_TESTS_25_26,
            F.lit("solo_25_26")
        )
        .when(
            F.col("n_tests_24_25_25_26") > 0,
            F.lit("24_25_y_25_26")
        )
        .otherwise(F.lit("todas_disponibles"))
    )
)

display(
    criterio_jugador
    .groupBy("temporada_fuente")
    .agg(
        F.count("*").alias("n_jugadores"),
        F.avg("n_tests_25_26").alias("avg_tests_25_26"),
        F.avg("n_tests_24_25_25_26").alias("avg_tests_24_25_25_26")
    )
)

# COMMAND ----------

df_con_criterio = (
    df_base
    .join(
        criterio_jugador.select(ID_COL, "temporada_fuente"),
        on=ID_COL,
        how="inner"
    )
)

df_tests_para_agregar = (
    df_con_criterio
    .filter(
        (
            (F.col("temporada_fuente") == "solo_25_26") &
            (F.col(SEASON_COL) == "25-26")
        )
        |
        (
            (F.col("temporada_fuente") == "24_25_y_25_26") &
            (F.col(SEASON_COL).isin(["24-25", "25-26"]))
        )
        |
        (
            F.col("temporada_fuente") == "todas_disponibles"
        )
    )
)

display(
    df_tests_para_agregar
    .groupBy("temporada_fuente")
    .agg(
        F.count("*").alias("n_tests"),
        F.countDistinct(ID_COL).alias("n_jugadores")
    )
)

# COMMAND ----------

display(df_tests_para_agregar)

# COMMAND ----------

# DBTITLE 1,Preprocesar match_time_range
# Preprocesamiento de match_time_range → midpoint numérico (minuto representativo)

midpoint_mapping = {
    "-45": 22.5,    # 0-45 min (primera parte)
    "-50": 25.0,    # 0-50 min (primera parte extendida)
    "45-65": 55.0,  # 45-65 min (inicio segunda parte)
    "50-85": 67.5,  # 50-85 min (grueso segunda parte)
    "+65": 77.5,    # 65+ min (fase final)
    "+85": 90.0,    # 85+ min (tramo final/descuento)
}

midpoint_expr = F.coalesce(*[
    F.when(F.col("match_time_range") == k, F.lit(v))
    for k, v in midpoint_mapping.items()
])

df_tests_para_agregar = (
    df_tests_para_agregar
    .withColumn("match_time_range", midpoint_expr)
)

# Verificar distribución
print("Distribución de match_time_range (ahora numérico, midpoint):")
display(
    df_tests_para_agregar
    .groupBy("match_time_range")
    .agg(F.count("*").alias("n_registros"))
    .orderBy("match_time_range")
)

# COMMAND ----------

# DBTITLE 1,Agregación por jugador
cols_excluir_agregacion = [
    ID_COL,
    "activity_date",
    "activity_team_id",
    "test_id",
    "test_hour_utc",
    "device",
    SEASON_COL,
    "temporada_fuente",
    "position_id",         # categórica (ID posición) — se agrega aparte como moda
    "is_last_profile",     # flag de control, no métrica física
]

# Columnas a excluir del clustering (post-agregación)
cols_excluir_clustering = [
    "n_tests_agregados",
    "n_temporadas_agregadas",
    "temporada_fuente",
    "first_test_date",
    "last_test_date",
    "position_id_mode",           # se usa como label, no como feature
    "played_matches_YTD_mean",    # volumen de muestra, no perfil físico
    "used_matches_YTD_mean",      # volumen de muestra, no perfil físico
]

numeric_cols = [
    field.name
    for field in df_tests_para_agregar.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_agregacion
    and not field.name.endswith("_missing")
]

missing_flag_cols = [
    field.name
    for field in df_tests_para_agregar.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name.endswith("_missing")
]

agg_exprs = [
    F.count("*").alias("n_tests_agregados"),
    F.countDistinct(SEASON_COL).alias("n_temporadas_agregadas"),
    F.first("temporada_fuente").alias("temporada_fuente"),
    F.min("activity_date").alias("first_test_date"),
    F.max("activity_date").alias("last_test_date"),
    F.mode("position_id").alias("position_id_mode"),  # posición más frecuente (label)
]

# Media de variables numéricas
for c in numeric_cols:
    agg_exprs.append(
        F.avg(c).alias(f"{c}_mean")
    )

# Flags de missing como porcentaje
for c in missing_flag_cols:
    agg_exprs.append(
        F.avg(c).alias(f"pct_{c}")
    )

df_player_agg_full = (
    df_tests_para_agregar
    .groupBy(ID_COL)
    .agg(*agg_exprs)
)

# Eliminar columnas no relevantes para clustering
cols_to_drop = [c for c in cols_excluir_clustering if c in df_player_agg_full.columns]
df_player_agg = df_player_agg_full.drop(*cols_to_drop)

print(f"Columnas eliminadas para clustering: {cols_to_drop}")
print(f"Columnas finales en df listo para gold: {len(df_player_agg.columns)}")
print(f"Columnas: {df_player_agg.columns}")
display(df_player_agg)

# COMMAND ----------

df_player_agg.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.competitive_profile_agregado_por_jugador")
