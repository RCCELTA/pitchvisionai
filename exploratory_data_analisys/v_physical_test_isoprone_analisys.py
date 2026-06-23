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

# DBTITLE 1,Header Isoprone
# MAGIC %md
# MAGIC ## # Tabla v_physical_test_isoprone

# COMMAND ----------

# DBTITLE 1,Carga tabla y EDA básico Isoprone
TABLE_NAME = "dev_silver.pitchvision.v_physical_test_isoprone"  

df = spark.table(TABLE_NAME)

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

# DBTITLE 1,Header nulos Isoprone
# MAGIC %md
# MAGIC ### Conteo de nulos

# COMMAND ----------

# DBTITLE 1,Nulos por columna Isoprone
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

from pyspark.sql import functions as F

cols_con_nulos = [
    "left_max_force_normalized_length_bw",
    "right_max_force_normalized_length_bw",
    "left_max_force_normalized_bw",
    "right_max_force_normalized_bw",
    "torque_asym_pct",
    "max_force_asym_pct",
    "avg_force_asym_pct",
    "impulse_asym_pct"
]

id_col = "id_unico_deportista"

resumen_variables = None

for col_name in cols_con_nulos:
    
    resumen_jugador = (
        df
        .groupBy(id_col)
        .agg(
            F.count("*").alias("n_tests"),
            F.sum(F.col(col_name).isNotNull().cast("int")).alias("n_medidas"),
            F.sum(F.col(col_name).isNull().cast("int")).alias("n_nulos")
        )
        .withColumn(
            "estado",
            F.when(F.col("n_medidas") == 0, "todas_nulas")
             .when(F.col("n_nulos") == 0, "todas_medidas")
             .otherwise("mixto")
        )
    )
    
    resumen_estado = (
        resumen_jugador
        .groupBy("estado")
        .agg(
            F.count("*").alias("n_jugadores"),
            F.round(F.avg("n_tests"), 2).alias("avg_tests"),
            F.round(F.avg("n_medidas"), 2).alias("avg_medidas"),
            F.min("n_medidas").alias("min_medidas"),
            F.max("n_medidas").alias("max_medidas")
        )
        .withColumn("variable", F.lit(col_name))
        .select(
            "variable",
            "estado",
            "n_jugadores",
            "avg_tests",
            "avg_medidas",
            "min_medidas",
            "max_medidas"
        )
    )
    
    resumen_variables = (
        resumen_estado
        if resumen_variables is None
        else resumen_variables.unionByName(resumen_estado)
    )

display(resumen_variables.orderBy("variable", "estado"))

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que para las variables left_max_force_normalized_bw y right_max_force_normalized_bw hay 150 jugadores sin ninguna medida de un total de 332 jugadores y para right y left max_force_normalized_length_bw 283 de 332 por lo que procedemos a eliminar estas dos columnas. El resto de variables con nulos no merece la pena eliminarlas ya que se podrán imputar practicamente todas con registros del propio jugador.

# COMMAND ----------

# MAGIC %md
# MAGIC Imputamos los valores faltantes usando la mediana del propio jugador; cuando un jugador no tenía ningún valor medido para una variable, usamos como respaldo la mediana global. Así mantenemos la información útil sin introducir un sesgo excesivo.

# COMMAND ----------

# 1. Columnas BW a eliminar
cols_bw_drop = [
    "left_max_force_normalized_bw",
    "right_max_force_normalized_bw",
    "left_max_force_normalized_length_bw",
    "right_max_force_normalized_length_bw"
]

df = df.drop(*cols_bw_drop)

# 2. Variables con pocos nulos que queremos imputar
cols_imputar = [
    "torque_asym_pct",
    "max_force_asym_pct",
    "avg_force_asym_pct",
    "impulse_asym_pct"
]

id_col = "id_unico_deportista"

# 3. Crear flags de nulo original
for c in cols_imputar:
    df = df.withColumn(
        f"{c}_missing",
        F.col(c).isNull().cast("int")
    )

# 4. Imputar con mediana del propio jugador
for c in cols_imputar:
    
    # Mediana global como fallback
    mediana_global = df.approxQuantile(c, [0.5], 0.01)[0]
    
    # Mediana por jugador
    mediana_jugador = (
        df
        .groupBy(id_col)
        .agg(
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_median_player")
        )
    )
    
    df = df.join(
        mediana_jugador,
        on=id_col,
        how="left"
    )
    
    # Primero mediana del jugador; si no existe, mediana global
    df = df.withColumn(
        c,
        F.when(
            F.col(c).isNull() & F.col(f"{c}_median_player").isNotNull(),
            F.col(f"{c}_median_player")
        )
        .when(
            F.col(c).isNull(),
            F.lit(mediana_global)
        )
        .otherwise(F.col(c))
    )
    
    df = df.drop(f"{c}_median_player")

# 5. Comprobación de nulos restantes
display(
    df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in cols_imputar
    ])
)

# COMMAND ----------

# DBTITLE 1,Deportistas únicos Isoprone
df.select("id_unico_deportista").distinct().count()

# COMMAND ----------

# DBTITLE 1,Percentiles Isoprone
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

# DBTITLE 1,Header outliers Isoprone
# MAGIC %md
# MAGIC ### Detección de outliers

# COMMAND ----------

# DBTITLE 1,Detección outliers percentiles Isoprone
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

# DBTITLE 1,Severidad de extremos Isoprone
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

# DBTITLE 1,Distribuciones extremos Isoprone
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

# WINSORIZACIÓN CONSERVADORA

ID_COL = "id_unico_deportista"

IQR_FACTOR = 3.0

exclude_cols = [
    ID_COL,
    "test_date",
    "season",
    "is_restricted"
]

numeric_cols_to_winsorize = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

bound_exprs = []

for c in numeric_cols_to_winsorize:
    bound_exprs.append(
        F.expr(f"percentile_approx(`{c}`, 0.25)").alias(f"{c}__q1")
    )
    bound_exprs.append(
        F.expr(f"percentile_approx(`{c}`, 0.75)").alias(f"{c}__q3")
    )

bounds_df = df.agg(*bound_exprs)

df_winsor = df.crossJoin(bounds_df)

select_exprs = []

for col_name in df.columns:

    if col_name in numeric_cols_to_winsorize:

        q1_col = f"{col_name}__q1"
        q3_col = f"{col_name}__q3"

        iqr_expr = F.col(q3_col) - F.col(q1_col)

        lower_expr = F.col(q1_col) - F.lit(IQR_FACTOR) * iqr_expr
        upper_expr = F.col(q3_col) + F.lit(IQR_FACTOR) * iqr_expr

        winsorized = (
            F.when(
                F.col(col_name) < lower_expr,
                lower_expr
            )
            .when(
                F.col(col_name) > upper_expr,
                upper_expr
            )
            .otherwise(F.col(col_name))
        )

        select_exprs.append(winsorized.alias(col_name))

    else:
        select_exprs.append(F.col(col_name))

df_winsor = df_winsor.select(*select_exprs)

print("Winsorización aplicada.")
print(f"Filas: {df_winsor.count():,}")

# COMMAND ----------

df=df_winsor

# COMMAND ----------

# DBTITLE 1,Header correlación Isoprone
# MAGIC %md
# MAGIC ### Estudio de correlación de variables

# COMMAND ----------

# DBTITLE 1,Matriz correlación Isoprone
numeric_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
]

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

# DBTITLE 1,Pares alta correlación Isoprone
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

# VARIABLES ELIMINADAS POR ALTA CORRELACIÓN

cols_to_drop_corr_pairs = [
    "torque_asym_pct"
]

df_reduced = df.drop(*cols_to_drop_corr_pairs)

print(f"Columnas eliminadas: {cols_to_drop_corr_pairs}")
print(f"Columnas restantes: {len(df_reduced.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminamos también las columnas que solo tienen 0s

# COMMAND ----------

cols_to_drop_corr_pairs = [
    "mdc_left_max_force_n",
    "mdc_left_avg_force_n",
    "mdc_left_impulse_ns",
    "mdc_left_torque_nm",
    "mdc_right_max_force_n",
    "mdc_right_avg_force_n",
    "mdc_right_impulse_ns",
    "mdc_right_torque_nm"
]

df_reduced = df_reduced.drop(*cols_to_drop_corr_pairs)

print(f"Columnas eliminadas: {cols_to_drop_corr_pairs}")
print(f"Columnas restantes: {len(df_reduced.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Agregado por jugador.

# COMMAND ----------

df_base = df_reduced

# Elegimos directamente las dos últimas temporadas
temporadas_usar = ["24-25", "25-26"]

df_base_ultimas_2 = (
    df_base
    .filter(F.col("season").isin(temporadas_usar))
)

# Comprobación de volumen por temporada
display(
    df_base_ultimas_2
    .groupBy("season")
    .agg(
        F.count("*").alias("n_tests"),
        F.countDistinct(ID_COL).alias("n_jugadores")
    )
    .orderBy("season")
)

# Columnas a excluir de la agregación física
exclude_cols = [
    ID_COL,
    "test_id",
    "test_date",
    "test_hour_utc",
    "device",
    "season",
    "is_restricted"
]

# Variables numéricas candidatas
numeric_cols = [
    field.name
    for field in df_base_ultimas_2.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

# Flags de missing
missing_flag_cols = [
    c for c in numeric_cols
    if c.endswith("_missing")
]

# Variables físicas reales
physical_cols = [
    c for c in numeric_cols
    if c not in missing_flag_cols
]

# Agregaciones de control
agg_exprs = [
    F.count("*").alias("n_tests_isoprone"),
    F.countDistinct("season").alias("n_seasons_isoprone"),
    F.min("test_date").alias("first_isoprone_test_date"),
    F.max("test_date").alias("last_isoprone_test_date")
]

# Media y P90 de cada variable física
for c in physical_cols:
    agg_exprs.extend([
        F.avg(c).alias(f"{c}_mean"),
        F.expr(f"percentile_approx(`{c}`, 0.90)").alias(f"{c}_p90")
    ])

# Porcentaje de imputación para flags
for c in missing_flag_cols:
    agg_exprs.append(
        F.avg(c).alias(f"pct_{c}")
    )

df_isoprone_player = (
    df_base_ultimas_2
    .groupBy(ID_COL)
    .agg(*agg_exprs)
)

display(df_isoprone_player)

# COMMAND ----------

display(
    df_isoprone_player
    .groupBy("n_tests_isoprone")
    .agg(F.count("*").alias("n_jugadores"))
    .orderBy("n_tests_isoprone")
)

# COMMAND ----------

null_check = df_isoprone_player.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_isoprone_player.columns
])

display(null_check)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS dev_gold.pitchvision.isoprone_agregado_por_jugador;

# COMMAND ----------

df_isoprone_player.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.isoprone_agregado_por_jugador"
)
