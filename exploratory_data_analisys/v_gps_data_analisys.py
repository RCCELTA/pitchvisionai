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
import numpy as np

# COMMAND ----------

# DBTITLE 1,Carga tabla y EDA básico GPS
TABLE_NAME = "dev_silver.pitchvision.v_gps_data"  

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

# DBTITLE 1,Header nulos GPS
# MAGIC %md
# MAGIC ### Conteo de nulos

# COMMAND ----------

# DBTITLE 1,Nulos por columna GPS
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
# MAGIC Eliminar las filas en las que falta el id del jugador??
# MAGIC que hacemos con las variables con tantos nulos?

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminamos las filas sin identificador de jugador:

# COMMAND ----------

ID_COL = "id_unico_deportista"
df = df.filter(F.col(ID_COL).isNotNull())
df.count()

# COMMAND ----------

cols_missing = [
    "sprint_high_intensity_count",
    "sprint_high_intensity_distance",
    "player_load"
]

for c in cols_missing:
    print(f"\nNulos de {c} por season")

    display(
        df
        .groupBy("season")
        .agg(
            F.count("*").alias("filas"),
            F.sum(F.col(c).isNull().cast("int")).alias("nulos"),
            F.round(F.sum(F.col(c).isNull().cast("int")) / F.count("*") * 100, 2).alias("pct_nulos")
        )
        .orderBy("season")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que claramente esas variables no se estaban midiendo las dos primeras temporadas.

# COMMAND ----------

# MAGIC %md
# MAGIC Añadimos columnas con flag de disponibilidad o no de la variable

# COMMAND ----------

total_players = df.select(ID_COL).distinct().count()

coverage_rows = []

for c in cols_missing:
    player_coverage = (
        df
        .groupBy(ID_COL)
        .agg(
            F.sum(F.col(c).isNotNull().cast("int")).alias("num_mediciones_validas")
        )
    )

    players_with_at_least_one = (
        player_coverage
        .filter(F.col("num_mediciones_validas") > 0)
        .count()
    )

    players_without_any = (
        player_coverage
        .filter(F.col("num_mediciones_validas") == 0)
        .count()
    )

    coverage_rows.append((
        c,
        total_players,
        players_with_at_least_one,
        players_without_any,
        round(players_with_at_least_one / total_players * 100, 2),
        round(players_without_any / total_players * 100, 2)
    ))

coverage_players_df = spark.createDataFrame(
    coverage_rows,
    [
        "columna",
        "jugadores_totales",
        "jugadores_con_al_menos_una_medicion",
        "jugadores_sin_ninguna_medicion",
        "pct_jugadores_con_medicion",
        "pct_jugadores_sin_medicion"
    ]
)

display(coverage_players_df)

# COMMAND ----------

# DBTITLE 1,Percentiles GPS
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

# DBTITLE 1,Header outliers GPS
# MAGIC %md
# MAGIC ### Detección de outliers

# COMMAND ----------

# DBTITLE 1,Detección outliers percentiles GPS
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

# DBTITLE 1,Severidad de extremos GPS
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

# DBTITLE 1,Distribuciones extremos GPS
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

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(plot_df[c], bins=50, alpha=0.8)
        ax.axvline(p05, linestyle="--", linewidth=2, label=f"p05 = {p05:.2f}")
        ax.axvline(p95, linestyle="--", linewidth=2, label=f"p95 = {p95:.2f}")

        ax.set_xlabel(c)
        ax.set_ylabel("Frecuencia")
        ax.set_title(
            f"Distribuci\u00f3n de {c}\n"
            f"max_exceso_normalizado \u2248 {max_exceso:.2f} IQR | "
            f"Exceso inferior: {lower_excess:.2f} IQR | "
            f"Exceso superior: {upper_excess:.2f} IQR"
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC No hay ningún caso extremo y desde cliente aseguran que los métodos de medición están estandarizados y es poco probable que haya errores. Por lo tanto no se procede con ninguna estrategia de valores atípicos. Para no eliminar posibles jugadores extraordinarios o que destaquen mucho en alguna sesión.

# COMMAND ----------

for c in top_extreme_cols:
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

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=plot_df[c], ax=ax)
    ax.set_xlabel(c)
    ax.set_title(f"Boxplot de {c}")
    fig.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Rango fechas GPS
print("\n11. RANGO DE COLUMNA FECHA: activity_date")

display(
    df.agg(
        F.min("activity_date").alias("fecha_minima"),
        F.max("activity_date").alias("fecha_maxima")
    )
)

# COMMAND ----------

# DBTITLE 1,Duplicados GPS
#  Duplicados
print("\n12. DUPLICADOS EXACTOS")
print("-" * 100)

distinct_rows = df.distinct().count()
duplicated_rows = num_rows - distinct_rows

print(f"Filas totales: {num_rows:,}")
print(f"Filas distintas: {distinct_rows:,}")
print(f"Filas duplicadas exactas: {duplicated_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminamos duplicados

# COMMAND ----------

df = df.distinct()

# COMMAND ----------

df.count()

# COMMAND ----------

# DBTITLE 1,Header correlación GPS
# MAGIC %md
# MAGIC ### Estudio de correlación de variables

# COMMAND ----------

# DBTITLE 1,Nota imputación GPS
# MAGIC %md
# MAGIC Para estudiar la correlación de las columnas con nulos utilizamos solo filas donde exista el dato, ya que rellenar los nulos con 0 o mediana no es una buena aproximación aquí.

# COMMAND ----------

# DBTITLE 1,Matriz correlación GPS
# Columnas con muchos nulos
cols_missing = [
    "sprint_high_intensity_count",
    "sprint_high_intensity_distance",
    "player_load"
]

# Columnas numéricas a correlacionar
cols_corr = numeric_cols

# Inicializamos matrices vacías
corr_pd = pd.DataFrame(
    np.nan,
    index=cols_corr,
    columns=cols_corr
)

n_valid_pd = pd.DataFrame(
    0,
    index=cols_corr,
    columns=cols_corr
)

# Calculamos correlación pairwise:
# para cada par de columnas, usamos solo filas donde ambas tienen dato
for i, c1 in enumerate(cols_corr):
    for j, c2 in enumerate(cols_corr):
        
        if j < i:
            # matriz simétrica: copiamos el resultado ya calculado
            corr_pd.loc[c1, c2] = corr_pd.loc[c2, c1]
            n_valid_pd.loc[c1, c2] = n_valid_pd.loc[c2, c1]
            continue
        
        if c1 == c2:
            corr_pd.loc[c1, c2] = 1.0
            n_valid_pd.loc[c1, c2] = df.filter(F.col(c1).isNotNull()).count()
            continue
        
        df_pair = (
            df
            .select(c1, c2)
            .filter(
                F.col(c1).isNotNull() &
                F.col(c2).isNotNull()
            )
        )
        
        n_valid = df_pair.count()
        n_valid_pd.loc[c1, c2] = n_valid
        
        # Evitamos correlaciones con muy pocos datos
        if n_valid >= 30:
            corr_value = df_pair.corr(c1, c2)
        else:
            corr_value = None
        
        corr_pd.loc[c1, c2] = corr_value

display(corr_pd)

# COMMAND ----------

display(n_valid_pd)

# COMMAND ----------

# DBTITLE 1,Pares alta correlación GPS
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
# MAGIC Teniendo en cuenta que las columnas con nulos son:
# MAGIC cols_missing = [
# MAGIC     "sprint_high_intensity_count",
# MAGIC     "sprint_high_intensity_distance",
# MAGIC     "player_load"
# MAGIC ]
# MAGIC Aquí tendría más sentido eliminar:
# MAGIC
# MAGIC | Pareja correlacionada | Mantendría | Eliminaría | Motivo |
# MAGIC |---|---|---|---|
# MAGIC | `total_distance` / `player_load` | `total_distance` | `player_load` | Correlación muy alta y `player_load` tiene muchos nulos. `total_distance` es más interpretable, estable y con mejor cobertura. |
# MAGIC | `acc_dcc_count` / `acc_count` | `acc_dcc_count` | `acc_count` | `acc_dcc_count` resume aceleraciones y deceleraciones, por lo que representa mejor la carga mecánica total. |
# MAGIC | `acc_dcc_count` / `dcc_count` | `acc_dcc_count` | `dcc_count` | Se mantiene la variable combinada porque captura el total de acciones ACC/DCC y evita duplicar información. |
# MAGIC | `sprint_high_intensity_count` / `sprint_high_intensity_distance` | `sprint_high_intensity_distance` | `sprint_high_intensity_count` | Ambas tienen los mismos nulos y están muy correlacionadas. La distancia de alta intensidad suele aportar más información que el simple conteo. |

# COMMAND ----------

TARGET_COL = "sprint_high_intensity_distance"

corr_target_pd = (
    corr_pd[TARGET_COL]
    .drop(labels=[TARGET_COL])
    .dropna()
    .abs()
    .sort_values(ascending=False)
)

display(corr_target_pd.reset_index().rename(
    columns={
        "index": "variable_comparada",
        TARGET_COL: "correlacion_abs"
    }
))

# COMMAND ----------

# MAGIC %md
# MAGIC Comprobamos sí la variable sprint_High_intensity_distance tiene alta correlación con alguna otra variable para ver si se podría eliminar y ahorrarnos el problema de los nulos. Vemos que solo con sprint_high_intensity_count que ya la vamos a eliminar.

# COMMAND ----------

cols_to_drop_corr = [
    "player_load",
    "acc_count",
    "dcc_count",
    "sprint_high_intensity_count"
]

df_reduced = df.drop(*cols_to_drop_corr)

# COMMAND ----------

df=df_reduced

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

df = df.withColumn(
    "sprint_high_intensity_distance_isnull",
    F.col("sprint_high_intensity_distance").isNull().cast("int")
)

# COMMAND ----------

display(df.limit(1))

# COMMAND ----------

# Variable objetivo a imputar
target = "sprint_high_intensity_distance"

# Variables completas/correlacionadas que quieres usar para definir registros similares
vars_similares = [
    "sprint_count",
    "sprint_distance"  # cambia o elimina esta si es la misma que target
]

# Seguridad: no usar el target como variable auxiliar
vars_similares = [v for v in vars_similares if v != target]

# Guardar copia original si no existe
if f"{target}_original" not in df.columns:
    df = df.withColumn(f"{target}_original", F.col(target))

# Indicador de nulo original si no existe
missing_col = f"{target}_missing"
if missing_col not in df.columns:
    df = df.withColumn(missing_col, F.col(target).isNull().cast("int"))

print("Nulos antes:", df.filter(F.col(target).isNull()).count())

# Crear bins por deciles para variables similares usando ntile
q = 10
bin_cols = []

for col_name in vars_similares:
    bin_col = f"{col_name}_bin"
    # ntile over non-null values, null stays null
    w = Window.orderBy(F.col(col_name))
    df = df.withColumn(
        bin_col,
        F.when(F.col(col_name).isNotNull(), F.ntile(q).over(w))
    )
    bin_cols.append(bin_col)

# Tamaño mínimo de grupo para fiarse de la mediana
min_group_size = 20

# Mediana y tamaño por grupo (solo filas con target no nulo)
group_stats = (
    df
    .filter(F.col(target).isNotNull())
    .groupBy(bin_cols)
    .agg(
        F.expr(f"percentile_approx({target}, 0.5)").alias("group_median"),
        F.count("*").alias("group_count")
    )
)

# Unir estadísticas al df
df = df.join(F.broadcast(group_stats), on=bin_cols, how="left")

# Imputar solo cuando el grupo tiene suficientes datos medidos
df = df.withColumn(
    target,
    F.when(
        F.col(target).isNull() &
        F.col("group_median").isNotNull() &
        (F.col("group_count") >= min_group_size),
        F.col("group_median")
    ).otherwise(F.col(target))
)

imputados_grupo = df.filter(
    (F.col(missing_col) == 1) & F.col(target).isNotNull()
).count()
print("Imputados por grupo:", imputados_grupo)
print("Nulos tras imputación por grupo:", df.filter(F.col(target).isNull()).count())

# Fallback: mediana global
global_median = df.filter(F.col(target).isNotNull()).approxQuantile(target, [0.5], 0.01)[0]
df = df.withColumn(
    target,
    F.when(F.col(target).isNull(), F.lit(global_median)).otherwise(F.col(target))
)

print("Nulos finales:", df.filter(F.col(target).isNull()).count())

# Limpiar columnas auxiliares de merge
df = df.drop("group_median", "group_count")

# Opcional: eliminar bins auxiliares si no los quieres conservar
df = df.drop(*bin_cols)

# COMMAND ----------

display(df.limit(1))

# COMMAND ----------

from pyspark.sql import functions as F

# Filtrar temporada 25-26
df_2526 = df.filter(F.col("season") == "25-26")

# Resumen general
resumen_2526 = df_2526.agg(
    F.countDistinct("id_unico_deportista").alias("n_jugadores"),
    F.count("*").alias("n_registros")
)

resumen_2526.show()

# COMMAND ----------

sesiones_por_jugador_2526 = (
    df_2526
    .groupBy("id_unico_deportista")
    .agg(
        F.count("*").alias("n_sesiones"),
        F.min("activity_date").alias("primera_sesion"),
        F.max("activity_date").alias("ultima_sesion")
    )
    .orderBy(F.desc("n_sesiones"))
)

sesiones_por_jugador_2526.show(50, truncate=False)

# COMMAND ----------

sesiones_por_jugador_2526.select("n_sesiones").summary().show()

# COMMAND ----------

umbrales = [0,5, 10, 15, 20, 30, 40]

for u in umbrales:
    n = (
        sesiones_por_jugador_2526
        .filter(F.col("n_sesiones") >= u)
        .count()
    )
    print(f"Jugadores con >= {u} sesiones: {n}")

# COMMAND ----------

# MAGIC %md
# MAGIC Nos quedamos con los registros de la temporada 25-26, para aquellos jugadores que no lleguen al mínimo de 20 registros en esta temporada se seleccionan también los registros de la temporada pasada(24-25).

# COMMAND ----------

min_sesiones = 20

# 1. Sesiones por jugador en 25-26
sesiones_2526_por_jugador = (
    df
    .filter(F.col("season") == "25-26")
    .groupBy("id_unico_deportista")
    .agg(F.count("*").alias("n_sesiones_25_26"))
)

# 2. Jugadores con menos de 20 sesiones en 25-26
jugadores_menos_20_2526 = (
    sesiones_2526_por_jugador
    .filter(F.col("n_sesiones_25_26") < min_sesiones)
    .select("id_unico_deportista")
)

# 3. Todas las sesiones de 25-26
df_2526 = (
    df
    .filter(F.col("season") == "25-26")
)

# 4. Sesiones de 24-25 solo para los jugadores con menos de 20 sesiones en 25-26
df_2425_extra = (
    df
    .filter(F.col("season") == "24-25")
    .join(jugadores_menos_20_2526, on="id_unico_deportista", how="inner")
)

# 5. Dataset final para clustering
df_cluster_base = df_2526.unionByName(df_2425_extra)

# 6. Comprobación
resumen_cluster_base = (
    df_cluster_base
    .groupBy("id_unico_deportista")
    .agg(
        F.count("*").alias("n_sesiones_total"),
        F.sum(F.when(F.col("season") == "25-26", 1).otherwise(0)).alias("n_sesiones_25_26"),
        F.sum(F.when(F.col("season") == "24-25", 1).otherwise(0)).alias("n_sesiones_24_25"),
        F.countDistinct("season").alias("n_temporadas")
    )
    .orderBy(F.asc("n_sesiones_total"))
)

resumen_cluster_base.show(100, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Sigue habiendo jugadores que no llegan al mínimo de sesiones, suponemos que se deebrán a jugadores que se incorporaron al club recientemente.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Umbral mínimo de minutos para considerar una sesión válida
min_minutos = 45

df_cluster_base_filtrado = (
    df_cluster_base
    .filter(F.col("participation_time_minutes") >= min_minutos)
)

# Comprobación general
print("Registros antes:", df_cluster_base.count())
print("Registros después:", df_cluster_base_filtrado.count())

print(
    "Jugadores antes:",
    df_cluster_base.select("id_unico_deportista").distinct().count()
)

print(
    "Jugadores después:",
    df_cluster_base_filtrado.select("id_unico_deportista").distinct().count()
)

# COMMAND ----------

sesiones_validas_por_jugador = (
    df_cluster_base_filtrado
    .groupBy("id_unico_deportista")
    .agg(
        F.count("*").alias("n_sesiones_validas"),
        F.avg("participation_time_minutes").alias("avg_minutes"),
        F.min("participation_time_minutes").alias("min_minutes"),
        F.max("participation_time_minutes").alias("max_minutes"),
        F.countDistinct("season").alias("n_temporadas")
    )
    .orderBy(F.asc("n_sesiones_validas"))
)

sesiones_validas_por_jugador.show(100, truncate=False)

# COMMAND ----------

vars_gps = [
    "total_distance",
    "max_velocity",
    "max_acceleration",
    "max_deceleration",
    "acc_dcc_distance",
    "acc_dcc_count",
    "HSR",
    "sprint_distance",
    "sprint_count",
    "sprint_high_intensity_distance",
    "TSS",
    "ratio_ac"
]

agg_exprs = [
    F.count("*").alias("n_sesiones_validas"),
    F.avg("participation_time_minutes").alias("avg_participation_minutes"),
    F.countDistinct("season").alias("n_temporadas")
]

for col_name in vars_gps:
    agg_exprs.extend([
        F.avg(col_name).alias(f"{col_name}_mean"),
        F.expr(f"percentile_approx({col_name}, 0.90)").alias(f"{col_name}_p90")
    ])

df_player_profile = (
    df_cluster_base_filtrado
    .groupBy("id_unico_deportista")
    .agg(*agg_exprs)
)

df_player_profile.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Antes de construir el clustering, hemos preparado la tabla GPS para que cada jugador tenga un perfil físico comparable y no dependa de sesiones aisladas o poco representativas. Primero seleccionamos como base todas las sesiones de la temporada 25-26 y, para los jugadores que no alcanzaban un mínimo de 20 sesiones en esa temporada, añadimos también sus sesiones de 24-25 para aumentar la fiabilidad del perfil. Después decidimos no usar la última sesión ni una sesión individual, sino agregar la información por jugador, ya que cada fila original representa un entrenamiento o partido y puede estar muy condicionada por el contexto. También descartamos sesiones con menos de 45 minutos de participación, porque los registros muy cortos pueden distorsionar las variables físicas acumuladas: no es comparable un jugador que participa 10 minutos con uno que juega una sesión completa. Además, tratamos previamente los nulos de sprint_high_intensity_distance: se conservó un indicador de si el dato era originalmente nulo y se imputó usando registros similares basados en variables correlacionadas. Finalmente, planteamos agrupar por jugador calculando la media de sus sesiones válidas para representar su perfil habitual y el percentil 90 para capturar sus valores altos o actuaciones de mayor rendimiento.

# COMMAND ----------

display(df_player_profile.head(10))

# COMMAND ----------

df_player_profile.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.gps_data_agregado_por_jugador"
)

# COMMAND ----------


