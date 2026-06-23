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
from functools import reduce
import numpy as np

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

# MAGIC %md
# MAGIC ### Imputación de nulos

# COMMAND ----------

# MAGIC %md
# MAGIC Para variables generales:
# MAGIC 1. Si hay valor, se deja.
# MAGIC 2. Si falta, usa el valor anterior del mismo jugador.
# MAGIC 3. Si no hay anterior, usa el valor posterior del mismo jugador.
# MAGIC
# MAGIC Para age_years:
# MAGIC 1. Si hay edad, se deja.
# MAGIC 2. Si falta, busca edad anterior/posterior del mismo jugador.
# MAGIC 3. Ajusta la edad según la diferencia entre fechas.
# MAGIC    Ejemplo: si en 24-25 tenía 18 y en 25-26 falta, estima 19.
# MAGIC No es exacto porque necesitaríamos la fecha de nacimiento pero respeta mucho más que usar la mediana del dataset.

# COMMAND ----------

ID_COL = "id_unico_deportista"
DATE_COL = "test_date"
AGE_COL = "age_years"

EXCLUDE_COLS = [
    ID_COL,
    AGE_COL,
    "season",
    "is_restricted"
]

# Aseguramos que test_date sea tipo date
df_imputed = df.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))

# Columnas numéricas a imputar, excepto edad e identificadores
numeric_cols_to_impute = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in EXCLUDE_COLS
]

print(f"Columnas numéricas a imputar con mediciones del mismo jugador: {len(numeric_cols_to_impute)}")
print(numeric_cols_to_impute)

# VENTANAS TEMPORALES POR JUGADOR
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

# 1. IMPUTACIÓN ESPECIAL PARA age_years
#    Ajusta edad según diferencia entre test_date y la fecha
#    donde sí existe age_years.
prev_age_struct = F.last(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_forward)

next_age_struct = F.first(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_backward)

df_imputed = (
    df_imputed
    .withColumn("_prev_age_ref", prev_age_struct)
    .withColumn("_next_age_ref", next_age_struct)
)

# Si usamos una medición anterior:
age_from_prev = (
    F.col("_prev_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_prev_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

# Si usamos una medición posterior:
age_from_next = (
    F.col("_next_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_next_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

df_imputed = (
    df_imputed
    .withColumn(
        AGE_COL,
        F.coalesce(
            F.col(AGE_COL),
            age_from_prev,
            age_from_next
        ).cast("int")
    )
    .drop("_prev_age_ref", "_next_age_ref")
)

print("Imputación de age_years terminada.")

# 2. IMPUTACIÓN DEL RESTO DE VARIABLES NUMÉRICAS
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

print("Imputación del resto de variables numéricas terminada.")

# 3. CACHEAR DF
df_imputed = df_imputed.cache()

print(f"Filas finales: {df_imputed.count():,}")

display(df_imputed.limit(20))

# COMMAND ----------

df_check = df_imputed

total_rows = df_check.count()

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_check.columns
]

nulls_row = df_check.agg(*null_exprs)

null_report_after = (
    nulls_row
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("columna"),
                    F.col(c).alias("nulos_restantes")
                )
                for c in df_check.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .withColumn(
        "pct_nulos_restantes",
        F.round(F.col("nulos_restantes") / F.lit(total_rows) * 100, 2)
    )
    .orderBy(F.col("nulos_restantes").desc())
)

display(
    null_report_after.filter(F.col("nulos_restantes") > 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Todavía quedan muchos nulos, lo que quiere decir que hay variables para los que algunos jugadores no tienen ningún registro.

# COMMAND ----------

exclude_cols = [
    "id_unico_deportista",
    "test_date",
    "season",
    "is_restricted",
    "age_years"
]

numeric_cols_check = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Columnas numéricas revisadas: {len(numeric_cols_check)}")

# 1. Una sola acción Spark: traer datos necesarios a pandas
pdf = (
    df_imputed
    .select(AGE_COL, *numeric_cols_check)
    .toPandas()
)

# 2. Crear grupos de edad en pandas
bins = [13, 16, 19, 23, 29, 34, 40]
labels = ["13-15", "16-18", "19-22", "23-28", "29-33", "34-39"]

pdf["age_group_imputation_check"] = pd.cut(
    pdf[AGE_COL],
    bins=bins,
    labels=labels,
    right=False
)

pdf["age_group_imputation_check"] = (
    pdf["age_group_imputation_check"]
    .astype("object")
    .fillna("SIN_GRUPO")
)

# 3. Calcular nulos por grupo y variable en pandas
rows = []

for age_group, group_df in pdf.groupby("age_group_imputation_check"):
    filas_grupo = len(group_df)

    null_counts = group_df[numeric_cols_check].isna().sum()

    for col, nulos in null_counts.items():
        if nulos > 0:
            rows.append({
                "columna": col,
                "age_group": age_group,
                "filas_grupo": filas_grupo,
                "nulos": int(nulos),
                "pct_nulos": round(nulos / filas_grupo * 100, 2)
            })

nulls_by_age_group_pd = (
    pd.DataFrame(rows)
    .sort_values(["nulos", "columna", "age_group"], ascending=[False, True, True])
)

display(nulls_by_age_group_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Como los grupos de 13-15 y 16-18 cuentan con una gran cantidad de nulos para algunas variables vamos a separar estos datos para un posible clustering por grupos de edad. En estos nuevos dfs eliminaremos las variables que cuentan con más de un 60% de nulos.

# COMMAND ----------

cols_drop_13_15 = [
    "ankle_perimeter_cm",
    "anteroposterior_abdomen_diameter_cm",
    "bimalleolar_diameter_cm",
    "braquial_index",
    "crural_index",
    "height_iliospinale_cm",
    "height_tibiale_laterale_cm",
    "height_trochanterion_cm",
    "intermembral_index",
    "length_acromiale_radiale_cm",
    "length_foot_cm",
    "length_radiale_stylion_cm",
    "length_stylion_medio_dactylion_cm",
    "length_tibiale_mediale_sphyrion_tibiale_cm",
    "length_trochanterion_tibiale_laterale_cm",
    "neck_perimeter_cm",
    "thigh_1cm_gluteus_perimeter_cm",
    "wrist_perimeter_cm"
]

cols_drop_16_18 = [
    "ankle_perimeter_cm",
    "anteroposterior_abdomen_diameter_cm",
    "bimalleolar_diameter_cm",
    "braquial_index",
    "crural_index",
    "height_iliospinale_cm",
    "height_tibiale_laterale_cm",
    "height_trochanterion_cm",
    "intermembral_index",
    "length_acromiale_radiale_cm",
    "length_foot_cm",
    "length_radiale_stylion_cm",
    "length_stylion_medio_dactylion_cm",
    "length_tibiale_mediale_sphyrion_tibiale_cm",
    "length_trochanterion_tibiale_laterale_cm",
    "neck_perimeter_cm",
    "thigh_1cm_gluteus_perimeter_cm",
    "wrist_perimeter_cm"
]

df_13_15 = (
    df_imputed
    .filter((F.col("age_years") >= 13) & (F.col("age_years") <= 15))
    .drop(*cols_drop_13_15)
)

df_16_18 = (
    df_imputed
    .filter((F.col("age_years") >= 16) & (F.col("age_years") <= 18))
    .drop(*cols_drop_16_18)
)

print(f"df_13_15: {df_13_15.count()} filas | {len(df_13_15.columns)} columnas")
print(f"df_16_18: {df_16_18.count()} filas | {len(df_16_18.columns)} columnas")

display(df_13_15.limit(10))
display(df_16_18.limit(10))

# COMMAND ----------

df_13_15.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.anthropometry_13_15"
)

df_16_18.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.anthropometry_16_18"
)

print("Tablas guardadas correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC El siguiente paso para el df principal es imputar los nulos que quedan utilizando la mediana del grupo de edad para la variable faltante.

# COMMAND ----------

# IMPUTACIÓN DE NULOS RESTANTES CON MEDIANA POR GRUPO DE EDAD

AGE_COL = "age_years"

# 1. Crear grupo de edad para imputación
df_general_imputed = (
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

# 2. Columnas numéricas candidatas a imputar
# No imputamos id, edad, flags ni columnas técnicas
exclude_cols = [
    "id_unico_deportista",
    "test_date",
    "season",
    "is_restricted",
    AGE_COL,
    "_age_group_imputation"
]

numeric_cols = [
    field.name
    for field in df_general_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Columnas numéricas candidatas: {len(numeric_cols)}")

# 3. Detectar solo columnas que todavía tienen nulos
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols
]

null_counts = df_general_imputed.agg(*null_exprs).collect()[0].asDict()

cols_with_nulls = [
    c for c, n in null_counts.items()
    if n > 0
]

print(f"Columnas con nulos restantes antes de mediana por edad: {len(cols_with_nulls)}")
print(cols_with_nulls)

# 4. Calcular medianas por grupo de edad para esas columnas
if len(cols_with_nulls) > 0:

    median_exprs = [
        F.expr(f"percentile_approx(`{c}`, 0.5)").alias(f"{c}__age_group_median")
        for c in cols_with_nulls
    ]

    age_group_medians = (
        df_general_imputed
        .filter(F.col("_age_group_imputation").isNotNull())
        .groupBy("_age_group_imputation")
        .agg(*median_exprs)
    )

    # 5. Unir medianas al dataframe
    df_joined = df_general_imputed.join(
        age_group_medians,
        on="_age_group_imputation",
        how="left"
    )

    # 6. Rellenar nulos con mediana del grupo en una sola selección
    select_exprs = []

    for c in df_general_imputed.columns:
        if c in cols_with_nulls:
            select_exprs.append(
                F.coalesce(
                    F.col(c),
                    F.col(f"{c}__age_group_median")
                ).alias(c)
            )
        else:
            select_exprs.append(F.col(c))

    df_general_imputed = (
        df_joined
        .select(*select_exprs)
        .drop("_age_group_imputation")
    )

else:
    print("No quedan nulos para imputar por grupo de edad.")
    df_general_imputed = df_general_imputed.drop("_age_group_imputation")

# 7. Cachear si vas a seguir trabajando con este dataframe
df_general_imputed = df_general_imputed.cache()

print(f"Filas finales: {df_general_imputed.count():,}")
display(df_general_imputed.limit(10))

# COMMAND ----------

df= df_general_imputed

# COMMAND ----------

df_check = df

total_rows = df_check.count()

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_check.columns
]

nulls_row = df_check.agg(*null_exprs)

null_report_after = (
    nulls_row
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("columna"),
                    F.col(c).alias("nulos_restantes")
                )
                for c in df_check.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .withColumn(
        "pct_nulos_restantes",
        F.round(F.col("nulos_restantes") / F.lit(total_rows) * 100, 2)
    )
    .orderBy(F.col("nulos_restantes").desc())
)

display(
    null_report_after.filter(F.col("nulos_restantes") > 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Hemos reducido claramente el número de nulos con valores propios del mismo jugador.
# MAGIC En el siguiente paso imputando por las mediana por grupo de edad todavía más. Los pocos nulos que quedan se deben a registros que no tienen la variable age_years y por lo tanto no se pudo realizar el paso previo.
# MAGIC
# MAGIC  Para los pocos nulos que quedan vamos a imputar usando la mediana de esa variable dentro de grupos discretizados por alturas y pesos usando stamps de 10 kg y 10cm.
# MAGIC  En caso de que haya algún grupo que no tenga datos para alguna variable se utiliza como fallback la mediana global (<0.1% de los casos)

# COMMAND ----------

# IMPUTACIÓN FINAL:
# 1. Mediana por grupos de height_cm y total_estimated_mass_kg de 10 en 10
# 2. Fallback: mediana global si todavía queda algún nulo

HEIGHT_COL = "height_cm"
MASS_COL = "total_estimated_mass_kg"

HEIGHT_BIN_SIZE = 10
MASS_BIN_SIZE = 10

# Partimos del dataframe imputado previamente
df_final_imputed = df

# Columnas que NO queremos imputar con esta lógica
EXCLUDE_COLS = [
    "id_unico_deportista",
    "test_date",
    "season",
    "is_restricted",
    HEIGHT_COL,
    MASS_COL
]

# Columnas numéricas candidatas
numeric_cols_remaining = [
    field.name
    for field in df_final_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in EXCLUDE_COLS
]

# 1. Detectar columnas que todavía tienen nulos
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols_remaining
]

null_counts = df_final_imputed.agg(*null_exprs).collect()[0].asDict()

cols_with_nulls = [
    c for c, n in null_counts.items()
    if n > 0
]

print(f"Columnas con nulos antes de imputación por grupos: {len(cols_with_nulls)}")
print(cols_with_nulls)

# 2. Crear grupos de altura y masa de 10 en 10
df_final_imputed = (
    df_final_imputed
    .withColumn(
        "_height_bin",
        F.floor(F.col(HEIGHT_COL) / F.lit(HEIGHT_BIN_SIZE)) * F.lit(HEIGHT_BIN_SIZE)
    )
    .withColumn(
        "_mass_bin",
        F.floor(F.col(MASS_COL) / F.lit(MASS_BIN_SIZE)) * F.lit(MASS_BIN_SIZE)
    )
)
# 3. Imputar con mediana por grupo height_bin + mass_bin
if len(cols_with_nulls) > 0:

    median_exprs_group = [
        F.expr(f"percentile_approx(`{c}`, 0.5)").alias(f"{c}__group_median")
        for c in cols_with_nulls
    ]

    group_medians = (
        df_final_imputed
        .groupBy("_height_bin", "_mass_bin")
        .agg(*median_exprs_group)
    )

    df_final_imputed = df_final_imputed.join(
        group_medians,
        on=["_height_bin", "_mass_bin"],
        how="left"
    )

    for c in cols_with_nulls:
        median_col = f"{c}__group_median"

        df_final_imputed = (
            df_final_imputed
            .withColumn(
                c,
                F.coalesce(
                    F.col(c),
                    F.col(median_col)
                )
            )
            .drop(median_col)
        )

else:
    print("No había nulos para imputar por grupos.")

# Quitamos columnas técnicas de grupos
df_final_imputed = df_final_imputed.drop("_height_bin", "_mass_bin")

# 4. Detectar columnas que todavía tienen nulos tras grupos
null_exprs_after_group = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols_remaining
]

null_counts_after_group = df_final_imputed.agg(*null_exprs_after_group).collect()[0].asDict()

cols_still_null = [
    c for c, n in null_counts_after_group.items()
    if n > 0
]

print(f"Columnas con nulos después de imputación por grupos: {len(cols_still_null)}")
print(cols_still_null)

# 5. Fallback: mediana global para nulos restantes
if len(cols_still_null) > 0:

    global_median_exprs = [
        F.expr(f"percentile_approx(`{c}`, 0.5)").alias(c)
        for c in cols_still_null
    ]

    global_medians_row = (
        df_final_imputed
        .agg(*global_median_exprs)
        .collect()[0]
        .asDict()
    )

    # Quitamos medianas None por seguridad
    global_medians = {
        c: v
        for c, v in global_medians_row.items()
        if v is not None
    }

    print("Medianas globales usadas como fallback:")
    print(global_medians)

    df_final_imputed = df_final_imputed.fillna(global_medians)

else:
    print("No quedan nulos después de imputación por grupos.")

# 6. Comprobar nulos finales
null_exprs_final = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols_remaining
]

null_counts_final = df_final_imputed.agg(*null_exprs_final).collect()[0].asDict()

cols_final_nulls = [
    (c, n)
    for c, n in null_counts_final.items()
    if n > 0
]

print(f"Columnas numéricas con nulos finales: {len(cols_final_nulls)}")
print(cols_final_nulls)

df_final_imputed = df_final_imputed.cache()
print(f"Filas finales: {df_final_imputed.count():,}")

# COMMAND ----------

df_check = df_final_imputed

total_rows = df_check.count()

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_check.columns
]

nulls_row = df_check.agg(*null_exprs)

null_report_after = (
    nulls_row
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("columna"),
                    F.col(c).alias("nulos_restantes")
                )
                for c in df_check.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .withColumn(
        "pct_nulos_restantes",
        F.round(F.col("nulos_restantes") / F.lit(total_rows) * 100, 2)
    )
    .orderBy(F.col("nulos_restantes").desc())
)

display(
    null_report_after.filter(F.col("nulos_restantes") > 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC No quedan nulos

# COMMAND ----------

df= df_final_imputed

# COMMAND ----------

print(df.is_cached)
print(df.count())

# COMMAND ----------

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

THRESHOLD_EXCESO = 5
MAX_POINTS = 10000

# 1. Seleccionamos variables extremas
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
    # 2. Traemos todas las columnas necesarias a pandas UNA SOLA VEZ
    plot_pd = (
        df
        .select(*top_extreme_cols)
        .limit(MAX_POINTS)
        .toPandas()
    )

    # 3. Pintamos desde pandas, sin volver a llamar a Spark
    for c in top_extreme_cols:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]
        iqr = p75 - p25

        values = plot_pd[c].dropna()

        if values.empty:
            print(f"No hay datos para plotear en columna: {c}")
            continue

        min_val = values.min()
        max_val = values.max()

        lower_excess = max(0, p05 - min_val) / iqr if iqr != 0 else 0
        upper_excess = max(0, max_val - p95) / iqr if iqr != 0 else 0
        max_exceso = max(lower_excess, upper_excess)

        plt.figure(figsize=(8, 5))

        plt.hist(
            values,
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

# MAGIC %md
# MAGIC ## Tratamiento de valores extremos mediante winsorización
# MAGIC
# MAGIC En lugar de eliminar filas completas, se aplica una winsorización. Esto consiste en limitar los valores extremos a los percentiles calculados previamente:
# MAGIC
# MAGIC - Si un valor está por debajo del percentil 5 (`p05`), se reemplaza por `p05`.
# MAGIC - Si un valor está por encima del percentil 95 (`p95`), se reemplaza por `p95`.
# MAGIC - Si el valor está dentro del rango `p05-p95`, se mantiene sin cambios.
# MAGIC
# MAGIC Este enfoque permite reducir el impacto de valores extremos sobre análisis posteriores, como PCA o clustering, sin perder observaciones completas de jugadores.

# COMMAND ----------

# WINSORIZACIÓN DE VARIABLES EXTREMAS
# Si una variable tiene max_exceso_normalizado > THRESHOLD_EXCESO:
#   valor < p05  -> p05
#   valor > p95  -> p95

THRESHOLD_EXCESO = 5

# 1. Seleccionar variables con exceso normalizado alto
extreme_cols_rows = (
    extreme_severity_df
    .filter(F.col("max_exceso_normalizado") > THRESHOLD_EXCESO)
    .orderBy(F.col("max_exceso_normalizado").desc())
    .collect()
)

cols_to_winsorize = [row["columna"] for row in extreme_cols_rows]

print(f"Variables a winsorizar con max_exceso_normalizado > {THRESHOLD_EXCESO}:")
print(f"Total: {len(cols_to_winsorize)}")
print(cols_to_winsorize)

# 2. Crear dataframe winsorizado
df_winsorized = df

# 3. Aplicar p05 / p95 por variable
for c in cols_to_winsorize:
    p05 = percentiles_dict[c]["p05"]
    p95 = percentiles_dict[c]["p95"]

    df_winsorized = df_winsorized.withColumn(
        c,
        F.when(F.col(c) < F.lit(p05), F.lit(p05))
         .when(F.col(c) > F.lit(p95), F.lit(p95))
         .otherwise(F.col(c))
    )

print("Winsorización aplicada.")

display(df_winsorized.limit(20))

# COMMAND ----------

df=df_winsorized

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estudio de correlación de variables

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
df = df_reduced
df.cache()
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estudio de outliers por grupos de edad
# MAGIC En la anterior sección de outliers se windsorizaron valores extremos que podemos asumir que son ruido pero, no se entro en detalle en otro tipo de outliers que pueden también afectar a los algoritmos de clusterización que serán aplicados a continuación.
# MAGIC En esta sección se revisan los valores extremos teniendo en cuenta la edad de los jugadores. En datos fisiológicos de futbolistas, comparar directamente a todos los jugadores puede llevar a errores, ya que las variables corporales y de composición física pueden variar mucho entre categorías formativas, jugadores sub-23, senior y veteranos. Por ello, primero se crean grupos de edad basados en rangos habituales del fútbol y después se analizan las distribuciones de las variables dentro de cada grupo mediante boxplots. Además, se detectan posibles outliers usando los límites del rango intercuartílico dentro de cada grupo de edad. De esta forma, un valor se considera extremo respecto a jugadores de una etapa similar, no respecto a toda la población. Estos outliers no se eliminan automáticamente; se utilizan como señales para revisar posibles errores de medición, problemas de unidades o jugadores con perfiles fisiológicos excepcionales.

# COMMAND ----------

df_age = (
    df
    .withColumn(
        "age_group",
        F.when((F.col("age_years") >= 13) & (F.col("age_years") <= 14), F.lit("13-14"))
         .when((F.col("age_years") >= 15) & (F.col("age_years") <= 16), F.lit("15-16"))
         .when((F.col("age_years") >= 17) & (F.col("age_years") <= 19), F.lit("17-19"))
         .when((F.col("age_years") >= 20) & (F.col("age_years") <= 23), F.lit("20-23"))
         .when((F.col("age_years") >= 24) & (F.col("age_years") <= 29), F.lit("24-29"))
         .when((F.col("age_years") >= 30) & (F.col("age_years") <= 34), F.lit("30-34"))
         .when((F.col("age_years") >= 35) & (F.col("age_years") <= 40), F.lit("35-39"))
         .otherwise(F.lit("Otros"))
    )
)

display(
    df_age
    .groupBy("age_group")
    .count()
    .orderBy("age_group")
)

# COMMAND ----------

cols_to_plot = [
    "height_cm",
    "total_estimated_mass_kg",
    "faulkner_plus_carter_mean_fat_pct",
    "skinfold_sum_8_mm",
    "fat_muscle_index",
    "body_surface_area",
    "ectomorphy",
    "endomorphy",
    "mesomorphy"
]

age_order = ["13-14", "15-16", "17-19", "20-23", "24-29", "30-34", "35-39"]

plot_pd = (
    df_age
    .select("age_group", *cols_to_plot)
    .where(F.col("age_group").isin(age_order))
    .toPandas()
)

for c in cols_to_plot:
    plt.figure(figsize=(10, 5))
    
    sns.boxplot(
        data=plot_pd,
        x="age_group",
        y=c,
        order=age_order
    )
    
    plt.title(f"Distribución de {c} por grupo de edad")
    plt.xlabel("Grupo de edad")
    plt.ylabel(c)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

df= df_age

# COMMAND ----------

df.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.anthropometry_puntodecontrol"
)

# COMMAND ----------



ID_COL = "id_unico_deportista"
AGE_COL = "age_years"
AGE_GROUP_COL = "age_group"

# Usa 1.5 para regla clásica boxplot
# Usa 3.0 si quieres criterio más conservador
IQR_FACTOR = 1.5

exclude_cols = [
    ID_COL,
    "test_date",
    "season",
    "is_restricted",
    AGE_COL
]

numeric_cols_for_outliers = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Columnas numéricas analizadas: {len(numeric_cols_for_outliers)}")

# 1. Una sola acción Spark
cols_to_collect = [
    ID_COL,
    AGE_COL,
    AGE_GROUP_COL
] + numeric_cols_for_outliers

pdf = df.select(*cols_to_collect).toPandas()

print(f"Filas traídas a pandas: {len(pdf)}")

# 2. Calcular outliers por grupo de edad y variable
outlier_rows = []

for c in numeric_cols_for_outliers:
    
    for age_group, group_df in pdf.groupby(AGE_GROUP_COL):
        
        if pd.isna(age_group):
            continue
        
        values = group_df[c].dropna()
        
        if len(values) == 0:
            continue
        
        p25 = values.quantile(0.25)
        p75 = values.quantile(0.75)
        iqr = p75 - p25
        
        if pd.isna(iqr) or iqr == 0:
            continue
        
        lower_bound = p25 - IQR_FACTOR * iqr
        upper_bound = p75 + IQR_FACTOR * iqr
        
        mask_outlier = (
            group_df[c].notna()
            & (
                (group_df[c] < lower_bound)
                | (group_df[c] > upper_bound)
            )
        )
        
        outliers_tmp = group_df.loc[
            mask_outlier,
            [ID_COL, AGE_COL, AGE_GROUP_COL, c]
        ].copy()
        
        if len(outliers_tmp) == 0:
            continue
        
        outliers_tmp["variable"] = c
        outliers_tmp["valor"] = outliers_tmp[c]
        outliers_tmp["p25"] = p25
        outliers_tmp["p75"] = p75
        outliers_tmp["iqr"] = iqr
        outliers_tmp["lower_bound"] = lower_bound
        outliers_tmp["upper_bound"] = upper_bound
        outliers_tmp["outlier_side"] = np.where(
            outliers_tmp["valor"] < lower_bound,
            "lower",
            "upper"
        )
        
        outlier_rows.append(
            outliers_tmp[
                [
                    ID_COL,
                    AGE_COL,
                    AGE_GROUP_COL,
                    "variable",
                    "valor",
                    "outlier_side",
                    "p25",
                    "p75",
                    "iqr",
                    "lower_bound",
                    "upper_bound"
                ]
            ]
        )

# 3. Resultado final
if len(outlier_rows) > 0:
    outliers_by_age_pd = (
        pd.concat(outlier_rows, ignore_index=True)
        .sort_values([AGE_GROUP_COL, "variable", "valor"])
    )
    
    display(outliers_by_age_pd)
    
    # Si quieres volver a Spark:
    outliers_by_age_df = spark.createDataFrame(outliers_by_age_pd)
    
else:
    print("No se detectaron outliers por grupo de edad.")
    outliers_by_age_pd = pd.DataFrame()

# COMMAND ----------

outlier_summary_by_variable = (
    outliers_by_age_df
    .groupBy("variable")
    .agg(
        F.count("*").alias("num_outliers"),
        F.countDistinct(ID_COL).alias("num_jugadores_afectados"),
        F.collect_set(AGE_GROUP_COL).alias("grupos_edad_afectados")
    )
    .orderBy(F.col("num_outliers").desc())
)

display(outlier_summary_by_variable)

# COMMAND ----------

outlier_summary_by_player = (
    outliers_by_age_df
    .groupBy(ID_COL)
    .agg(
        F.count("*").alias("num_outliers"),
        F.collect_set("variable").alias("variables_outlier"),
        F.collect_set(AGE_GROUP_COL).alias("age_groups")
    )
    .orderBy(F.col("num_outliers").desc())
)

display(outlier_summary_by_player)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Winsorización conservadora por grupos de edad
# MAGIC
# MAGIC Tras analizar los outliers por grupos de edad, se observa que algunos jugadores acumulan un número elevado de valores extremos. Sin embargo, este conteo puede estar inflado porque el mismo jugador puede tener varias mediciones en distintas fechas o temporadas, y una misma característica física puede aparecer como outlier repetidamente. Por tanto, no todos estos casos deben interpretarse como errores de dato: en muchos casos pueden representar un perfil físico estable y diferencial del jugador. Para evitar eliminar jugadores o modificar en exceso características potencialmente relevantes, se aplica una winsorización conservadora por grupo de edad usando límites basados en `Q1 - 3 × IQR` y `Q3 + 3 × IQR`. Este criterio es menos agresivo que la regla clásica de `1.5 × IQR` y permite conservar perfiles fisiológicos extraordinarios, limitando únicamente valores extremadamente alejados que podrían distorsionar análisis posteriores como PCA o clustering.

# COMMAND ----------

# DBTITLE 1,Winsorización conservadora por edad
# WINSORIZACIÓN CONSERVADORA POR GRUPO DE EDAD
# Regla: Q1 - 3*IQR / Q3 + 3*IQR

ID_COL = "id_unico_deportista"
AGE_COL = "age_years"
AGE_GROUP_COL = "age_group"

IQR_FACTOR = 3.0

exclude_cols = [
    ID_COL,
    AGE_COL,
    AGE_GROUP_COL,
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

print(f"Columnas numéricas a winsorizar: {len(numeric_cols_to_winsorize)}")

# 1. Calcular Q1, Q3, IQR y límites para TODAS las columnas en UN solo groupBy
bound_exprs = []
for c in numeric_cols_to_winsorize:
    bound_exprs.append(F.expr(f"percentile_approx(`{c}`, 0.25)").alias(f"{c}__q1"))
    bound_exprs.append(F.expr(f"percentile_approx(`{c}`, 0.75)").alias(f"{c}__q3"))

bounds_df = df.groupBy(AGE_GROUP_COL).agg(*bound_exprs)

# 2. Unir bounds al df original (UN solo join)
df_winsor_age = df.join(bounds_df, on=AGE_GROUP_COL, how="left")

# 3. Aplicar winsorización con un solo select
select_exprs = []
for col_name in df.columns:
    if col_name in numeric_cols_to_winsorize:
        c = col_name
        q1_col = f"{c}__q1"
        q3_col = f"{c}__q3"
        iqr_expr = F.col(q3_col) - F.col(q1_col)
        lower_expr = F.col(q1_col) - F.lit(IQR_FACTOR) * iqr_expr
        upper_expr = F.col(q3_col) + F.lit(IQR_FACTOR) * iqr_expr

        winsorized = (
            F.when(
                F.col(c).isNotNull()
                & iqr_expr.isNotNull()
                & (iqr_expr > 0)
                & (F.col(c) < lower_expr),
                lower_expr
            )
            .when(
                F.col(c).isNotNull()
                & iqr_expr.isNotNull()
                & (iqr_expr > 0)
                & (F.col(c) > upper_expr),
                upper_expr
            )
            .otherwise(F.col(c))
        )
        select_exprs.append(winsorized.alias(c))
    else:
        select_exprs.append(F.col(col_name))

df_winsor_age = df_winsor_age.select(*select_exprs)

print("Winsorizáción conservadora por grupo de edad aplicada.")

df_winsor_age = df_winsor_age.cache()
print(f"Filas finales: {df_winsor_age.count():,}")

display(df_winsor_age.limit(20))

# COMMAND ----------

df= df_winsor_age


# COMMAND ----------

df.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.anthropometrylimpio_varios_registros_por_jugador"
)

# COMMAND ----------

# DATASET FINAL PARA CLUSTERING
# 1 fila = 1 jugador
# Variables escaladas por grupo de edad

ID_COL = "id_unico_deportista"
DATE_COL = "test_date"
AGE_COL = "age_years"
AGE_GROUP_COL = "age_group"

df_base = df
# Aseguramos que test_date sea fecha
df_base = df_base.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))

# 1. Quedarse con la medición más reciente por jugador
w_latest = (
    Window
    .partitionBy(ID_COL)
    .orderBy(F.col(DATE_COL).desc())
)

df_latest = (
    df_base
    .withColumn("_rn_latest", F.row_number().over(w_latest))
    .filter(F.col("_rn_latest") == 1)
    .drop("_rn_latest")
)

print(f"Filas tras quedarnos con última medición: {df_latest.count():,}")
print(f"Jugadores únicos: {df_latest.select(ID_COL).distinct().count():,}")

# 2. Seleccionar variables numéricas para clustering
exclude_cols = [
    ID_COL,
    DATE_COL,
    AGE_COL,
    AGE_GROUP_COL,
    "season",
    "is_restricted"
]

feature_cols = [
    field.name
    for field in df_latest.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Variables originales usadas para clustering: {len(feature_cols)}")
print(feature_cols)

# 3. Escalado por grupo de edad
# z = (valor - media del grupo edad) / std del grupo edad
w_age = Window.partitionBy(AGE_GROUP_COL)

df_age_scaled = df_latest

for c in feature_cols:
    mean_c = F.avg(F.col(c)).over(w_age)
    std_c = F.stddev(F.col(c)).over(w_age)

    df_age_scaled = df_age_scaled.withColumn(
        f"{c}_age_scaled",
        F.when(
            F.col(c).isNotNull()
            & std_c.isNotNull()
            & (std_c != 0),
            (F.col(c) - mean_c) / std_c
        ).otherwise(F.lit(0.0))
    )

age_scaled_feature_cols = [
    f"{c}_age_scaled"
    for c in feature_cols
]

print(f"Variables escaladas por edad creadas: {len(age_scaled_feature_cols)}")

# 4. VectorAssembler para clustering
assembler = VectorAssembler(
    inputCols=age_scaled_feature_cols,
    outputCol="features_age_scaled",
    handleInvalid="keep"
)

df_cluster_ready = assembler.transform(df_age_scaled)

# 5. Selección final de columnas útiles
df_cluster_ready = df_cluster_ready.select(
    ID_COL,
    DATE_COL,
    AGE_COL,
    AGE_GROUP_COL,
    *feature_cols,
    *age_scaled_feature_cols,
    "features_age_scaled"
)

df_cluster_ready = df_cluster_ready.cache()

print(f"Filas finales para clustering: {df_cluster_ready.count():,}")
print(f"Jugadores únicos finales: {df_cluster_ready.select(ID_COL).distinct().count():,}")

display(df_cluster_ready.limit(20))

# COMMAND ----------

df_cluster_ready.write.mode("overwrite").saveAsTable(
    "dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering"
)
