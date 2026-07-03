# Databricks notebook source
dbutils.library.restartPython()

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

TABLE_INJURY = "dev_silver.pitchvision.v_injury"
TABLE_GPS = "dev_silver.pitchvision.v_gps_data"
TABLE_COMPETITIVE = "dev_silver.pitchvision.v_competitive_profile"
TABLE_ISOPRONE = "dev_silver.pitchvision.v_physical_test_isoprone"
TABLE_NORDIC = "dev_silver.pitchvision.v_physical_test_nordic"

gps_raw        = spark.table(TABLE_GPS)
competitive_raw = spark.table(TABLE_COMPETITIVE)
isoprone_raw   = spark.table(TABLE_ISOPRONE)
nordic_raw     = spark.table(TABLE_NORDIC)
injury         = spark.table(TABLE_INJURY)

# COMMAND ----------

# MAGIC %md
# MAGIC # Historial de pruebas por jugador

# COMMAND ----------

gps_j = gps_raw.filter(F.col("season") != "23-24").select(
    F.col("id_unico_deportista").alias("gps_player"),
    F.col("activity_date").alias("gps_date"),
    F.col("activity_team_id").alias("gps_activity_team_id"),
    F.col("season").alias("gps_season"), 
    *[c for c in gps_raw.columns
      if c not in ("id_unico_deportista", "activity_date", "activity_team_id", "season")]
)

comp_j = competitive_raw.filter(F.col("season") != "23-24").select(
    F.col("id_unico_deportista").alias("comp_player"),
    F.col("activity_date").alias("comp_date"),
    F.col("activity_team_id").alias("comp_activity_team_id"),
    F.col("season").alias("comp_season"), 
    *[c for c in competitive_raw.columns
      if c not in ("id_unico_deportista", "activity_date", "activity_team_id", "season")]
)

iso_j = isoprone_raw.filter(F.col("season") != "23-24").select(
    F.col("id_unico_deportista").alias("iso_player"),
    F.col("test_date").alias("iso_date"),
    F.col("season").alias("iso_season"), 
    *[F.col(c).alias(f"iso_{c}")
      for c in isoprone_raw.columns
      if c not in ("id_unico_deportista", "test_date", "season")]
)

nor_j = nordic_raw.filter(F.col("season") != "23-24").select(
    F.col("id_unico_deportista").alias("nor_player"),
    F.col("test_date").alias("nor_date"),
    F.col("season").alias("nor_season"), 
    *[F.col(c).alias(f"nor_{c}")
      for c in nordic_raw.columns
      if c not in ("id_unico_deportista", "test_date", "season")]
)

# COMMAND ----------

spine = (
    gps_j.select(F.col("gps_player").alias("id_unico_deportista"),
                 F.col("gps_date").alias("activity_date"))
    .union(comp_j.select(F.col("comp_player").alias("id_unico_deportista"),
                         F.col("comp_date").alias("activity_date")))
    .union(iso_j.select(F.col("iso_player").alias("id_unico_deportista"),
                        F.col("iso_date").alias("activity_date")))
    .union(nor_j.select(F.col("nor_player").alias("id_unico_deportista"),
                        F.col("nor_date").alias("activity_date")))
    .distinct()
).cache()

print(f"Spine total          : {spine.count():,}")
display(spine)

# COMMAND ----------

df = (
    spine
    .join(gps_j,
          on=[spine["id_unico_deportista"] == gps_j["gps_player"],
              spine["activity_date"]       == gps_j["gps_date"]],
          how="left")
    .drop("gps_player", "gps_date")

    .join(comp_j,
          on=[spine["id_unico_deportista"] == comp_j["comp_player"],
              spine["activity_date"]       == comp_j["comp_date"]],
          how="left")
    .drop("comp_player", "comp_date")

    .join(iso_j,
          on=[spine["id_unico_deportista"] == iso_j["iso_player"],
              spine["activity_date"]       == iso_j["iso_date"]],
          how="left")
    .drop("iso_player", "iso_date")

    .join(nor_j,
          on=[spine["id_unico_deportista"] == nor_j["nor_player"],
              spine["activity_date"]       == nor_j["nor_date"]],
          how="left")
    .drop("nor_player", "nor_date")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Lesiones

# COMMAND ----------

injury_filtered = injury.filter(
    F.lower(F.col("injury_type")).isin([
        "muscolotendinosas",
        "muscotendinosa",
    ])
).filter(
    F.lower(F.col("injury_situation")).isin([
        "sin contacto"
        ])
)
print(f"Lesiones totales      : {injury.count():,}")
print(f"Lesiones filtradas    : {injury_filtered.count():,}")
injury_filtered.groupBy("injury_type").count().show()

# COMMAND ----------

# Jugadores que aparecen en injury
players_with_injury = (
    injury_filtered
    .select("id_unico_deportista")
    .distinct()
)

# Filtrar df para quedarnos solo con esos jugadores
df_filtered = df.join(
    players_with_injury,
    on="id_unico_deportista",
    how="inner"
)

print(f"Jugadores en injury : {players_with_injury.count():,}")
print(f"Filas df original   : {df.count():,}")
print(f"Filas df filtrado   : {df_filtered.count():,}")
print(f"Jugadores df filtrado: {df_filtered.select('id_unico_deportista').distinct().count():,}")

# COMMAND ----------

display(df_filtered)

# COMMAND ----------

display(injury_filtered)

# COMMAND ----------

# ── Preparar fechas de lesión filtradas ───────────────────────────────────────
injury_dates = (
    injury_filtered
    .select("id_unico_deportista", "injury_date", "injury_id")
    .withColumn("injury_date", F.to_date("injury_date"))
    .withColumnRenamed("id_unico_deportista", "inj_player")
    .distinct()
)

# ── Join y label ──────────────────────────────────────────────────────────────
df_final = (
    df_filtered
    .join(
        injury_dates,
        on=[
            df_filtered["id_unico_deportista"] == injury_dates["inj_player"],
            injury_dates["injury_date"]         >= df_filtered["activity_date"],
            injury_dates["injury_date"]         <= F.date_add(df_filtered["activity_date"], 7),
        ],
        how="left"
    )
    .drop("inj_player")
    .withColumn(
        "injury_risk",
        F.when(F.col("injury_id").isNotNull(), 1).otherwise(0)
    )
    .dropDuplicates(["id_unico_deportista", "activity_date"])
)

# ── Resumen ───────────────────────────────────────────────────────────────────
total     = df_final.count()
positivos = df_final.filter(F.col("injury_risk") == 1).count()
negativos = total - positivos

# print(f"Filas totales  : {total:,}")
# print(f"injury_risk = 1: {positivos:,}  ({round(positivos/total*100, 1)}%)")
# print(f"injury_risk = 0: {negativos:,}  ({round(negativos/total*100, 1)}%)")

# COMMAND ----------

display(df_final)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── 1. Preparar datos ──────────────────────────────────────────────────────────
# Si df_final es un Spark DataFrame, conviértelo primero:
df = df_final.toPandas()

df["activity_date"] = pd.to_datetime(df["activity_date"])
df["injury_date"]   = pd.to_datetime(df["injury_date"])

# ── 2. Histograma 1: actividades con lesión por fecha ─────────────────────────
df_lesion = df[df["injury_date"].notna()]  # solo filas con lesión registrada

lesiones_por_fecha = (
    df_lesion
    .groupby("activity_date")
    .size()
    .reset_index(name="n_lesiones")
)

fig, ax = plt.subplots(figsize=(14, 4))

ax.bar(
    lesiones_por_fecha["activity_date"],
    lesiones_por_fecha["n_lesiones"],
    width=1,
    color="#2a78d6",
    edgecolor="white",
    linewidth=0.4,
)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, ha="right")

ax.set_xlabel("Fecha de actividad")
ax.set_ylabel("Nº de lesiones")
ax.set_title("Lesiones por fecha de actividad")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

# ── 3. Histograma 2: injury_risk (0/1) por fecha de actividad ─────────────────
risk_por_fecha = (
    df
    .groupby("activity_date")["injury_risk"]
    .sum()          # suma los 1s → total en riesgo ese día
    .reset_index(name="n_risk")
)

fig, ax = plt.subplots(figsize=(14, 4))

ax.bar(
    risk_por_fecha["activity_date"],
    risk_por_fecha["n_risk"],
    width=1,
    color="#e34948",
    edgecolor="white",
    linewidth=0.4,
)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, ha="right")

ax.set_xlabel("Fecha de actividad")
ax.set_ylabel("Deportistas en riesgo (injury_risk = 1)")
ax.set_title("Injury risk por fecha de actividad")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Untitled
df_final.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.injury_risk_input_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC # Análisis exploratorio

# COMMAND ----------

TABLE_RAW = "dev_gold.pitchvision.injury_risk_input_raw"

df = spark.table(TABLE_RAW)

# COMMAND ----------

print(f"EDA DE TABLA: {TABLE_RAW}")


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

#  Nulos por columna

print("\n5. NULOS POR COLUMNA")
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
# MAGIC ### Eliminar variables cualitativas

# COMMAND ----------

# 1. Definimos las columnas intocables (las que NO queremos borrar aunque sean texto)
columnas_a_salvar = ["id_unico_deportista", "activity_date"]

# 2. Identificamos qué columnas cualitativas (strings) vamos a eliminar realmente
columnas_a_borrar = [
    col for col in string_cols 
    if col not in columnas_a_salvar
]

# 3. Mostramos por pantalla lo que vamos a hacer por seguridad y control
print(f"Se van a eliminar {len(columnas_a_borrar)} columnas cualitativas.")
print(f"Columnas eliminadas: {columnas_a_borrar}")

# COMMAND ----------

# 4. Ejecutamos el borrado masivo desempaquetando la lista con el asterisco (*)
df_filtrado = df.drop(*columnas_a_borrar)

# 5. Comprobamos las nuevas dimensiones
print("\n--- RESULTADO TRAS LA LIMPIEZA ---")
print(f"Columnas originales: {len(df.columns)}")
print(f"Columnas tras borrar cualitativas: {len(df_filtrado.columns)}")

# Visualizamos cómo ha quedado la tabla
display(df_filtrado.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estudio de jugadores sin ningún dato por columna

# COMMAND ----------

print("1. Calculando la distribución de nulos por jugador...")

# Extraemos la lista de IDs únicos de los deportistas
lista_jugadores = [row["id_unico_deportista"] for row in df_filtrado.select("id_unico_deportista").distinct().collect()]

# 1. Agrupamos por jugador y calculamos si TODO su histórico en esa columna es nulo.
# Si el conteo de registros válidos (no nulos) de un jugador es 0, da 1 (True), si no, 0.
exprs_jugador_vacio = [
    F.when(F.count(F.col(c)) == 0, 1).otherwise(0).alias(c) 
    for c in df_filtrado.columns if c != "id_unico_deportista"
]

df_jugadores_vacios = df_filtrado.groupBy("id_unico_deportista").agg(*exprs_jugador_vacio)

# 2. Sumamos cuántos jugadores totales quedaron en 1 (sin datos) para cada columna
exprs_suma_total = [F.sum(F.col(c)).alias(c) for c in df_jugadores_vacios.columns if c != "id_unico_deportista"]
totales_por_columna = df_jugadores_vacios.agg(*exprs_suma_total).first()

# 3. Umbrales basados en el total real de jugadores únicos
total_jugadores = len(lista_jugadores)
umbral_95_porciento = total_jugadores * 0.95

# Creamos la lista solicitada para almacenar las columnas críticas
columnas_criticas_95 = []

# Cabecera de la tabla solicitada
print(f"\nTABLA DE DISTRIBUCIÓN DE NULOS (Total plantilla: {total_jugadores} jugadores):")
print(f"{'COLUMNA':<45} | {'JUGADORES TODO NULL':<22} | {'% JUGADORES TODO NULL':<25}")
print("-" * 100)

# Paso D: Iteramos los resultados para pintar la tabla y rellenar la variable
for c in df_jugadores_vacios.columns:
    if c == "id_unico_deportista":
        continue
        
    jugadores_sin_datos = totales_por_columna[c]
    porcentaje_sin_datos = (jugadores_sin_datos / total_jugadores) * 100
    
    # Si supera el 95% de la plantilla sin un solo registro, va a la variable de críticas
    if jugadores_sin_datos > umbral_95_porciento:
        columnas_criticas_95.append(c)
        print(f"{c:<45} | {jugadores_sin_datos:<22} | {porcentaje_sin_datos:<24.2f}% <-- CRÍTICA (>95%)")
    else:
        print(f"{c:<45} | {jugadores_sin_datos:<22} | {porcentaje_sin_datos:<24.2f}%")

print("-" * 100)
print(f"¡Análisis completado! La variable 'columnas_criticas_95' ha guardado {len(columnas_criticas_95)} columnas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imputar valores para isoporne, nordic y competitive, imputación por arrastre 
# MAGIC Para evitar perder los días de GPS, rellenamos los nulos de las variables de fuerza (Isoprone y Nordic) y competición con el último valor válido registrado para ese deportista. 
# MAGIC Para ello, creamos una ventana particionada por `id_unico_deportista` y ordenada por `activity_date`. Usamos la función `last` con el parámetro `ignorenulls=True` para que busque hacia atrás en el tiempo el test más reciente.

# COMMAND ----------

# 1. Identificamos dinámicamente las columnas que empiezan por iso_, nor_ o comp_
prefijos_objetivo = ("iso_", "nor_", "comp_")
numeric_cols_to_impute = [
    c for c in df_filtrado.columns 
    if c.startswith(prefijos_objetivo)
]
print(f"Se han detectado {len(numeric_cols_to_impute)} columnas para aplicar Forward Fill.")
print(numeric_cols_to_impute)

# COMMAND ----------

ID_COL = "id_unico_deportista"
DATE_COL = "activity_date"


# Aseguramos que test_date sea tipo date
df_imputed = df_filtrado.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))


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

#  Nulos por columna

print("\n5. NULOS POR COLUMNA")
null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_imputed.columns
]

nulls_df_imputed = df_imputed.agg(*null_exprs)

nulls_long = nulls_df_imputed.select(
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
# MAGIC ### Eliminación de filas sin dato GPS

# COMMAND ----------

print(f"Registros antes del filtrado: {df_imputed.count()}")

# Eliminamos las filas donde 'acc_count' sea nulo
df_sin_nulos_gps = df_imputed.dropna(subset=["acc_count"])

# Guardamos el conteo final para verificar el impacto
total_final_filas = df_sin_nulos_gps.count()

print(f"Registros después del filtrado: {total_final_filas}")
print("¡Filas eliminadas con éxito! El nuevo DataFrame es 'df_sin_nulos_gps'.")

# COMMAND ----------

df_check = df_sin_nulos_gps

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
# MAGIC ### Eliminación de jugadores sin dato en isoprone o nordic

# COMMAND ----------

from pyspark.sql import functions as F

columna_objetivo = "nor_mdc_right_torque_nm"

# 1. Filtramos las filas donde la columna específica sea nula
df_nulos_especificos = df_sin_nulos_gps.filter(F.col(columna_objetivo).isNull()) \
                                       .select("id_unico_deportista", "activity_date", columna_objetivo) \
                                       .orderBy("id_unico_deportista", "activity_date")

# 2. Contamos cuántos días/filas se han quedado colgadas
total_nulos_especificos = df_nulos_especificos.count()

print(f"Se han detectado {total_nulos_especificos} filas con nulos en la columna '{columna_objetivo}'.")
print("A continuación se muestran los jugadores y las fechas afectadas:")

# 3. Mostramos la tabla en Databricks
display(df_nulos_especificos)

# COMMAND ----------

from pyspark.sql import functions as F

# Coloca aquí el ID del jugador que quieras revisar (por ejemplo: 3, "Jugador_05", etc.)
id_jugador_a_consultar = "41de6bfa-470b-4486-8f9f-145e365768cf"

# 1. Filtramos por el jugador y seleccionamos la columna objetivo junto a la fecha
df_historial_jugador = df_sin_nulos_gps.filter(F.col("id_unico_deportista") == id_jugador_a_consultar) \
                                       .select("id_unico_deportista", "activity_date", "nor_mdc_right_torque_nm") \
                                       .orderBy("activity_date")

# 2. Contamos los días registrados para este jugador para tener contexto
total_dias = df_historial_jugador.count()
print(f"Mostrando los {total_dias} registros de actividad del jugador '{id_jugador_a_consultar}':")

# 3. Mostramos la tabla interactiva en Databricks
display(df_historial_jugador)

# COMMAND ----------

# Coloca aquí el ID del jugador que vas a eliminar de la base de datos
id_jugador_a_eliminar = "41de6bfa-470b-4486-8f9f-145e365768cf"

print(f"Registros totales antes de la eliminación: {df_sin_nulos_gps.count()}")

# Filtramos el DataFrame para quedarnos con todos los jugadores EXCEPTO el elegido
# El operador != significa "diferente de"
df_less_players_1 = df_sin_nulos_gps.filter(F.col("id_unico_deportista") != id_jugador_a_eliminar)

print(f"Registros totales después de eliminar al jugador '{id_jugador_a_eliminar}': {df_less_players_1.count()}")
print("¡Jugador eliminado con éxito! El DataFrame se ha actualizado correctamente.")

# COMMAND ----------


id_jugador_a_eliminar = "f2625b27-af5c-462a-9f6a-1344dd9cdb9c"

print(f"Registros totales antes de la eliminación: {df_less_players_1.count()}")

# Filtramos el DataFrame para quedarnos con todos los jugadores EXCEPTO el elegido
# El operador != significa "diferente de"
df_less_players_2 = df_less_players_1.filter(F.col("id_unico_deportista") != id_jugador_a_eliminar)

print(f"Registros totales después de eliminar al jugador '{id_jugador_a_eliminar}': {df_less_players_2.count()}")
print("¡Jugador eliminado con éxito! El DataFrame se ha actualizado correctamente.")

# COMMAND ----------

df_check = df_less_players_2

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

columna_objetivo = "iso_right_repetitions"

# 1. Filtramos las filas donde la columna específica sea nula
df_nulos_especificos = df_less_players_2.filter(F.col(columna_objetivo).isNull()) \
                                       .select("id_unico_deportista", "activity_date", columna_objetivo) \
                                       .orderBy("id_unico_deportista", "activity_date")

# 2. Contamos cuántos días/filas se han quedado colgadas
total_nulos_especificos = df_nulos_especificos.count()

print(f"Se han detectado {total_nulos_especificos} filas con nulos en la columna '{columna_objetivo}'.")
print("A continuación se muestran los jugadores y las fechas afectadas:")

# 3. Mostramos la tabla en Databricks
display(df_nulos_especificos)

# COMMAND ----------

# Coloca aquí el ID del jugador que quieras revisar (por ejemplo: 3, "Jugador_05", etc.)
id_jugador_a_consultar = "dcec03cb-6b85-48c6-afc6-02409e0c54e7"

# 1. Filtramos por el jugador y seleccionamos la columna objetivo junto a la fecha
df_historial_jugador = df_less_players_2.filter(F.col("id_unico_deportista") == id_jugador_a_consultar) \
                                       .select("id_unico_deportista", "activity_date", "iso_right_repetitions") \
                                       .orderBy("activity_date")

# 2. Contamos los días registrados para este jugador para tener contexto
total_dias = df_historial_jugador.count()
print(f"Mostrando los {total_dias} registros de actividad del jugador '{id_jugador_a_consultar}':")

# 3. Mostramos la tabla interactiva en Databricks
display(df_historial_jugador)

# COMMAND ----------

id_jugador_a_eliminar = "dcec03cb-6b85-48c6-afc6-02409e0c54e7"

print(f"Registros totales antes de la eliminación: {df_less_players_2.count()}")

# Filtramos el DataFrame para quedarnos con todos los jugadores EXCEPTO el elegido
# El operador != significa "diferente de"
df_less_players_3 = df_less_players_2.filter(F.col("id_unico_deportista") != id_jugador_a_eliminar)

print(f"Registros totales después de eliminar al jugador '{id_jugador_a_eliminar}': {df_less_players_3.count()}")
print("¡Jugador eliminado con éxito! El DataFrame se ha actualizado correctamente.")

# COMMAND ----------

df_check = df_less_players_3

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
# MAGIC ### Eliminamos las columnas que siguen mostrando nulos, después de comprobar que o bien presentan un número demasiado alto de nulos o bien se determinó que era adecuado eliminarlas en la fase de clusterización

# COMMAND ----------

from pyspark.sql.functions import col

# Identificamos las columnas que no tienen nulos
cols_no_nulls = [
    c for c in df_less_players_3.columns
    if df_less_players_3.filter(col(c).isNull()).count() == 0
]

# Creamos un nuevo DataFrame solo con las columnas sin nulos
df_no_null_cols = df_less_players_3.select(*cols_no_nulls)

display(df_no_null_cols)

# COMMAND ----------

df_check = df_no_null_cols

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
# MAGIC ### Comprbamos el número de resultados de riesgo que teníamos al principio y que nos quedan

# COMMAND ----------

suma_injury_risk_old = df.agg(F.sum("injury_risk").alias("suma_injury_risk"))
display(suma_injury_risk_old)

# COMMAND ----------

suma_injury_risk_new = df_no_null_cols.agg(F.sum("injury_risk").alias("suma_injury_risk"))
display(suma_injury_risk_new)

# COMMAND ----------

# DBTITLE 1,Cell 50
df_no_null_cols.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.injury_risk_input_filtered")
