# Databricks notebook source
df_isoprone = spark.table("dev_gold.pitchvision.clustersresults_isoprone") \
    .select("id_unico_deportista", "cluster_PCA3_K4") \
    .withColumnRenamed("cluster_PCA3_K4", "cluster_isoprone")

df_nordic = spark.table("dev_gold.pitchvision.clustersresults_nordic") \
    .select("id_unico_deportista", "cluster_PCA4_K4") \
    .withColumnRenamed("cluster_PCA4_K4", "cluster_nordic")

display(df_isoprone)
display(df_nordic)

# COMMAND ----------

from pyspark.sql import functions as F

ID_COL = "id_unico_deportista"

# Cambia estos nombres si tus columnas se llaman distinto
CLUSTER_ISO = "cluster_isoprone"
CLUSTER_NORDIC = "cluster_nordic"

df_iso_nordic = (
    df_isoprone
    .select(ID_COL, F.col(CLUSTER_ISO).alias("cluster_isoprone"))
    .join(
        df_nordic.select(ID_COL, F.col(CLUSTER_NORDIC).alias("cluster_nordic")),
        on=ID_COL,
        how="inner"
    )
)

print("Jugadores en ambas tablas:", df_iso_nordic.count())

display(df_iso_nordic)

# COMMAND ----------

df_transition = (
    df_iso_nordic
    .groupBy("cluster_isoprone", "cluster_nordic")
    .agg(F.count("*").alias("n_jugadores"))
    .orderBy("cluster_isoprone", "cluster_nordic")
)

display(df_transition)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# NOMBRES INTERPRETABLES
# =========================

iso_cluster_names = {
    0: "Iso 0 | Baja fuerza + mayor asimetría",
    1: "Iso 1 | Perfil medio-alto equilibrado",
    2: "Iso 2 | Alta fuerza + baja asimetría",
    3: "Iso 3 | Perfil intermedio + baja asimetría"
}

nordic_cluster_names = {
    0: "Nordic 0 | Perfil medio",
    1: "Nordic 1 | Alta fuerza",
    2: "Nordic 2 | Baja fuerza",
    3: "Nordic 3 | Alta asimetría"
}

# =========================
# MATRIZ ISOPRONE x NORDIC
# =========================

transition_pd = df_transition.toPandas()

pivot_pd = transition_pd.pivot(
    index="cluster_isoprone",
    columns="cluster_nordic",
    values="n_jugadores"
).fillna(0)

display(pivot_pd)

# Etiquetas interpretables
x_labels = [
    nordic_cluster_names.get(int(c), f"Nordic {c}")
    for c in pivot_pd.columns
]

y_labels = [
    iso_cluster_names.get(int(i), f"Isoprone {i}")
    for i in pivot_pd.index
]

# =========================
# HEATMAP CON NOMBRES DE PERFIL
# =========================

plt.figure(figsize=(12, 6))

plt.imshow(
    pivot_pd,
    aspect="auto"
)

plt.colorbar(label="Nº jugadores")

plt.xticks(
    ticks=range(len(pivot_pd.columns)),
    labels=x_labels,
    rotation=35,
    ha="right"
)

plt.yticks(
    ticks=range(len(pivot_pd.index)),
    labels=y_labels
)

# Añadir números dentro de cada celda
for i in range(pivot_pd.shape[0]):
    for j in range(pivot_pd.shape[1]):
        value = int(pivot_pd.iloc[i, j])
        plt.text(
            j,
            i,
            value,
            ha="center",
            va="center"
        )

plt.xlabel("Perfil Nordic")
plt.ylabel("Perfil Isoprone")
plt.title("Coincidencia de jugadores entre perfiles Isoprone y Nordic")

plt.tight_layout()
plt.show()

# COMMAND ----------

import plotly.graph_objects as go

transition_pd = df_transition.toPandas()

iso_clusters = sorted(transition_pd["cluster_isoprone"].unique())
nordic_clusters = sorted(transition_pd["cluster_nordic"].unique())

labels = (
    [f"Isoprone {c}" for c in iso_clusters] +
    [f"Nordic {c}" for c in nordic_clusters]
)

iso_map = {c: i for i, c in enumerate(iso_clusters)}
nordic_map = {c: i + len(iso_clusters) for i, c in enumerate(nordic_clusters)}

source = transition_pd["cluster_isoprone"].map(iso_map).tolist()
target = transition_pd["cluster_nordic"].map(nordic_map).tolist()
value = transition_pd["n_jugadores"].tolist()

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

fig.update_layout(
    title_text="Flujo de jugadores entre clusters Isoprone y Nordic",
    font_size=10
)

fig.show()

# COMMAND ----------

iso_cluster_names = {
    0: "Iso 0 | Baja fuerza + mayor asimetría",
    1: "Iso 1 | Perfil medio-alto equilibrado",
    2: "Iso 2 | Alta fuerza + baja asimetría",
    3: "Iso 3 | Perfil intermedio + baja asimetría"
}

nordic_cluster_names = {
    0: "Nordic 0 | Perfil medio",
    1: "Nordic 1 | Alta fuerza",
    2: "Nordic 2 | Baja fuerza",
    3: "Nordic 3 | Alta asimetría"
}

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

# df_transition debe tener:
# cluster_isoprone, cluster_nordic, n_jugadores

transition_interp_pd = df_transition.toPandas()

transition_interp_pd["perfil_isoprone"] = transition_interp_pd["cluster_isoprone"].map(iso_cluster_names)
transition_interp_pd["perfil_nordic"] = transition_interp_pd["cluster_nordic"].map(nordic_cluster_names)

# Añadir porcentaje dentro de cada cluster Isoprone
transition_interp_pd["total_iso_cluster"] = (
    transition_interp_pd
    .groupby("cluster_isoprone")["n_jugadores"]
    .transform("sum")
)

transition_interp_pd["pct_dentro_isoprone"] = (
    transition_interp_pd["n_jugadores"] / transition_interp_pd["total_iso_cluster"] * 100
).round(2)

transition_interp_pd = transition_interp_pd.sort_values(
    ["cluster_isoprone", "n_jugadores"],
    ascending=[True, False]
)

display(transition_interp_pd[
    [
        "cluster_isoprone",
        "perfil_isoprone",
        "cluster_nordic",
        "perfil_nordic",
        "n_jugadores",
        "pct_dentro_isoprone"
    ]
])

# COMMAND ----------

dominant_transition_pd = (
    transition_interp_pd
    .sort_values(["cluster_isoprone", "n_jugadores"], ascending=[True, False])
    .groupby("cluster_isoprone", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

display(dominant_transition_pd[
    [
        "perfil_isoprone",
        "perfil_nordic",
        "n_jugadores",
        "pct_dentro_isoprone"
    ]
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPS + ANTHROPOMETRY

# COMMAND ----------

df_gps = spark.table("dev_gold.pitchvision.clustersresults_gps") \
    .select("id_unico_deportista", "cluster_umap_k4") \
    .withColumnRenamed("cluster_umap_k4", "cluster_gps")

df_anthropometry = spark.table("dev_gold.pitchvision.resultsanthropometry") \
    .select("id_unico_deportista", "cluster_umap_2_k4") \
    .withColumnRenamed("cluster_umap_2_k4", "cluster_anthropometry")

display(df_gps)
display(df_anthropometry)

# COMMAND ----------

from pyspark.sql import functions as F

ID_COL = "id_unico_deportista"

# Cambia estos nombres si tus columnas se llaman distinto
CLUSTER_GPS = "cluster_gps"
CLUSTER_ANTHRO = "cluster_anthropometry"

df_gps_anthro = (
    df_gps
    .select(ID_COL, F.col(CLUSTER_GPS).alias("cluster_gps"))
    .join(
        df_anthropometry.select(ID_COL, F.col(CLUSTER_ANTHRO).alias("cluster_anthropometry")),
        on=ID_COL,
        how="inner"
    )
)

print("Jugadores en ambas tablas:", df_gps_anthro.count())

display(df_gps_anthro)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

# =========================
# NOMBRES INTERPRETABLES DE CLUSTERS
# Ajusta los textos si quieres hacerlos más cortos/largos
# =========================

gps_cluster_names = {
    0: "GPS 0 | Medio/positivo",
    1: "GPS 1 | Baja carga externa",
    2: "GPS 2 | Alta carga/intensidad",
    3: "GPS 3 | Intermedio"
}

anthro_cluster_names = {
    0: "Anthro 0 | Robusto/muscular",
    1: "Anthro 1 | Mayor adiposidad",
    2: "Anthro 2 | Menor estructura",
    3: "Anthro 3 | Mayor estructura"
}

# =========================
# TABLA DE TRANSICIÓN GPS x ANTHROPOMETRY
# =========================

transition_pd = (
    df_gps_anthro
    .groupBy("cluster_gps", "cluster_anthropometry")
    .agg(F.count("*").alias("n_jugadores"))
    .orderBy("cluster_gps", "cluster_anthropometry")
    .toPandas()
)

pivot_pd = transition_pd.pivot(
    index="cluster_gps",
    columns="cluster_anthropometry",
    values="n_jugadores"
).fillna(0)

display(pivot_pd)

# Etiquetas interpretables
x_labels = [
    anthro_cluster_names.get(int(c), f"Anthro {c}")
    for c in pivot_pd.columns
]

y_labels = [
    gps_cluster_names.get(int(i), f"GPS {i}")
    for i in pivot_pd.index
]

# =========================
# HEATMAP CON NOMBRES DE PERFIL
# =========================

plt.figure(figsize=(10, 6))

plt.imshow(
    pivot_pd,
    aspect="auto"
)

plt.colorbar(label="Nº jugadores")

plt.xticks(
    ticks=range(len(pivot_pd.columns)),
    labels=x_labels,
    rotation=35,
    ha="right"
)

plt.yticks(
    ticks=range(len(pivot_pd.index)),
    labels=y_labels
)

# Añadir números dentro de cada celda
for i in range(pivot_pd.shape[0]):
    for j in range(pivot_pd.shape[1]):
        value = int(pivot_pd.iloc[i, j])
        plt.text(
            j,
            i,
            value,
            ha="center",
            va="center"
        )

plt.xlabel("Perfil Anthropometry")
plt.ylabel("Perfil GPS")
plt.title("Coincidencia de jugadores entre perfiles GPS y Anthropometry")

plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# TRANSICIÓN DOMINANTE GPS → ANTHROPOMETRY
# =========================

from pyspark.sql import functions as F
import pandas as pd

ID_COL = "id_unico_deportista"

# Cambia estos nombres por los reales de tus columnas
CLUSTER_GPS = "cluster_gps"
CLUSTER_ANTHRO = "cluster_anthropometry"


# Conteo por combinación
df_transition_gps_anthro = (
    df_gps_anthro
    .groupBy("cluster_gps", "cluster_anthropometry")
    .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
)

# Total por cluster GPS
df_total_gps = (
    df_gps_anthro
    .groupBy("cluster_gps")
    .agg(F.countDistinct(ID_COL).alias("n_total_gps"))
)

# Añadir porcentaje dentro de cada cluster GPS
df_transition_gps_anthro_pct = (
    df_transition_gps_anthro
    .join(df_total_gps, on="cluster_gps", how="left")
    .withColumn(
        "pct_dentro_gps",
        F.round(F.col("n_jugadores") / F.col("n_total_gps") * 100, 2)
    )
    .orderBy("cluster_gps", F.desc("n_jugadores"))
)

display(df_transition_gps_anthro_pct)

# COMMAND ----------

from pyspark.sql.window import Window

w = Window.partitionBy("cluster_gps").orderBy(F.desc("n_jugadores"))

df_dominant_gps_anthro = (
    df_transition_gps_anthro_pct
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("cluster_gps")
)

display(df_dominant_gps_anthro)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cluster más común de anthropometry por cada cluster de GPS

# COMMAND ----------


gps_cluster_names = {
    0: "GPS 0 | Perfil medio/positivo",
    1: "GPS 1 | Baja carga externa",
    2: "GPS 2 | Alta carga e intensidad",
    3: "GPS 3 | Perfil intermedio"
}

anthro_cluster_names = {
    0: "Anthro 0 | Robusto/muscular",
    1: "Anthro 1 | Mayor adiposidad",
    2: "Anthro 2 | Menor estructura corporal",
    3: "Anthro 3 | Mayor estructura corporal"
}

dominant_gps_anthro_pd = df_dominant_gps_anthro.toPandas()

dominant_gps_anthro_pd["perfil_gps"] = dominant_gps_anthro_pd["cluster_gps"].map(gps_cluster_names)
dominant_gps_anthro_pd["perfil_anthropometry"] = dominant_gps_anthro_pd["cluster_anthropometry"].map(anthro_cluster_names)

dominant_gps_anthro_pd = dominant_gps_anthro_pd[
    [
        "perfil_gps",
        "perfil_anthropometry",
        "n_jugadores",
        "pct_dentro_gps"
    ]
]

display(dominant_gps_anthro_pd)

# COMMAND ----------

# =========================
# GPS MÁS COMÚN DENTRO DE CADA CLUSTER ANTHROPOMETRY
# =========================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

ID_COL = "id_unico_deportista"

CLUSTER_GPS = "cluster_gps"
CLUSTER_ANTHRO = "cluster_anthropometry"

# 2. Conteo por combinación Anthropometry x GPS
df_transition_anthro_gps = (
    df_gps_anthro
    .groupBy("cluster_anthropometry", "cluster_gps")
    .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
)

# 3. Total por cluster Anthropometry
df_total_anthro = (
    df_gps_anthro
    .groupBy("cluster_anthropometry")
    .agg(F.countDistinct(ID_COL).alias("n_total_anthropometry"))
)

# 4. Añadir porcentaje dentro de cada cluster Anthropometry
df_transition_anthro_gps_pct = (
    df_transition_anthro_gps
    .join(df_total_anthro, on="cluster_anthropometry", how="left")
    .withColumn(
        "pct_dentro_anthropometry",
        F.round(F.col("n_jugadores") / F.col("n_total_anthropometry") * 100, 2)
    )
    .orderBy("cluster_anthropometry", F.desc("n_jugadores"))
)

display(df_transition_anthro_gps_pct)

# COMMAND ----------

# =========================
# GPS DOMINANTE POR CADA CLUSTER ANTHROPOMETRY
# =========================

w = Window.partitionBy("cluster_anthropometry").orderBy(F.desc("n_jugadores"))

df_dominant_anthro_gps = (
    df_transition_anthro_gps_pct
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("cluster_anthropometry")
)

display(df_dominant_anthro_gps)

# COMMAND ----------

# =========================
# AÑADIR NOMBRES DE PERFIL
# =========================

anthro_cluster_names = {
    0: "Anthro 0 | Robusto/muscular",
    1: "Anthro 1 | Mayor adiposidad",
    2: "Anthro 2 | Menor estructura corporal",
    3: "Anthro 3 | Mayor estructura corporal"
}

gps_cluster_names = {
    0: "GPS 0 | Perfil medio/positivo",
    1: "GPS 1 | Baja carga externa",
    2: "GPS 2 | Alta carga e intensidad",
    3: "GPS 3 | Perfil intermedio"
}

dominant_anthro_gps_pd = df_dominant_anthro_gps.toPandas()

dominant_anthro_gps_pd["perfil_anthropometry"] = dominant_anthro_gps_pd["cluster_anthropometry"].map(anthro_cluster_names)
dominant_anthro_gps_pd["perfil_gps"] = dominant_anthro_gps_pd["cluster_gps"].map(gps_cluster_names)

dominant_anthro_gps_pd = dominant_anthro_gps_pd[
    [
        "perfil_anthropometry",
        "perfil_gps",
        "n_jugadores",
        "n_total_anthropometry",
        "pct_dentro_anthropometry"
    ]
]

display(dominant_anthro_gps_pd)

# COMMAND ----------

# =========================
# COMBINAR ANTHROPOMETRY CON ISOPRONE
# =========================

ID_COL = "id_unico_deportista"
CLUSTER_ANTHRO = "cluster_anthropometry"
CLUSTER_ISO = "cluster_isoprone"

df_anthro_iso = (
    df_anthropometry
    .select(ID_COL, F.col(CLUSTER_ANTHRO).alias("cluster_anthropometry"))
    .join(
        df_isoprone.select(ID_COL, F.col(CLUSTER_ISO).alias("cluster_isoprone")),
        on=ID_COL,
        how="inner"
    )
)

print("Jugadores en ambas tablas:", df_anthro_iso.count())

display(df_anthro_iso)

# COMMAND ----------

# =========================
# HEATMAP ANTHROPOMETRY x ISOPRONE
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

# Nombres interpretables de los clusters
anthro_cluster_names = {
    0: "Anthro 0 | Robusto/muscular",
    1: "Anthro 1 | Mayor adiposidad",
    2: "Anthro 2 | Menor estructura",
    3: "Anthro 3 | Mayor estructura"
}

iso_cluster_names = {
    0: "Iso 0 | Baja fuerza + mayor asimetría",
    1: "Iso 1 | Perfil medio-alto equilibrado",
    2: "Iso 2 | Alta fuerza + baja asimetría",
    3: "Iso 3 | Perfil intermedio + baja asimetría"
}

# Tabla de transición
transition_anthro_iso_pd = (
    df_anthro_iso
    .groupBy("cluster_anthropometry", "cluster_isoprone")
    .agg(F.count("*").alias("n_jugadores"))
    .orderBy("cluster_anthropometry", "cluster_isoprone")
    .toPandas()
)

pivot_anthro_iso_pd = transition_anthro_iso_pd.pivot(
    index="cluster_anthropometry",
    columns="cluster_isoprone",
    values="n_jugadores"
).fillna(0)

display(pivot_anthro_iso_pd)

# Etiquetas interpretables
x_labels = [
    iso_cluster_names.get(int(c), f"Isoprone {c}")
    for c in pivot_anthro_iso_pd.columns
]

y_labels = [
    anthro_cluster_names.get(int(i), f"Anthro {i}")
    for i in pivot_anthro_iso_pd.index
]

# Heatmap
plt.figure(figsize=(12, 6))

plt.imshow(
    pivot_anthro_iso_pd,
    aspect="auto"
)

plt.colorbar(label="Nº jugadores")

plt.xticks(
    ticks=range(len(pivot_anthro_iso_pd.columns)),
    labels=x_labels,
    rotation=35,
    ha="right"
)

plt.yticks(
    ticks=range(len(pivot_anthro_iso_pd.index)),
    labels=y_labels
)

# Añadir valores dentro de cada celda
for i in range(pivot_anthro_iso_pd.shape[0]):
    for j in range(pivot_anthro_iso_pd.shape[1]):
        value = int(pivot_anthro_iso_pd.iloc[i, j])
        plt.text(
            j,
            i,
            value,
            ha="center",
            va="center"
        )

plt.xlabel("Perfil Isoprone")
plt.ylabel("Perfil Anthropometry")
plt.title("Coincidencia de jugadores entre perfiles Anthropometry e Isoprone")

plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# HEATMAP PORCENTUAL ANTHROPOMETRY → ISOPRONE
# =========================

pivot_pct_anthro_iso_pd = (
    pivot_anthro_iso_pd
    .div(pivot_anthro_iso_pd.sum(axis=1), axis=0)
    * 100
)

display(pivot_pct_anthro_iso_pd.round(2))

plt.figure(figsize=(12, 6))

plt.imshow(
    pivot_pct_anthro_iso_pd,
    aspect="auto",
    vmin=0,
    vmax=100
)

plt.colorbar(label="% dentro del perfil Anthropometry")

plt.xticks(
    ticks=range(len(pivot_pct_anthro_iso_pd.columns)),
    labels=x_labels,
    rotation=35,
    ha="right"
)

plt.yticks(
    ticks=range(len(pivot_pct_anthro_iso_pd.index)),
    labels=y_labels
)

# Añadir porcentajes dentro de cada celda
for i in range(pivot_pct_anthro_iso_pd.shape[0]):
    for j in range(pivot_pct_anthro_iso_pd.shape[1]):
        value = pivot_pct_anthro_iso_pd.iloc[i, j]
        plt.text(
            j,
            i,
            f"{value:.1f}%",
            ha="center",
            va="center"
        )

plt.xlabel("Perfil Isoprone")
plt.ylabel("Perfil Anthropometry")
plt.title("Distribución porcentual Anthropometry → Isoprone")

plt.tight_layout()
plt.show()

# COMMAND ----------

df_isoprone = spark.table("dev_gold.pitchvision.clustersresults_isoprone") \
    .select("id_unico_deportista", "cluster_PCA3_K4") \
    .withColumnRenamed("cluster_PCA3_K4", "cluster_isoprone")

df_nordic = spark.table("dev_gold.pitchvision.clustersresults_nordic") \
    .select("id_unico_deportista", "cluster_PCA4_K4") \
    .withColumnRenamed("cluster_PCA4_K4", "cluster_nordic")

df_gps = spark.table("dev_gold.pitchvision.clustersresults_gps") \
    .select("id_unico_deportista", "cluster_umap_k4") \
    .withColumnRenamed("cluster_umap_k4", "cluster_gps")

df_anthropometry = spark.table("dev_gold.pitchvision.resultsanthropometry") \
    .select("id_unico_deportista", "cluster_umap_2_k4") \
    .withColumnRenamed("cluster_umap_2_k4", "cluster_anthropometry")

df_cmj = spark.table("dev_gold.pitchvision.clustersresults_cmj") \
    .select("id_unico_deportista", "cluster_PCA4_K4") \
    .withColumnRenamed("cluster_PCA4_K4", "cluster_cmj")

df_slj = spark.table("dev_gold.pitchvision.clustersresults_slj") \
    .select("id_unico_deportista", "cluster_PCA2_K4") \
    .withColumnRenamed("cluster_PCA2_K4", "cluster_slj")

# COMMAND ----------

from pyspark.sql import functions as F
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

ID_COL = "id_unico_deportista"

# DataFrames individuales ya cargados
dfs_cluster = {
    "isoprone": df_isoprone,
    "nordic": df_nordic,
    "gps": df_gps,
    "anthropometry": df_anthropometry,
    "cmj": df_cmj,
    "slj": df_slj
}

# Columna de cluster dentro de cada df
cluster_cols = {
    "isoprone": "cluster_isoprone",
    "nordic": "cluster_nordic",
    "gps": "cluster_gps",
    "anthropometry": "cluster_anthropometry",
    "cmj": "cluster_cmj",
    "slj": "cluster_slj"
}

profile_names = {
    "isoprone": {
        0: "Iso 0 | Baja fuerza + mayor asimetría",
        1: "Iso 1 | Perfil medio-alto equilibrado",
        2: "Iso 2 | Alta fuerza + baja asimetría",
        3: "Iso 3 | Perfil intermedio + baja asimetría"
    },
    "nordic": {
        0: "Nordic 0 | Perfil medio",
        1: "Nordic 1 | Alta fuerza",
        2: "Nordic 2 | Baja fuerza",
        3: "Nordic 3 | Alta asimetría"
    },
    "gps": {
        0: "GPS 0 | Medio/positivo",
        1: "GPS 1 | Baja carga externa",
        2: "GPS 2 | Alta carga/intensidad",
        3: "GPS 3 | Intermedio"
    },
    "anthropometry": {
        0: "Anthro 0 | Robusto/muscular",
        1: "Anthro 1 | Mayor adiposidad",
        2: "Anthro 2 | Menor estructura",
        3: "Anthro 3 | Mayor estructura"
    },
    "cmj": {
        0: "CMJ 0 | Explosivo/potente",
        1: "CMJ 1 | Perfil equilibrado-alto",
        2: "CMJ 2 | Bajo rendimiento",
        3: "CMJ 3 | Mayor tiempo de contracción"
    },
    "slj": {
        0: "SLJ 0 | Medio-alto",
        1: "SLJ 1 | Bajo rendimiento",
        2: "SLJ 2 | Medio-bajo",
        3: "SLJ 3 | Alto rendimiento"
    }
}

# COMMAND ----------

def plot_pair_heatmap(table_a, table_b, percent=False):
    """
    Hace join solo entre dos tablas y representa el heatmap de coincidencia.
    percent=False -> nº jugadores
    percent=True  -> % dentro del cluster de table_a
    """
    
    df_a = dfs_cluster[table_a]
    df_b = dfs_cluster[table_b]
    
    col_a = cluster_cols[table_a]
    col_b = cluster_cols[table_b]
    
    df_pair = (
        df_a
        .select(ID_COL, F.col(col_a).alias("cluster_a"))
        .join(
            df_b.select(ID_COL, F.col(col_b).alias("cluster_b")),
            on=ID_COL,
            how="inner"
        )
        .dropna(subset=["cluster_a", "cluster_b"])
    )
    
    n_players = df_pair.select(ID_COL).distinct().count()
    print(f"{table_a} vs {table_b} | jugadores en ambas tablas: {n_players}")
    
    transition_pd = (
        df_pair
        .groupBy("cluster_a", "cluster_b")
        .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
        .orderBy("cluster_a", "cluster_b")
        .toPandas()
    )
    
    pivot_pd = transition_pd.pivot(
        index="cluster_a",
        columns="cluster_b",
        values="n_jugadores"
    ).fillna(0)
    
    pivot_pd = pivot_pd.sort_index().sort_index(axis=1)
    
    if percent:
        plot_data = pivot_pd.div(pivot_pd.sum(axis=1), axis=0) * 100
        colorbar_label = f"% dentro de {table_a}"
        title = f"Distribución porcentual {table_a} → {table_b}"
        vmin, vmax = 0, 100
        text_format = "{:.0f}%"
    else:
        plot_data = pivot_pd
        colorbar_label = "Nº jugadores"
        title = f"Coincidencia de jugadores: {table_a} vs {table_b}"
        vmin, vmax = None, None
        text_format = "{:.0f}"
    
    x_labels = [
        profile_names.get(table_b, {}).get(int(c), f"{table_b} {c}")
        for c in plot_data.columns
    ]
    
    y_labels = [
        profile_names.get(table_a, {}).get(int(i), f"{table_a} {i}")
        for i in plot_data.index
    ]
    
    plt.figure(figsize=(12, 6))
    
    plt.imshow(
        plot_data,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )
    
    plt.colorbar(label=colorbar_label)
    
    plt.xticks(
        ticks=range(len(plot_data.columns)),
        labels=x_labels,
        rotation=35,
        ha="right"
    )
    
    plt.yticks(
        ticks=range(len(plot_data.index)),
        labels=y_labels
    )
    
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            value = plot_data.iloc[i, j]
            plt.text(
                j,
                i,
                text_format.format(value),
                ha="center",
                va="center"
            )
    
    plt.xlabel(f"Perfil {table_b}")
    plt.ylabel(f"Perfil {table_a}")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return df_pair, pivot_pd

# COMMAND ----------

from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

pairs = list(combinations(dfs_cluster.keys(), 2))
pair_groups = np.array_split(pairs, 3)

pair_group = list(pair_groups[0])

percent = True  # cambia a False si quieres nº jugadores

n_cols = 2
n_rows = math.ceil(len(pair_group) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(9 * n_cols, 6 * n_rows)
)

if n_rows == 1:
    axes = np.array([axes])

axes_flat = axes.flatten()

for idx, (table_a, table_b) in enumerate(pair_group):
    
    ax = axes_flat[idx]
    
    df_a = dfs_cluster[table_a]
    df_b = dfs_cluster[table_b]
    
    col_a = cluster_cols[table_a]
    col_b = cluster_cols[table_b]
    
    df_pair = (
        df_a
        .select(ID_COL, F.col(col_a).alias("cluster_a"))
        .join(
            df_b.select(ID_COL, F.col(col_b).alias("cluster_b")),
            on=ID_COL,
            how="inner"
        )
        .dropna(subset=["cluster_a", "cluster_b"])
    )
    
    transition_pd = (
        df_pair
        .groupBy("cluster_a", "cluster_b")
        .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
        .orderBy("cluster_a", "cluster_b")
        .toPandas()
    )
    
    pivot_pd = transition_pd.pivot(
        index="cluster_a",
        columns="cluster_b",
        values="n_jugadores"
    ).fillna(0)
    
    pivot_pd = pivot_pd.sort_index().sort_index(axis=1)
    
    if percent:
        plot_data = pivot_pd.div(pivot_pd.sum(axis=1), axis=0) * 100
        vmin, vmax = 0, 100
        label_fmt = "{:.0f}%"
    else:
        plot_data = pivot_pd
        vmin, vmax = None, None
        label_fmt = "{:.0f}"
    
    ax.imshow(plot_data, aspect="auto", vmin=vmin, vmax=vmax)
    
    x_labels = [
        profile_names.get(table_b, {}).get(int(c), f"{table_b} {c}")
        for c in plot_data.columns
    ]
    
    y_labels = [
        profile_names.get(table_a, {}).get(int(i), f"{table_a} {i}")
        for i in plot_data.index
    ]
    
    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlabel(table_b)
    ax.set_ylabel(table_a)
    ax.set_title(f"{table_a} → {table_b}", fontsize=11)
    
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            ax.text(
                j,
                i,
                label_fmt.format(plot_data.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=8
            )

for j in range(len(pair_group), len(axes_flat)):
    axes_flat[j].axis("off")

plt.suptitle(
    "Coincidencia entre perfiles por pares de tablas | Parte 1/3",
    fontsize=16,
    y=1.02
)

plt.tight_layout()
plt.show()

# COMMAND ----------

from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

pairs = list(combinations(dfs_cluster.keys(), 2))
pair_groups = np.array_split(pairs, 3)

pair_group = list(pair_groups[1])

percent = True  # cambia a False si quieres nº jugadores

n_cols = 2
n_rows = math.ceil(len(pair_group) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(9 * n_cols, 6 * n_rows)
)

if n_rows == 1:
    axes = np.array([axes])

axes_flat = axes.flatten()

for idx, (table_a, table_b) in enumerate(pair_group):
    
    ax = axes_flat[idx]
    
    df_a = dfs_cluster[table_a]
    df_b = dfs_cluster[table_b]
    
    col_a = cluster_cols[table_a]
    col_b = cluster_cols[table_b]
    
    df_pair = (
        df_a
        .select(ID_COL, F.col(col_a).alias("cluster_a"))
        .join(
            df_b.select(ID_COL, F.col(col_b).alias("cluster_b")),
            on=ID_COL,
            how="inner"
        )
        .dropna(subset=["cluster_a", "cluster_b"])
    )
    
    transition_pd = (
        df_pair
        .groupBy("cluster_a", "cluster_b")
        .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
        .orderBy("cluster_a", "cluster_b")
        .toPandas()
    )
    
    pivot_pd = transition_pd.pivot(
        index="cluster_a",
        columns="cluster_b",
        values="n_jugadores"
    ).fillna(0)
    
    pivot_pd = pivot_pd.sort_index().sort_index(axis=1)
    
    if percent:
        plot_data = pivot_pd.div(pivot_pd.sum(axis=1), axis=0) * 100
        vmin, vmax = 0, 100
        label_fmt = "{:.0f}%"
    else:
        plot_data = pivot_pd
        vmin, vmax = None, None
        label_fmt = "{:.0f}"
    
    ax.imshow(plot_data, aspect="auto", vmin=vmin, vmax=vmax)
    
    x_labels = [
        profile_names.get(table_b, {}).get(int(c), f"{table_b} {c}")
        for c in plot_data.columns
    ]
    
    y_labels = [
        profile_names.get(table_a, {}).get(int(i), f"{table_a} {i}")
        for i in plot_data.index
    ]
    
    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlabel(table_b)
    ax.set_ylabel(table_a)
    ax.set_title(f"{table_a} → {table_b}", fontsize=11)
    
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            ax.text(
                j,
                i,
                label_fmt.format(plot_data.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=8
            )

for j in range(len(pair_group), len(axes_flat)):
    axes_flat[j].axis("off")

plt.suptitle(
    "Coincidencia entre perfiles por pares de tablas | Parte 2/3",
    fontsize=16,
    y=1.02
)

plt.tight_layout()
plt.show()

# COMMAND ----------

from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

pairs = list(combinations(dfs_cluster.keys(), 2))
pair_groups = np.array_split(pairs, 3)

pair_group = list(pair_groups[2])

percent = True  # cambia a False si quieres nº jugadores

n_cols = 2
n_rows = math.ceil(len(pair_group) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(9 * n_cols, 6 * n_rows)
)

if n_rows == 1:
    axes = np.array([axes])

axes_flat = axes.flatten()

for idx, (table_a, table_b) in enumerate(pair_group):
    
    ax = axes_flat[idx]
    
    df_a = dfs_cluster[table_a]
    df_b = dfs_cluster[table_b]
    
    col_a = cluster_cols[table_a]
    col_b = cluster_cols[table_b]
    
    df_pair = (
        df_a
        .select(ID_COL, F.col(col_a).alias("cluster_a"))
        .join(
            df_b.select(ID_COL, F.col(col_b).alias("cluster_b")),
            on=ID_COL,
            how="inner"
        )
        .dropna(subset=["cluster_a", "cluster_b"])
    )
    
    transition_pd = (
        df_pair
        .groupBy("cluster_a", "cluster_b")
        .agg(F.countDistinct(ID_COL).alias("n_jugadores"))
        .orderBy("cluster_a", "cluster_b")
        .toPandas()
    )
    
    pivot_pd = transition_pd.pivot(
        index="cluster_a",
        columns="cluster_b",
        values="n_jugadores"
    ).fillna(0)
    
    pivot_pd = pivot_pd.sort_index().sort_index(axis=1)
    
    if percent:
        plot_data = pivot_pd.div(pivot_pd.sum(axis=1), axis=0) * 100
        vmin, vmax = 0, 100
        label_fmt = "{:.0f}%"
    else:
        plot_data = pivot_pd
        vmin, vmax = None, None
        label_fmt = "{:.0f}"
    
    ax.imshow(plot_data, aspect="auto", vmin=vmin, vmax=vmax)
    
    x_labels = [
        profile_names.get(table_b, {}).get(int(c), f"{table_b} {c}")
        for c in plot_data.columns
    ]
    
    y_labels = [
        profile_names.get(table_a, {}).get(int(i), f"{table_a} {i}")
        for i in plot_data.index
    ]
    
    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlabel(table_b)
    ax.set_ylabel(table_a)
    ax.set_title(f"{table_a} → {table_b}", fontsize=11)
    
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            ax.text(
                j,
                i,
                label_fmt.format(plot_data.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=8
            )

for j in range(len(pair_group), len(axes_flat)):
    axes_flat[j].axis("off")

plt.suptitle(
    "Coincidencia entre perfiles por pares de tablas | Parte 3/3",
    fontsize=16,
    y=1.02
)

plt.tight_layout()
plt.show()

# COMMAND ----------


