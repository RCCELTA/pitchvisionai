# Databricks notebook source
# MAGIC %md
# MAGIC ### Anthropometry

# COMMAND ----------

# DBTITLE 1,Carga datos Anthropometry
df_anthro = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry")
df_cluster_interp_anthro = df_anthro
display(df_anthro)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Anthropometry)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_ANTHRO = "cluster_umap_2_k2"
ID_COL_ANTHRO = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_anthro = re.search(r'[Kk](\d+)', CLUSTER_COL_ANTHRO)
K_ANTHRO = int(_k_match_anthro.group(1)) if _k_match_anthro else None

_pca_match_anthro = re.search(r'PCA(\d+)', CLUSTER_COL_ANTHRO, re.IGNORECASE)
N_COMPONENTS_ANTHRO = int(_pca_match_anthro.group(1)) if _pca_match_anthro else None

# Excluir todas las columnas de cluster
all_cluster_cols_anthro = [
    f.name for f in df_cluster_interp_anthro.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_anthro = list(set([
    ID_COL_ANTHRO,
    CLUSTER_COL_ANTHRO,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_anthro))

features_interpretacion_anthro = [
    field.name
    for field in df_cluster_interp_anthro.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_anthro
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_ANTHRO}  |  K={K_ANTHRO}  |  N_COMPONENTS={N_COMPONENTS_ANTHRO}")
print(f"Nº variables para interpretar: {len(features_interpretacion_anthro)}")
for c in features_interpretacion_anthro:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Anthropometry)
df_z_anthro = df_cluster_interp_anthro

for col_name in features_interpretacion_anthro:
    stats = (
        df_cluster_interp_anthro
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_anthro = df_z_anthro.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_anthro = df_z_anthro.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_anthro = [f"{c}_z" for c in features_interpretacion_anthro]

agg_exprs_anthro = [F.count("*").alias("n_jugadores")]
for c in features_z_anthro:
    agg_exprs_anthro.append(F.avg(c).alias(c))

df_cluster_z_means_anthro = (
    df_z_anthro
    .groupBy(CLUSTER_COL_ANTHRO)
    .agg(*agg_exprs_anthro)
    .orderBy(CLUSTER_COL_ANTHRO)
)

display(df_cluster_z_means_anthro)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Anthropometry)
cluster_z_pd_anthro = df_cluster_z_means_anthro.toPandas()

z_cols_anthro = [c for c in cluster_z_pd_anthro.columns if c.endswith("_z")]

importance_rows_anthro = []

for col in z_cols_anthro:
    values = cluster_z_pd_anthro[col]
    importance_rows_anthro.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_anthro = (
    pd.DataFrame(importance_rows_anthro)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_anthro.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Anthropometry)
TOP_N_ANTHRO = 30

top_vars_anthro = importance_pd_anthro.head(TOP_N_ANTHRO)["variable"].tolist()
top_z_cols_anthro = [f"{v}_z" for v in top_vars_anthro]

heatmap_data_anthro = (
    cluster_z_pd_anthro
    .set_index(CLUSTER_COL_ANTHRO)[top_z_cols_anthro]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_anthro,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_anthro)),
    labels=top_vars_anthro,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_anthro.index)),
    labels=heatmap_data_anthro.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_ANTHRO is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_ANTHRO}) + KMeans(k={K_ANTHRO})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_ANTHRO} (k={K_ANTHRO})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPS

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### CMJ

# COMMAND ----------

# DBTITLE 1,Carga datos CMJ
df_cmj = spark.read.table("dev_gold.pitchvision.clustersresults_cmj")
df_cluster_interp_cmj = df_cmj
display(df_cmj)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (CMJ)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_CMJ = "cluster_UMAP5_K2_nn5_md0.0"
ID_COL_CMJ = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_cmj = re.search(r'[Kk](\d+)', CLUSTER_COL_CMJ)
K_CMJ = int(_k_match_cmj.group(1)) if _k_match_cmj else None

_pca_match_cmj = re.search(r'PCA(\d+)', CLUSTER_COL_CMJ, re.IGNORECASE)
N_COMPONENTS_CMJ = int(_pca_match_cmj.group(1)) if _pca_match_cmj else None

# Excluir todas las columnas de cluster
all_cluster_cols_cmj = [
    f.name for f in df_cluster_interp_cmj.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_cmj = list(set([
    ID_COL_CMJ,
    CLUSTER_COL_CMJ,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_cmj))

features_interpretacion_cmj = [
    field.name
    for field in df_cluster_interp_cmj.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cmj
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_CMJ}  |  K={K_CMJ}  |  N_COMPONENTS={N_COMPONENTS_CMJ}")
print(f"Nº variables para interpretar: {len(features_interpretacion_cmj)}")
for c in features_interpretacion_cmj:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (CMJ)
df_z_cmj = df_cluster_interp_cmj

for col_name in features_interpretacion_cmj:
    stats = (
        df_cluster_interp_cmj
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_cmj = df_z_cmj.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_cmj = df_z_cmj.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_cmj = [f"{c}_z" for c in features_interpretacion_cmj]

agg_exprs_cmj = [F.count("*").alias("n_jugadores")]
for c in features_z_cmj:
    agg_exprs_cmj.append(F.avg(c).alias(c))

df_cluster_z_means_cmj = (
    df_z_cmj
    .groupBy(F.col(f"`{CLUSTER_COL_CMJ}`"))
    .agg(*agg_exprs_cmj)
    .orderBy(F.col(f"`{CLUSTER_COL_CMJ}`"))
)

display(df_cluster_z_means_cmj)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (CMJ)
cluster_z_pd_cmj = df_cluster_z_means_cmj.toPandas()

z_cols_cmj = [c for c in cluster_z_pd_cmj.columns if c.endswith("_z")]

importance_rows_cmj = []

for col in z_cols_cmj:
    values = cluster_z_pd_cmj[col]
    importance_rows_cmj.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_cmj = (
    pd.DataFrame(importance_rows_cmj)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_cmj.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (CMJ)
TOP_N_CMJ = 30

top_vars_cmj = importance_pd_cmj.head(TOP_N_CMJ)["variable"].tolist()
top_z_cols_cmj = [f"{v}_z" for v in top_vars_cmj]

heatmap_data_cmj = (
    cluster_z_pd_cmj
    .set_index(CLUSTER_COL_CMJ)[top_z_cols_cmj]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_cmj,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_cmj)),
    labels=top_vars_cmj,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_cmj.index)),
    labels=heatmap_data_cmj.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_CMJ is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_CMJ}) + KMeans(k={K_CMJ})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_CMJ} (k={K_CMJ})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Isoprone

# COMMAND ----------

# DBTITLE 1,Carga datos Isoprone
df_isoprone = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone")
df_cluster_interp_isoprone = df_isoprone
display(df_isoprone)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Isoprone)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_ISOPRONE = "umap_k7_best"
ID_COL_ISOPRONE = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_isoprone = re.search(r'[Kk](\d+)', CLUSTER_COL_ISOPRONE)
K_ISOPRONE = int(_k_match_isoprone.group(1)) if _k_match_isoprone else None

_pca_match_isoprone = re.search(r'PCA(\d+)', CLUSTER_COL_ISOPRONE, re.IGNORECASE)
N_COMPONENTS_ISOPRONE = int(_pca_match_isoprone.group(1)) if _pca_match_isoprone else None

# Excluir todas las columnas de cluster
all_cluster_cols_isoprone = [
    f.name for f in df_cluster_interp_isoprone.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_isoprone = list(set([
    ID_COL_ISOPRONE,
    CLUSTER_COL_ISOPRONE,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_isoprone))

features_interpretacion_isoprone = [
    field.name
    for field in df_cluster_interp_isoprone.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_isoprone
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_ISOPRONE}  |  K={K_ISOPRONE}  |  N_COMPONENTS={N_COMPONENTS_ISOPRONE}")
print(f"Nº variables para interpretar: {len(features_interpretacion_isoprone)}")
for c in features_interpretacion_isoprone:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Isoprone)
df_z_isoprone = df_cluster_interp_isoprone

for col_name in features_interpretacion_isoprone:
    stats = (
        df_cluster_interp_isoprone
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_isoprone = df_z_isoprone.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_isoprone = df_z_isoprone.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_isoprone = [f"{c}_z" for c in features_interpretacion_isoprone]

agg_exprs_isoprone = [F.count("*").alias("n_jugadores")]
for c in features_z_isoprone:
    agg_exprs_isoprone.append(F.avg(c).alias(c))

df_cluster_z_means_isoprone = (
    df_z_isoprone
    .groupBy(CLUSTER_COL_ISOPRONE)
    .agg(*agg_exprs_isoprone)
    .orderBy(CLUSTER_COL_ISOPRONE)
)

display(df_cluster_z_means_isoprone)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Isoprone)
cluster_z_pd_isoprone = df_cluster_z_means_isoprone.toPandas()

z_cols_isoprone = [c for c in cluster_z_pd_isoprone.columns if c.endswith("_z")]

importance_rows_isoprone = []

for col in z_cols_isoprone:
    values = cluster_z_pd_isoprone[col]
    importance_rows_isoprone.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_isoprone = (
    pd.DataFrame(importance_rows_isoprone)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_isoprone.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Isoprone)
TOP_N_ISOPRONE = 30

top_vars_isoprone = importance_pd_isoprone.head(TOP_N_ISOPRONE)["variable"].tolist()
top_z_cols_isoprone = [f"{v}_z" for v in top_vars_isoprone]

heatmap_data_isoprone = (
    cluster_z_pd_isoprone
    .set_index(CLUSTER_COL_ISOPRONE)[top_z_cols_isoprone]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_isoprone,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_isoprone)),
    labels=top_vars_isoprone,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_isoprone.index)),
    labels=heatmap_data_isoprone.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_ISOPRONE is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_ISOPRONE}) + KMeans(k={K_ISOPRONE})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_ISOPRONE} (k={K_ISOPRONE})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nordic

# COMMAND ----------

# DBTITLE 1,Carga datos Nordic
df_nordic = spark.read.table("dev_gold.pitchvision.clustersresults_nordic")
df_cluster_interp_nordic = df_nordic
display(df_nordic)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Nordic)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_NORDIC = "cluster_PCA2_K5"
ID_COL_NORDIC = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_nordic = re.search(r'[Kk](\d+)', CLUSTER_COL_NORDIC)
K_NORDIC = int(_k_match_nordic.group(1)) if _k_match_nordic else None

_pca_match_nordic = re.search(r'PCA(\d+)', CLUSTER_COL_NORDIC, re.IGNORECASE)
N_COMPONENTS_NORDIC = int(_pca_match_nordic.group(1)) if _pca_match_nordic else None

# Excluir todas las columnas de cluster
all_cluster_cols_nordic = [
    f.name for f in df_cluster_interp_nordic.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_nordic = list(set([
    ID_COL_NORDIC,
    CLUSTER_COL_NORDIC,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_nordic))

features_interpretacion_nordic = [
    field.name
    for field in df_cluster_interp_nordic.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_nordic
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_NORDIC}  |  K={K_NORDIC}  |  N_COMPONENTS={N_COMPONENTS_NORDIC}")
print(f"Nº variables para interpretar: {len(features_interpretacion_nordic)}")
for c in features_interpretacion_nordic:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Nordic)
df_z_nordic = df_cluster_interp_nordic

for col_name in features_interpretacion_nordic:
    stats = (
        df_cluster_interp_nordic
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_nordic = df_z_nordic.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_nordic = df_z_nordic.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_nordic = [f"{c}_z" for c in features_interpretacion_nordic]

agg_exprs_nordic = [F.count("*").alias("n_jugadores")]
for c in features_z_nordic:
    agg_exprs_nordic.append(F.avg(c).alias(c))

df_cluster_z_means_nordic = (
    df_z_nordic
    .groupBy(CLUSTER_COL_NORDIC)
    .agg(*agg_exprs_nordic)
    .orderBy(CLUSTER_COL_NORDIC)
)

display(df_cluster_z_means_nordic)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Nordic)
cluster_z_pd_nordic = df_cluster_z_means_nordic.toPandas()

z_cols_nordic = [c for c in cluster_z_pd_nordic.columns if c.endswith("_z")]

importance_rows_nordic = []

for col in z_cols_nordic:
    values = cluster_z_pd_nordic[col]
    importance_rows_nordic.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_nordic = (
    pd.DataFrame(importance_rows_nordic)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_nordic.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Nordic)
TOP_N_NORDIC = 30

top_vars_nordic = importance_pd_nordic.head(TOP_N_NORDIC)["variable"].tolist()
top_z_cols_nordic = [f"{v}_z" for v in top_vars_nordic]

heatmap_data_nordic = (
    cluster_z_pd_nordic
    .set_index(CLUSTER_COL_NORDIC)[top_z_cols_nordic]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_nordic,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_nordic)),
    labels=top_vars_nordic,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_nordic.index)),
    labels=heatmap_data_nordic.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_NORDIC is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_NORDIC}) + KMeans(k={K_NORDIC})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_NORDIC} (k={K_NORDIC})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SLJ

# COMMAND ----------

# DBTITLE 1,Carga datos SLJ
df_slj = spark.read.table("dev_gold.pitchvision.clustersresults_slj")
df_cluster_interp_slj = df_slj
display(df_slj)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (SLJ)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_SLJ = "cluster_PCA3_K3"
ID_COL_SLJ = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_slj = re.search(r'[Kk](\d+)', CLUSTER_COL_SLJ)
K_SLJ = int(_k_match_slj.group(1)) if _k_match_slj else None

_pca_match_slj = re.search(r'PCA(\d+)', CLUSTER_COL_SLJ, re.IGNORECASE)
N_COMPONENTS_SLJ = int(_pca_match_slj.group(1)) if _pca_match_slj else None

# Excluir todas las columnas de cluster
all_cluster_cols_slj = [
    f.name for f in df_cluster_interp_slj.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_slj = list(set([
    ID_COL_SLJ,
    CLUSTER_COL_SLJ,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_slj))

features_interpretacion_slj = [
    field.name
    for field in df_cluster_interp_slj.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_slj
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_SLJ}  |  K={K_SLJ}  |  N_COMPONENTS={N_COMPONENTS_SLJ}")
print(f"Nº variables para interpretar: {len(features_interpretacion_slj)}")
for c in features_interpretacion_slj:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (SLJ)
df_z_slj = df_cluster_interp_slj

for col_name in features_interpretacion_slj:
    stats = (
        df_cluster_interp_slj
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_slj = df_z_slj.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_slj = df_z_slj.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_slj = [f"{c}_z" for c in features_interpretacion_slj]

agg_exprs_slj = [F.count("*").alias("n_jugadores")]
for c in features_z_slj:
    agg_exprs_slj.append(F.avg(c).alias(c))

df_cluster_z_means_slj = (
    df_z_slj
    .groupBy(CLUSTER_COL_SLJ)
    .agg(*agg_exprs_slj)
    .orderBy(CLUSTER_COL_SLJ)
)

display(df_cluster_z_means_slj)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (SLJ)
cluster_z_pd_slj = df_cluster_z_means_slj.toPandas()

z_cols_slj = [c for c in cluster_z_pd_slj.columns if c.endswith("_z")]

importance_rows_slj = []

for col in z_cols_slj:
    values = cluster_z_pd_slj[col]
    importance_rows_slj.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_slj = (
    pd.DataFrame(importance_rows_slj)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_slj.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (SLJ)
TOP_N_SLJ = 30

top_vars_slj = importance_pd_slj.head(TOP_N_SLJ)["variable"].tolist()
top_z_cols_slj = [f"{v}_z" for v in top_vars_slj]

heatmap_data_slj = (
    cluster_z_pd_slj
    .set_index(CLUSTER_COL_SLJ)[top_z_cols_slj]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_slj,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_slj)),
    labels=top_vars_slj,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_slj.index)),
    labels=heatmap_data_slj.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_SLJ is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_SLJ}) + KMeans(k={K_SLJ})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_SLJ} (k={K_SLJ})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado pruebas físicas

# COMMAND ----------

# DBTITLE 1,Carga datos Agregado pruebas físicas
df_fisicas = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas")
df_cluster_interp_fisicas = df_fisicas
display(df_fisicas)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Agregado físicas)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_FISICAS = "UMAP_best_K2"
ID_COL_FISICAS = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_fisicas = re.search(r'[Kk](\d+)', CLUSTER_COL_FISICAS)
K_FISICAS = int(_k_match_fisicas.group(1)) if _k_match_fisicas else None

_pca_match_fisicas = re.search(r'PCA(\d+)', CLUSTER_COL_FISICAS, re.IGNORECASE)
N_COMPONENTS_FISICAS = int(_pca_match_fisicas.group(1)) if _pca_match_fisicas else None

# Excluir todas las columnas de cluster
all_cluster_cols_fisicas = [
    f.name for f in df_cluster_interp_fisicas.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_fisicas = list(set([
    ID_COL_FISICAS,
    CLUSTER_COL_FISICAS,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_fisicas))

features_interpretacion_fisicas = [
    field.name
    for field in df_cluster_interp_fisicas.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_fisicas
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_FISICAS}  |  K={K_FISICAS}  |  N_COMPONENTS={N_COMPONENTS_FISICAS}")
print(f"Nº variables para interpretar: {len(features_interpretacion_fisicas)}")
for c in features_interpretacion_fisicas:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agregado físicas)
df_z_fisicas = df_cluster_interp_fisicas

for col_name in features_interpretacion_fisicas:
    stats = (
        df_cluster_interp_fisicas
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_fisicas = df_z_fisicas.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_fisicas = df_z_fisicas.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_fisicas = [f"{c}_z" for c in features_interpretacion_fisicas]

agg_exprs_fisicas = [F.count("*").alias("n_jugadores")]
for c in features_z_fisicas:
    agg_exprs_fisicas.append(F.avg(c).alias(c))

df_cluster_z_means_fisicas = (
    df_z_fisicas
    .groupBy(CLUSTER_COL_FISICAS)
    .agg(*agg_exprs_fisicas)
    .orderBy(CLUSTER_COL_FISICAS)
)

display(df_cluster_z_means_fisicas)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agregado físicas)
cluster_z_pd_fisicas = df_cluster_z_means_fisicas.toPandas()

z_cols_fisicas = [c for c in cluster_z_pd_fisicas.columns if c.endswith("_z")]

importance_rows_fisicas = []

for col in z_cols_fisicas:
    values = cluster_z_pd_fisicas[col]
    importance_rows_fisicas.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_fisicas = (
    pd.DataFrame(importance_rows_fisicas)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_fisicas.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agregado físicas)
TOP_N_FISICAS = 30

top_vars_fisicas = importance_pd_fisicas.head(TOP_N_FISICAS)["variable"].tolist()
top_z_cols_fisicas = [f"{v}_z" for v in top_vars_fisicas]

heatmap_data_fisicas = (
    cluster_z_pd_fisicas
    .set_index(CLUSTER_COL_FISICAS)[top_z_cols_fisicas]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_fisicas,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_fisicas)),
    labels=top_vars_fisicas,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_fisicas.index)),
    labels=heatmap_data_fisicas.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_FISICAS is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_FISICAS}) + KMeans(k={K_FISICAS})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_FISICAS} (k={K_FISICAS})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado físicas + anthropometry

# COMMAND ----------

# DBTITLE 1,Carga datos Agregado físicas + Anthropometry
df_fisicas_anthro = spark.read.table("dev_gold.pitchvision.clustersresults_Agregadopruebasfisicas_anthropometry")
df_cluster_interp_fisicas_anthro = df_fisicas_anthro
display(df_fisicas_anthro)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Agregado físicas + Anthro)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_FISICAS_ANTHRO = "cluster_UMAP10_nn5_md00_K5"
ID_COL_FISICAS_ANTHRO = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_fisicas_anthro = re.search(r'[Kk](\d+)', CLUSTER_COL_FISICAS_ANTHRO)
K_FISICAS_ANTHRO = int(_k_match_fisicas_anthro.group(1)) if _k_match_fisicas_anthro else None

_pca_match_fisicas_anthro = re.search(r'PCA(\d+)', CLUSTER_COL_FISICAS_ANTHRO, re.IGNORECASE)
N_COMPONENTS_FISICAS_ANTHRO = int(_pca_match_fisicas_anthro.group(1)) if _pca_match_fisicas_anthro else None

# Excluir todas las columnas de cluster
all_cluster_cols_fisicas_anthro = [
    f.name for f in df_cluster_interp_fisicas_anthro.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_fisicas_anthro = list(set([
    ID_COL_FISICAS_ANTHRO,
    CLUSTER_COL_FISICAS_ANTHRO,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_fisicas_anthro))

features_interpretacion_fisicas_anthro = [
    field.name
    for field in df_cluster_interp_fisicas_anthro.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_fisicas_anthro
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_FISICAS_ANTHRO}  |  K={K_FISICAS_ANTHRO}  |  N_COMPONENTS={N_COMPONENTS_FISICAS_ANTHRO}")
print(f"Nº variables para interpretar: {len(features_interpretacion_fisicas_anthro)}")
for c in features_interpretacion_fisicas_anthro:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agregado físicas + Anthro)
df_z_fisicas_anthro = df_cluster_interp_fisicas_anthro

for col_name in features_interpretacion_fisicas_anthro:
    stats = (
        df_cluster_interp_fisicas_anthro
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_fisicas_anthro = df_z_fisicas_anthro.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_fisicas_anthro = df_z_fisicas_anthro.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_fisicas_anthro = [f"{c}_z" for c in features_interpretacion_fisicas_anthro]

agg_exprs_fisicas_anthro = [F.count("*").alias("n_jugadores")]
for c in features_z_fisicas_anthro:
    agg_exprs_fisicas_anthro.append(F.avg(c).alias(c))

df_cluster_z_means_fisicas_anthro = (
    df_z_fisicas_anthro
    .groupBy(CLUSTER_COL_FISICAS_ANTHRO)
    .agg(*agg_exprs_fisicas_anthro)
    .orderBy(CLUSTER_COL_FISICAS_ANTHRO)
)

display(df_cluster_z_means_fisicas_anthro)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agregado físicas + Anthro)
cluster_z_pd_fisicas_anthro = df_cluster_z_means_fisicas_anthro.toPandas()

z_cols_fisicas_anthro = [c for c in cluster_z_pd_fisicas_anthro.columns if c.endswith("_z")]

importance_rows_fisicas_anthro = []

for col in z_cols_fisicas_anthro:
    values = cluster_z_pd_fisicas_anthro[col]
    importance_rows_fisicas_anthro.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_fisicas_anthro = (
    pd.DataFrame(importance_rows_fisicas_anthro)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_fisicas_anthro.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agregado físicas + Anthro)
TOP_N_FISICAS_ANTHRO = 30

top_vars_fisicas_anthro = importance_pd_fisicas_anthro.head(TOP_N_FISICAS_ANTHRO)["variable"].tolist()
top_z_cols_fisicas_anthro = [f"{v}_z" for v in top_vars_fisicas_anthro]

heatmap_data_fisicas_anthro = (
    cluster_z_pd_fisicas_anthro
    .set_index(CLUSTER_COL_FISICAS_ANTHRO)[top_z_cols_fisicas_anthro]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_fisicas_anthro,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_fisicas_anthro)),
    labels=top_vars_fisicas_anthro,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_fisicas_anthro.index)),
    labels=heatmap_data_fisicas_anthro.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_FISICAS_ANTHRO is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_FISICAS_ANTHRO}) + KMeans(k={K_FISICAS_ANTHRO})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_FISICAS_ANTHRO} (k={K_FISICAS_ANTHRO})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado todo

# COMMAND ----------

# DBTITLE 1,Carga datos Agregado todo
df_todo = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo")
df_cluster_interp_todo = df_todo
display(df_todo)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features (Agregado todo)
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL_TODO = "UMAP_best_K6"
ID_COL_TODO = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match_todo = re.search(r'[Kk](\d+)', CLUSTER_COL_TODO)
K_TODO = int(_k_match_todo.group(1)) if _k_match_todo else None

_pca_match_todo = re.search(r'PCA(\d+)', CLUSTER_COL_TODO, re.IGNORECASE)
N_COMPONENTS_TODO = int(_pca_match_todo.group(1)) if _pca_match_todo else None

# Excluir todas las columnas de cluster
all_cluster_cols_todo = [
    f.name for f in df_cluster_interp_todo.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_todo = list(set([
    ID_COL_TODO,
    CLUSTER_COL_TODO,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols_todo))

features_interpretacion_todo = [
    field.name
    for field in df_cluster_interp_todo.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_todo
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_TODO}  |  K={K_TODO}  |  N_COMPONENTS={N_COMPONENTS_TODO}")
print(f"Nº variables para interpretar: {len(features_interpretacion_todo)}")
for c in features_interpretacion_todo:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agregado todo)
df_z_todo = df_cluster_interp_todo

for col_name in features_interpretacion_todo:
    stats = (
        df_cluster_interp_todo
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )

    mean_val = stats["mean"]
    std_val = stats["std"]

    if std_val is not None and std_val != 0:
        df_z_todo = df_z_todo.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_todo = df_z_todo.withColumn(f"{col_name}_z", F.lit(0.0))

features_z_todo = [f"{c}_z" for c in features_interpretacion_todo]

agg_exprs_todo = [F.count("*").alias("n_jugadores")]
for c in features_z_todo:
    agg_exprs_todo.append(F.avg(c).alias(c))

df_cluster_z_means_todo = (
    df_z_todo
    .groupBy(CLUSTER_COL_TODO)
    .agg(*agg_exprs_todo)
    .orderBy(CLUSTER_COL_TODO)
)

display(df_cluster_z_means_todo)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agregado todo)
cluster_z_pd_todo = df_cluster_z_means_todo.toPandas()

z_cols_todo = [c for c in cluster_z_pd_todo.columns if c.endswith("_z")]

importance_rows_todo = []

for col in z_cols_todo:
    values = cluster_z_pd_todo[col]
    importance_rows_todo.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd_todo = (
    pd.DataFrame(importance_rows_todo)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_todo.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agregado todo)
TOP_N_TODO = 30

top_vars_todo = importance_pd_todo.head(TOP_N_TODO)["variable"].tolist()
top_z_cols_todo = [f"{v}_z" for v in top_vars_todo]

heatmap_data_todo = (
    cluster_z_pd_todo
    .set_index(CLUSTER_COL_TODO)[top_z_cols_todo]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_todo,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_todo)),
    labels=top_vars_todo,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_todo.index)),
    labels=heatmap_data_todo.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS_TODO is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS_TODO}) + KMeans(k={K_TODO})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL_TODO} (k={K_TODO})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado pruebas nordic isoprone y gps

# COMMAND ----------

# DBTITLE 1,Cell 1
df = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps")
df_cluster_interp = df
display(df)

# COMMAND ----------

# DBTITLE 1,Configuración de columnas y features
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

CLUSTER_COL = "UMAP_best_K5"
ID_COL = "id_unico_deportista"

# Extraer K y N_COMPONENTS del nombre de CLUSTER_COL
_k_match = re.search(r'K(\d+)', CLUSTER_COL)
K = int(_k_match.group(1)) if _k_match else None

_pca_match = re.search(r'PCA(\d+)', CLUSTER_COL)
N_COMPONENTS = int(_pca_match.group(1)) if _pca_match else None

# Excluir todas las columnas de cluster (cluster_* y UMAP_*)
all_cluster_cols = [
    f.name for f in df_cluster_interp.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_")
]

cols_excluir_interp = list(set([
    ID_COL,
    CLUSTER_COL,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
] + all_cluster_cols))

features_interpretacion = [
    field.name
    for field in df_cluster_interp.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_interp
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL}  |  K={K}  |  N_COMPONENTS={N_COMPONENTS}")
print(f"Nº variables para interpretar: {len(features_interpretacion)}")
for c in features_interpretacion:
    print(c)

# COMMAND ----------

df_z = df_cluster_interp

for col_name in features_interpretacion:
    stats = (
        df_cluster_interp
        .agg(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        )
        .collect()[0]
    )
    
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    if std_val is not None and std_val != 0:
        df_z = df_z.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z = df_z.withColumn(f"{col_name}_z", F.lit(0.0))

features_z = [f"{c}_z" for c in features_interpretacion]

agg_exprs = [F.count("*").alias("n_jugadores")]

for c in features_z:
    agg_exprs.append(F.avg(c).alias(c))

df_cluster_z_means = (
    df_z
    .groupBy(CLUSTER_COL)
    .agg(*agg_exprs)
    .orderBy(CLUSTER_COL)
)

display(df_cluster_z_means)

# COMMAND ----------

cluster_z_pd = df_cluster_z_means.toPandas()

z_cols = [c for c in cluster_z_pd.columns if c.endswith("_z")]

importance_rows = []

for col in z_cols:
    values = cluster_z_pd[col]
    
    importance_rows.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_pd = (
    pd.DataFrame(importance_rows)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster
TOP_N = 30

top_vars = importance_pd.head(TOP_N)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = (
    cluster_z_pd
    .set_index(CLUSTER_COL)[top_z_cols]
)

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars)),
    labels=top_vars,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data.index)),
    labels=heatmap_data.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
if N_COMPONENTS is not None:
    _title = f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS}) + KMeans(k={K})"
else:
    _title = f"Perfil medio estandarizado por cluster | {CLUSTER_COL} (k={K})"
plt.title(_title)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Best_k4

# COMMAND ----------

# Obtener los nombres de columnas de la tabla original
cols_anthro_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry_limpio").columns
cols_anthro = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry").columns



# COMMAND ----------

# DBTITLE 1,Mapeo columnas limpio → original
import difflib

set_limpio = set(cols_anthro_limpio)
set_orig   = set(cols_anthro)

# Columnas exactamente iguales en ambas tablas
shared = sorted(set_limpio & set_orig)

# En limpio pero NO en original (columnas nuevas / renombradas)
only_limpio = sorted(set_limpio - set_orig)

# En original pero NO en limpio (columnas eliminadas / renombradas)
only_orig = sorted(set_orig - set_limpio)

print(f"Compartidas : {len(shared)}")
print(f"Solo limpio : {len(only_limpio)}")
print(f"Solo original: {len(only_orig)}")
print()

# Fuzzy match: para cada columna exclusiva de limpio, busca la columna
# original más cercana por similitud de nombre
print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
mapping = []
for col_l in only_limpio:
    matches = difflib.get_close_matches(col_l, only_orig, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    mapping.append({"col_limpio": col_l, "col_original": best, "score": round(score, 3)})
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original (posibles eliminadas):")
for c in only_orig:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4
# Columnas candidatas K=4 en la tabla original
candidatas_k4 = [c for c in cols_anthro if c.lower().endswith("_k4") or c.lower().endswith("k4")]
print("Candidatas K4 en original:", candidatas_k4)

# Unir ambas tablas por id_unico_deportista y comparar valores
id_col = "id_unico_deportista"

df_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry_limpio").select(id_col, "best_k4")
df_orig   = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry").select([id_col] + candidatas_k4)

df_joined = df_limpio.join(df_orig, on=id_col, how="inner")

# Para cada candidata, contar cuántas filas coinciden exactamente con best_k4
results = []
for c in candidatas_k4:
    n_total = df_joined.count()
    n_match = df_joined.filter(F.col("best_k4") == F.col(c)).count()
    results.append({"candidata": c, "n_total": n_total, "n_match": n_match, "pct_match": round(n_match / n_total * 100, 2) if n_total > 0 else 0})

import pandas as pd
results_pd = pd.DataFrame(results).sort_values("pct_match", ascending=False)
print(results_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 = cluster_umap_5_k4 (Anthropometry)
from pyspark.sql.types import NumericType

CLUSTER_COL_K4 = "cluster_umap_5_k4"
ID_COL_K4 = "id_unico_deportista"

df_anthro_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_anthropometry")

_k_match_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_K4)
K_K4 = int(_k_match_k4.group(1)) if _k_match_k4 else None

# Excluir columnas de cluster / UMAP
all_cluster_cols_k4 = [
    f.name for f in df_anthro_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_k4 = list(set([
    ID_COL_K4, CLUSTER_COL_K4,
    "temporada_fuente", "first_test_date", "last_test_date", "test_date", "age_group"
] + all_cluster_cols_k4))

features_k4 = [
    field.name
    for field in df_anthro_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_k4
    and not field.name.endswith("_age_scaled")
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_K4}  |  K={K_K4}")
print(f"Nº features: {len(features_k4)}")

# COMMAND ----------

# DBTITLE 1,Z-scores por cluster (Anthropometry best_k4)
# Calcular medias y std una sola vez con una sola pasada
stats_exprs = [e for col_name in features_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row = df_anthro_k4.agg(*stats_exprs).collect()[0]

# Construir todas las columnas z con withColumns en una sola llamada
z_exprs = {}
for col_name in features_k4:
    mean_val = stats_row[f"{col_name}__mean"]
    std_val  = stats_row[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs[f"{col_name}_z"] = F.lit(0.0)

df_z_k4 = df_anthro_k4.withColumns(z_exprs)

features_z_k4 = [f"{c}_z" for c in features_k4]
agg_exprs_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_k4]

df_cluster_z_means_k4 = (
    df_z_k4
    .groupBy(CLUSTER_COL_K4)
    .agg(*agg_exprs_k4)
    .orderBy(CLUSTER_COL_K4)
)

display(df_cluster_z_means_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Anthropometry best_k4)
cluster_z_pd_k4 = df_cluster_z_means_k4.toPandas()

importance_pd_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_k4[c].max() - cluster_z_pd_k4[c].min(),
        "mean_abs_z": cluster_z_pd_k4[c].abs().mean()
    } for c in features_z_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Anthropometry cluster_umap_5_k4)
TOP_N_K4 = 30

top_vars_k4     = importance_pd_k4.head(TOP_N_K4)["variable"].tolist()
top_z_cols_k4   = [f"{v}_z" for v in top_vars_k4]
heatmap_data_k4 = cluster_z_pd_k4.set_index(CLUSTER_COL_K4)[top_z_cols_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_k4)),
    labels=top_vars_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_k4.index)),
    labels=heatmap_data_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_K4} (k={K_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### CMJ

# COMMAND ----------

# DBTITLE 1,Carga columnas CMJ limpio vs original
cols_cmj_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_cmj_limpio").columns
cols_cmj       = spark.read.table("dev_gold.pitchvision.clustersresults_cmj").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas CMJ limpio → original
import difflib

set_limpio_cmj = set(cols_cmj_limpio)
set_orig_cmj   = set(cols_cmj)

shared_cmj    = sorted(set_limpio_cmj & set_orig_cmj)
only_limpio_cmj = sorted(set_limpio_cmj - set_orig_cmj)
only_orig_cmj   = sorted(set_orig_cmj - set_limpio_cmj)

print(f"Compartidas : {len(shared_cmj)}")
print(f"Solo limpio : {len(only_limpio_cmj)}")
print(f"Solo original: {len(only_orig_cmj)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_cmj:
    matches = difflib.get_close_matches(col_l, only_orig_cmj, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original (posibles eliminadas):")
for c in only_orig_cmj:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (CMJ)
candidatas_k4_cmj = [c for c in cols_cmj if c.lower().endswith("_k4") or c.lower().endswith("k4")]
print("Candidatas K4 en CMJ original:", candidatas_k4_cmj)

id_col = "id_unico_deportista"

df_limpio_cmj = spark.read.table("dev_gold.pitchvision.clustersresults_cmj_limpio").select(id_col, "best_k4")
df_orig_cmj   = spark.read.table("dev_gold.pitchvision.clustersresults_cmj").select([id_col] + candidatas_k4_cmj)

df_joined_cmj = df_limpio_cmj.join(df_orig_cmj, on=id_col, how="inner")

results_cmj = []
for c in candidatas_k4_cmj:
    n_total = df_joined_cmj.count()
    n_match = df_joined_cmj.filter(F.col("best_k4") == F.col(c)).count()
    results_cmj.append({
        "candidata": c,
        "n_total": n_total,
        "n_match": n_match,
        "pct_match": round(n_match / n_total * 100, 2) if n_total > 0 else 0
    })

results_cmj_pd = pd.DataFrame(results_cmj).sort_values("pct_match", ascending=False)
print(results_cmj_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Probar cluster_UMAP2_K4_nn5_md0.0 como origen de best_k4 (CMJ)
col_candidata = "cluster_UMAP2_K4_nn5_md0.0"
id_col = "id_unico_deportista"

df_limpio_cmj2 = spark.read.table("dev_gold.pitchvision.clustersresults_cmj_limpio").select(id_col, "best_k4")
df_orig_cmj2   = spark.read.table("dev_gold.pitchvision.clustersresults_cmj").select(id_col, F.col(f"`{col_candidata}`"))

df_joined_cmj2 = df_limpio_cmj2.join(df_orig_cmj2, on=id_col, how="inner")

n_total = df_joined_cmj2.count()
n_match = df_joined_cmj2.filter(F.col("best_k4") == F.col(f"`{col_candidata}`")).count()

print(f"Candidata : {col_candidata}")
print(f"Total filas: {n_total}")
print(f"Coincidencias exactas: {n_match}")
print(f"Porcentaje: {round(n_match / n_total * 100, 2)}%")

# COMMAND ----------

# DBTITLE 1,Config best_k4 = cluster_UMAP2_K4_nn5_md0.0 (CMJ)
from pyspark.sql.types import NumericType

CLUSTER_COL_CMJ_K4 = "cluster_UMAP2_K4_nn5_md0.0"
ID_COL_CMJ_K4 = "id_unico_deportista"

df_cmj_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_cmj")

_k_match_cmj_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_CMJ_K4)
K_CMJ_K4 = int(_k_match_cmj_k4.group(1)) if _k_match_cmj_k4 else None

# Excluir columnas de cluster / UMAP
all_cluster_cols_cmj_k4 = [
    f.name for f in df_cmj_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_cmj_k4 = list(set([
    ID_COL_CMJ_K4, CLUSTER_COL_CMJ_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_cmj_k4))

features_cmj_k4 = [
    field.name
    for field in df_cmj_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cmj_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_CMJ_K4}  |  K={K_CMJ_K4}")
print(f"Nº features: {len(features_cmj_k4)}")
for c in features_cmj_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (CMJ best_k4)
# Calcular medias y std en una sola pasada
stats_exprs_cmj_k4 = [e for col_name in features_cmj_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_cmj_k4 = df_cmj_k4.agg(*stats_exprs_cmj_k4).collect()[0]

# Construir columnas z con withColumns en una sola llamada
z_exprs_cmj_k4 = {}
for col_name in features_cmj_k4:
    mean_val = stats_row_cmj_k4[f"{col_name}__mean"]
    std_val  = stats_row_cmj_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_cmj_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_cmj_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_cmj_k4 = df_cmj_k4.withColumns(z_exprs_cmj_k4)

features_z_cmj_k4 = [f"{c}_z" for c in features_cmj_k4]
agg_exprs_cmj_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_cmj_k4]

df_cluster_z_means_cmj_k4 = (
    df_z_cmj_k4
    .groupBy(F.col(f"`{CLUSTER_COL_CMJ_K4}`"))
    .agg(*agg_exprs_cmj_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_CMJ_K4}`"))
)

display(df_cluster_z_means_cmj_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (CMJ best_k4)
cluster_z_pd_cmj_k4 = df_cluster_z_means_cmj_k4.toPandas()

importance_pd_cmj_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_cmj_k4[c].max() - cluster_z_pd_cmj_k4[c].min(),
        "mean_abs_z": cluster_z_pd_cmj_k4[c].abs().mean()
    } for c in features_z_cmj_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_cmj_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (CMJ cluster_UMAP2_K4_nn5_md0.0)
TOP_N_CMJ_K4 = 30

top_vars_cmj_k4   = importance_pd_cmj_k4.head(TOP_N_CMJ_K4)["variable"].tolist()
top_z_cols_cmj_k4 = [f"{v}_z" for v in top_vars_cmj_k4]
heatmap_data_cmj_k4 = cluster_z_pd_cmj_k4.set_index(CLUSTER_COL_CMJ_K4)[top_z_cols_cmj_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_cmj_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_cmj_k4)),
    labels=top_vars_cmj_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_cmj_k4.index)),
    labels=heatmap_data_cmj_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_CMJ_K4} (k={K_CMJ_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ISOPRONE

# COMMAND ----------

# DBTITLE 1,Carga columnas Isoprone limpio vs original
cols_isoprone_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone_limpio").columns
cols_isoprone        = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Isoprone limpio → original
import difflib

set_limpio_iso = set(cols_isoprone_limpio)
set_orig_iso   = set(cols_isoprone)

shared_iso      = sorted(set_limpio_iso & set_orig_iso)
only_limpio_iso = sorted(set_limpio_iso - set_orig_iso)
only_orig_iso   = sorted(set_orig_iso - set_limpio_iso)

print(f"Compartidas : {len(shared_iso)}")
print(f"Solo limpio : {len(only_limpio_iso)}")
print(f"Solo original: {len(only_orig_iso)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_iso:
    matches = difflib.get_close_matches(col_l, only_orig_iso, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original (posibles eliminadas):")
for c in only_orig_iso:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Isoprone)
id_col = "id_unico_deportista"

# Candidatas: columnas K4 (incluyendo las que tienen punto en el nombre)
candidatas_k4_iso = [
    c for c in cols_isoprone
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Isoprone original:", candidatas_k4_iso)

df_limpio_iso = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone_limpio").select(id_col, "best_k4")
df_orig_iso   = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_iso])

df_joined_iso = df_limpio_iso.join(df_orig_iso, on=id_col, how="inner")
n_total_iso = df_joined_iso.count()

results_iso = []
for c in candidatas_k4_iso:
    n_match = df_joined_iso.filter(F.col("best_k4") == F.col(c)).count()
    results_iso.append({
        "candidata": c,
        "n_total": n_total_iso,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_iso * 100, 2) if n_total_iso > 0 else 0
    })

results_iso_pd = pd.DataFrame(results_iso).sort_values("pct_match", ascending=False)
print(results_iso_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 = umap_k4_best (Isoprone)
from pyspark.sql.types import NumericType

CLUSTER_COL_ISO_K4 = "umap_k4_best"
ID_COL_ISO_K4 = "id_unico_deportista"

df_iso_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_isoprone")

_k_match_iso_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_ISO_K4)
K_ISO_K4 = int(_k_match_iso_k4.group(1)) if _k_match_iso_k4 else None

# Excluir columnas de cluster / UMAP / umap
all_cluster_cols_iso_k4 = [
    f.name for f in df_iso_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_iso_k4 = list(set([
    ID_COL_ISO_K4, CLUSTER_COL_ISO_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_iso_k4))

features_iso_k4 = [
    field.name
    for field in df_iso_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_iso_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_ISO_K4}  |  K={K_ISO_K4}")
print(f"Nº features: {len(features_iso_k4)}")
for c in features_iso_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Isoprone best_k4)
# Calcular medias y std en una sola pasada
stats_exprs_iso_k4 = [e for col_name in features_iso_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_iso_k4 = df_iso_k4.agg(*stats_exprs_iso_k4).collect()[0]

z_exprs_iso_k4 = {}
for col_name in features_iso_k4:
    mean_val = stats_row_iso_k4[f"{col_name}__mean"]
    std_val  = stats_row_iso_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_iso_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_iso_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_iso_k4 = df_iso_k4.withColumns(z_exprs_iso_k4)

features_z_iso_k4 = [f"{c}_z" for c in features_iso_k4]
agg_exprs_iso_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_iso_k4]

df_cluster_z_means_iso_k4 = (
    df_z_iso_k4
    .groupBy(CLUSTER_COL_ISO_K4)
    .agg(*agg_exprs_iso_k4)
    .orderBy(CLUSTER_COL_ISO_K4)
)

display(df_cluster_z_means_iso_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Isoprone best_k4)
cluster_z_pd_iso_k4 = df_cluster_z_means_iso_k4.toPandas()

importance_pd_iso_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_iso_k4[c].max() - cluster_z_pd_iso_k4[c].min(),
        "mean_abs_z": cluster_z_pd_iso_k4[c].abs().mean()
    } for c in features_z_iso_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_iso_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Isoprone umap_k4_best)
TOP_N_ISO_K4 = 30

top_vars_iso_k4   = importance_pd_iso_k4.head(TOP_N_ISO_K4)["variable"].tolist()
top_z_cols_iso_k4 = [f"{v}_z" for v in top_vars_iso_k4]
heatmap_data_iso_k4 = cluster_z_pd_iso_k4.set_index(CLUSTER_COL_ISO_K4)[top_z_cols_iso_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_iso_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_iso_k4)),
    labels=top_vars_iso_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_iso_k4.index)),
    labels=heatmap_data_iso_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_ISO_K4} (k={K_ISO_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### NORDIC

# COMMAND ----------

# DBTITLE 1,Carga columnas Nordic limpio vs original
cols_nordic_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_nordic_limpio").columns
cols_nordic        = spark.read.table("dev_gold.pitchvision.clustersresults_nordic").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Nordic limpio → original
import difflib

set_limpio_nordic = set(cols_nordic_limpio)
set_orig_nordic   = set(cols_nordic)

shared_nordic      = sorted(set_limpio_nordic & set_orig_nordic)
only_limpio_nordic = sorted(set_limpio_nordic - set_orig_nordic)
only_orig_nordic   = sorted(set_orig_nordic - set_limpio_nordic)

print(f"Compartidas : {len(shared_nordic)}")
print(f"Solo limpio : {len(only_limpio_nordic)}")
print(f"Solo original: {len(only_orig_nordic)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_nordic:
    matches = difflib.get_close_matches(col_l, only_orig_nordic, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_nordic:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Nordic)
id_col = "id_unico_deportista"

candidatas_k4_nordic = [
    c for c in cols_nordic
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Nordic original:", candidatas_k4_nordic)

df_limpio_nordic = spark.read.table("dev_gold.pitchvision.clustersresults_nordic_limpio").select(id_col, "best_k4")
df_orig_nordic   = spark.read.table("dev_gold.pitchvision.clustersresults_nordic").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_nordic])

df_joined_nordic = df_limpio_nordic.join(df_orig_nordic, on=id_col, how="inner")
n_total_nordic = df_joined_nordic.count()

results_nordic_k4 = []
for c in candidatas_k4_nordic:
    n_match = df_joined_nordic.filter(F.col("best_k4") == F.col(c)).count()
    results_nordic_k4.append({
        "candidata": c,
        "n_total": n_total_nordic,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_nordic * 100, 2) if n_total_nordic > 0 else 0
    })

results_nordic_k4_pd = pd.DataFrame(results_nordic_k4).sort_values("pct_match", ascending=False)
print(results_nordic_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (Nordic)
# Actualizar con el nombre exacto de la columna origen tras revisar resultados arriba
CLUSTER_COL_NORDIC_K4 = results_nordic_k4_pd.iloc[0]["candidata"]
ID_COL_NORDIC_K4 = "id_unico_deportista"

df_nordic_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_nordic")

_k_match_nordic_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_NORDIC_K4)
K_NORDIC_K4 = int(_k_match_nordic_k4.group(1)) if _k_match_nordic_k4 else None

all_cluster_cols_nordic_k4 = [
    f.name for f in df_nordic_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_nordic_k4 = list(set([
    ID_COL_NORDIC_K4, CLUSTER_COL_NORDIC_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_nordic_k4))

features_nordic_k4 = [
    field.name
    for field in df_nordic_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_nordic_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_NORDIC_K4}  |  K={K_NORDIC_K4}")
print(f"Nº features: {len(features_nordic_k4)}")
for c in features_nordic_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Nordic best_k4)
stats_exprs_nordic_k4 = [e for col_name in features_nordic_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_nordic_k4 = df_nordic_k4.agg(*stats_exprs_nordic_k4).collect()[0]

z_exprs_nordic_k4 = {}
for col_name in features_nordic_k4:
    mean_val = stats_row_nordic_k4[f"{col_name}__mean"]
    std_val  = stats_row_nordic_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_nordic_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_nordic_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_nordic_k4 = df_nordic_k4.withColumns(z_exprs_nordic_k4)

features_z_nordic_k4 = [f"{c}_z" for c in features_nordic_k4]
agg_exprs_nordic_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_nordic_k4]

df_cluster_z_means_nordic_k4 = (
    df_z_nordic_k4
    .groupBy(F.col(f"`{CLUSTER_COL_NORDIC_K4}`"))
    .agg(*agg_exprs_nordic_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_NORDIC_K4}`"))
)

display(df_cluster_z_means_nordic_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Nordic best_k4)
cluster_z_pd_nordic_k4 = df_cluster_z_means_nordic_k4.toPandas()

importance_pd_nordic_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_nordic_k4[c].max() - cluster_z_pd_nordic_k4[c].min(),
        "mean_abs_z": cluster_z_pd_nordic_k4[c].abs().mean()
    } for c in features_z_nordic_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_nordic_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Nordic best_k4)
TOP_N_NORDIC_K4 = 30

top_vars_nordic_k4   = importance_pd_nordic_k4.head(TOP_N_NORDIC_K4)["variable"].tolist()
top_z_cols_nordic_k4 = [f"{v}_z" for v in top_vars_nordic_k4]
heatmap_data_nordic_k4 = cluster_z_pd_nordic_k4.set_index(CLUSTER_COL_NORDIC_K4)[top_z_cols_nordic_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_nordic_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_nordic_k4)),
    labels=top_vars_nordic_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_nordic_k4.index)),
    labels=heatmap_data_nordic_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_NORDIC_K4} (k={K_NORDIC_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SLJ

# COMMAND ----------

# DBTITLE 1,Carga columnas SLJ limpio vs original
cols_slj_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_slj_limpio").columns
cols_slj        = spark.read.table("dev_gold.pitchvision.clustersresults_slj").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas SLJ limpio → original
import difflib

set_limpio_slj = set(cols_slj_limpio)
set_orig_slj   = set(cols_slj)

shared_slj      = sorted(set_limpio_slj & set_orig_slj)
only_limpio_slj = sorted(set_limpio_slj - set_orig_slj)
only_orig_slj   = sorted(set_orig_slj - set_limpio_slj)

print(f"Compartidas : {len(shared_slj)}")
print(f"Solo limpio : {len(only_limpio_slj)}")
print(f"Solo original: {len(only_orig_slj)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_slj:
    matches = difflib.get_close_matches(col_l, only_orig_slj, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_slj:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (SLJ)
id_col = "id_unico_deportista"

candidatas_k4_slj = [
    c for c in cols_slj
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en SLJ original:", candidatas_k4_slj)

df_limpio_slj = spark.read.table("dev_gold.pitchvision.clustersresults_slj_limpio").select(id_col, "best_k4")
df_orig_slj   = spark.read.table("dev_gold.pitchvision.clustersresults_slj").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_slj])

df_joined_slj = df_limpio_slj.join(df_orig_slj, on=id_col, how="inner")
n_total_slj = df_joined_slj.count()

results_slj_k4 = []
for c in candidatas_k4_slj:
    n_match = df_joined_slj.filter(F.col("best_k4") == F.col(c)).count()
    results_slj_k4.append({
        "candidata": c,
        "n_total": n_total_slj,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_slj * 100, 2) if n_total_slj > 0 else 0
    })

results_slj_k4_pd = pd.DataFrame(results_slj_k4).sort_values("pct_match", ascending=False)
print(results_slj_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (SLJ)
CLUSTER_COL_SLJ_K4 = results_slj_k4_pd.iloc[0]["candidata"]
ID_COL_SLJ_K4 = "id_unico_deportista"

df_slj_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_slj")

_k_match_slj_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_SLJ_K4)
K_SLJ_K4 = int(_k_match_slj_k4.group(1)) if _k_match_slj_k4 else None

all_cluster_cols_slj_k4 = [
    f.name for f in df_slj_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_slj_k4 = list(set([
    ID_COL_SLJ_K4, CLUSTER_COL_SLJ_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_slj_k4))

features_slj_k4 = [
    field.name
    for field in df_slj_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_slj_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_SLJ_K4}  |  K={K_SLJ_K4}")
print(f"Nº features: {len(features_slj_k4)}")
for c in features_slj_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (SLJ best_k4)
stats_exprs_slj_k4 = [e for col_name in features_slj_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_slj_k4 = df_slj_k4.agg(*stats_exprs_slj_k4).collect()[0]

z_exprs_slj_k4 = {}
for col_name in features_slj_k4:
    mean_val = stats_row_slj_k4[f"{col_name}__mean"]
    std_val  = stats_row_slj_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_slj_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_slj_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_slj_k4 = df_slj_k4.withColumns(z_exprs_slj_k4)

features_z_slj_k4 = [f"{c}_z" for c in features_slj_k4]
agg_exprs_slj_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_slj_k4]

df_cluster_z_means_slj_k4 = (
    df_z_slj_k4
    .groupBy(F.col(f"`{CLUSTER_COL_SLJ_K4}`"))
    .agg(*agg_exprs_slj_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_SLJ_K4}`"))
)

display(df_cluster_z_means_slj_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (SLJ best_k4)
cluster_z_pd_slj_k4 = df_cluster_z_means_slj_k4.toPandas()

importance_pd_slj_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_slj_k4[c].max() - cluster_z_pd_slj_k4[c].min(),
        "mean_abs_z": cluster_z_pd_slj_k4[c].abs().mean()
    } for c in features_z_slj_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_slj_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (SLJ best_k4)
TOP_N_SLJ_K4 = 30

top_vars_slj_k4   = importance_pd_slj_k4.head(TOP_N_SLJ_K4)["variable"].tolist()
top_z_cols_slj_k4 = [f"{v}_z" for v in top_vars_slj_k4]
heatmap_data_slj_k4 = cluster_z_pd_slj_k4.set_index(CLUSTER_COL_SLJ_K4)[top_z_cols_slj_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_slj_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_slj_k4)),
    labels=top_vars_slj_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_slj_k4.index)),
    labels=heatmap_data_slj_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_SLJ_K4} (k={K_SLJ_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado pruebas físicas

# COMMAND ----------

# DBTITLE 1,Carga columnas Agregado físicas limpio vs original
cols_fisicas_k4_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas_limpio").columns
cols_fisicas_k4_orig   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Agregado físicas limpio → original
import difflib

set_limpio_fk4 = set(cols_fisicas_k4_limpio)
set_orig_fk4   = set(cols_fisicas_k4_orig)

shared_fk4      = sorted(set_limpio_fk4 & set_orig_fk4)
only_limpio_fk4 = sorted(set_limpio_fk4 - set_orig_fk4)
only_orig_fk4   = sorted(set_orig_fk4  - set_limpio_fk4)

print(f"Compartidas : {len(shared_fk4)}")
print(f"Solo limpio : {len(only_limpio_fk4)}")
print(f"Solo original: {len(only_orig_fk4)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_fk4:
    matches = difflib.get_close_matches(col_l, only_orig_fk4, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_fk4:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Agregado físicas)
id_col = "id_unico_deportista"

candidatas_k4_fis = [
    c for c in cols_fisicas_k4_orig
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Agregado físicas original:", candidatas_k4_fis)

df_limpio_fis_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas_limpio").select(id_col, "best_k4")
df_orig_fis_k4   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_fis])

df_joined_fis_k4 = df_limpio_fis_k4.join(df_orig_fis_k4, on=id_col, how="inner")
n_total_fis_k4 = df_joined_fis_k4.count()

results_fis_k4 = []
for c in candidatas_k4_fis:
    n_match = df_joined_fis_k4.filter(F.col("best_k4") == F.col(c)).count()
    results_fis_k4.append({
        "candidata": c,
        "n_total": n_total_fis_k4,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_fis_k4 * 100, 2) if n_total_fis_k4 > 0 else 0
    })

results_fis_k4_pd = pd.DataFrame(results_fis_k4).sort_values("pct_match", ascending=False)
print(results_fis_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (Agregado físicas)
CLUSTER_COL_FIS_K4 = results_fis_k4_pd.iloc[0]["candidata"]
ID_COL_FIS_K4 = "id_unico_deportista"

df_fis_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas")

_k_match_fis_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_FIS_K4)
K_FIS_K4 = int(_k_match_fis_k4.group(1)) if _k_match_fis_k4 else None

all_cluster_cols_fis_k4 = [
    f.name for f in df_fis_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_fis_k4 = list(set([
    ID_COL_FIS_K4, CLUSTER_COL_FIS_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_fis_k4))

features_fis_k4 = [
    field.name
    for field in df_fis_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_fis_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_FIS_K4}  |  K={K_FIS_K4}")
print(f"Nº features: {len(features_fis_k4)}")
for c in features_fis_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agregado físicas best_k4)
stats_exprs_fis_k4 = [e for col_name in features_fis_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_fis_k4 = df_fis_k4.agg(*stats_exprs_fis_k4).collect()[0]

z_exprs_fis_k4 = {}
for col_name in features_fis_k4:
    mean_val = stats_row_fis_k4[f"{col_name}__mean"]
    std_val  = stats_row_fis_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_fis_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_fis_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_fis_k4 = df_fis_k4.withColumns(z_exprs_fis_k4)

features_z_fis_k4 = [f"{c}_z" for c in features_fis_k4]
agg_exprs_fis_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_fis_k4]

df_cluster_z_means_fis_k4 = (
    df_z_fis_k4
    .groupBy(F.col(f"`{CLUSTER_COL_FIS_K4}`"))
    .agg(*agg_exprs_fis_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_FIS_K4}`"))
)

display(df_cluster_z_means_fis_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agregado físicas best_k4)
cluster_z_pd_fis_k4 = df_cluster_z_means_fis_k4.toPandas()

importance_pd_fis_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_fis_k4[c].max() - cluster_z_pd_fis_k4[c].min(),
        "mean_abs_z": cluster_z_pd_fis_k4[c].abs().mean()
    } for c in features_z_fis_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_fis_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agregado físicas best_k4)
TOP_N_FIS_K4 = 30

top_vars_fis_k4   = importance_pd_fis_k4.head(TOP_N_FIS_K4)["variable"].tolist()
top_z_cols_fis_k4 = [f"{v}_z" for v in top_vars_fis_k4]
heatmap_data_fis_k4 = cluster_z_pd_fis_k4.set_index(CLUSTER_COL_FIS_K4)[top_z_cols_fis_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_fis_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_fis_k4)),
    labels=top_vars_fis_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_fis_k4.index)),
    labels=heatmap_data_fis_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_FIS_K4} (k={K_FIS_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### AGREGADO FISICAS + ANTHROPOMETRY

# COMMAND ----------

# DBTITLE 1,Carga columnas Agr. físicas+Anthro limpio vs original
cols_fa_k4_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas_anthropometry_limpio").columns
cols_fa_k4_orig   = spark.read.table("dev_gold.pitchvision.clustersresults_Agregadopruebasfisicas_anthropometry").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Agr. físicas+Anthro limpio → original
import difflib

set_limpio_fa = set(cols_fa_k4_limpio)
set_orig_fa   = set(cols_fa_k4_orig)

shared_fa      = sorted(set_limpio_fa & set_orig_fa)
only_limpio_fa = sorted(set_limpio_fa - set_orig_fa)
only_orig_fa   = sorted(set_orig_fa  - set_limpio_fa)

print(f"Compartidas : {len(shared_fa)}")
print(f"Solo limpio : {len(only_limpio_fa)}")
print(f"Solo original: {len(only_orig_fa)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_fa:
    matches = difflib.get_close_matches(col_l, only_orig_fa, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_fa:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Agr. físicas+Anthro)
id_col = "id_unico_deportista"

candidatas_k4_fa = [
    c for c in cols_fa_k4_orig
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Agr. físicas+Anthro original:", candidatas_k4_fa)

df_limpio_fa_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas_anthropometry_limpio").select(id_col, "best_k4")
df_orig_fa_k4   = spark.read.table("dev_gold.pitchvision.clustersresults_Agregadopruebasfisicas_anthropometry").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_fa])

df_joined_fa_k4 = df_limpio_fa_k4.join(df_orig_fa_k4, on=id_col, how="inner")
n_total_fa_k4 = df_joined_fa_k4.count()

results_fa_k4 = []
for c in candidatas_k4_fa:
    n_match = df_joined_fa_k4.filter(F.col("best_k4") == F.col(c)).count()
    results_fa_k4.append({
        "candidata": c,
        "n_total": n_total_fa_k4,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_fa_k4 * 100, 2) if n_total_fa_k4 > 0 else 0
    })

results_fa_k4_pd = pd.DataFrame(results_fa_k4).sort_values("pct_match", ascending=False)
print(results_fa_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (Agr. físicas+Anthro)
CLUSTER_COL_FA_K4 = results_fa_k4_pd.iloc[0]["candidata"]
ID_COL_FA_K4 = "id_unico_deportista"

df_fa_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_Agregadopruebasfisicas_anthropometry")

_k_match_fa_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_FA_K4)
K_FA_K4 = int(_k_match_fa_k4.group(1)) if _k_match_fa_k4 else None

all_cluster_cols_fa_k4 = [
    f.name for f in df_fa_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_fa_k4 = list(set([
    ID_COL_FA_K4, CLUSTER_COL_FA_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_fa_k4))

features_fa_k4 = [
    field.name
    for field in df_fa_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_fa_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_FA_K4}  |  K={K_FA_K4}")
print(f"Nº features: {len(features_fa_k4)}")
for c in features_fa_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agr. físicas+Anthro best_k4)
stats_exprs_fa_k4 = [e for col_name in features_fa_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_fa_k4 = df_fa_k4.agg(*stats_exprs_fa_k4).collect()[0]

z_exprs_fa_k4 = {}
for col_name in features_fa_k4:
    mean_val = stats_row_fa_k4[f"{col_name}__mean"]
    std_val  = stats_row_fa_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_fa_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_fa_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_fa_k4 = df_fa_k4.withColumns(z_exprs_fa_k4)

features_z_fa_k4 = [f"{c}_z" for c in features_fa_k4]
agg_exprs_fa_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_fa_k4]

df_cluster_z_means_fa_k4 = (
    df_z_fa_k4
    .groupBy(F.col(f"`{CLUSTER_COL_FA_K4}`"))
    .agg(*agg_exprs_fa_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_FA_K4}`"))
)

display(df_cluster_z_means_fa_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agr. físicas+Anthro best_k4)
cluster_z_pd_fa_k4 = df_cluster_z_means_fa_k4.toPandas()

importance_pd_fa_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_fa_k4[c].max() - cluster_z_pd_fa_k4[c].min(),
        "mean_abs_z": cluster_z_pd_fa_k4[c].abs().mean()
    } for c in features_z_fa_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_fa_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agr. físicas+Anthro best_k4)
TOP_N_FA_K4 = 30

top_vars_fa_k4   = importance_pd_fa_k4.head(TOP_N_FA_K4)["variable"].tolist()
top_z_cols_fa_k4 = [f"{v}_z" for v in top_vars_fa_k4]
heatmap_data_fa_k4 = cluster_z_pd_fa_k4.set_index(CLUSTER_COL_FA_K4)[top_z_cols_fa_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_fa_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_fa_k4)),
    labels=top_vars_fa_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_fa_k4.index)),
    labels=heatmap_data_fa_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_FA_K4} (k={K_FA_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregado nordic isoprone y gps

# COMMAND ----------

# DBTITLE 1,Carga columnas Agr. Nordic+Iso+GPS limpio vs original
cols_nig_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps_limpio").columns
cols_nig_orig   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Agr. Nordic+Iso+GPS limpio → original
import difflib

set_limpio_nig = set(cols_nig_limpio)
set_orig_nig   = set(cols_nig_orig)

shared_nig      = sorted(set_limpio_nig & set_orig_nig)
only_limpio_nig = sorted(set_limpio_nig - set_orig_nig)
only_orig_nig   = sorted(set_orig_nig  - set_limpio_nig)

print(f"Compartidas : {len(shared_nig)}")
print(f"Solo limpio : {len(only_limpio_nig)}")
print(f"Solo original: {len(only_orig_nig)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_nig:
    matches = difflib.get_close_matches(col_l, only_orig_nig, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_nig:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Agr. Nordic+Iso+GPS)
id_col = "id_unico_deportista"

candidatas_k4_nig = [
    c for c in cols_nig_orig
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Agr. Nordic+Iso+GPS original:", candidatas_k4_nig)

df_limpio_nig_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps_limpio").select(id_col, "best_k4")
df_orig_nig_k4   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_nig])

df_joined_nig_k4 = df_limpio_nig_k4.join(df_orig_nig_k4, on=id_col, how="inner")
n_total_nig_k4 = df_joined_nig_k4.count()

results_nig_k4 = []
for c in candidatas_k4_nig:
    n_match = df_joined_nig_k4.filter(F.col("best_k4") == F.col(c)).count()
    results_nig_k4.append({
        "candidata": c,
        "n_total": n_total_nig_k4,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_nig_k4 * 100, 2) if n_total_nig_k4 > 0 else 0
    })

results_nig_k4_pd = pd.DataFrame(results_nig_k4).sort_values("pct_match", ascending=False)
print(results_nig_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (Agr. Nordic+Iso+GPS)
CLUSTER_COL_NIG_K4 = results_nig_k4_pd.iloc[0]["candidata"]
ID_COL_NIG_K4 = "id_unico_deportista"

df_nig_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadopruebasnordic_isoprone_gps")

_k_match_nig_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_NIG_K4)
K_NIG_K4 = int(_k_match_nig_k4.group(1)) if _k_match_nig_k4 else None

all_cluster_cols_nig_k4 = [
    f.name for f in df_nig_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_nig_k4 = list(set([
    ID_COL_NIG_K4, CLUSTER_COL_NIG_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_nig_k4))

features_nig_k4 = [
    field.name
    for field in df_nig_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_nig_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_NIG_K4}  |  K={K_NIG_K4}")
print(f"Nº features: {len(features_nig_k4)}")
for c in features_nig_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agr. Nordic+Iso+GPS best_k4)
stats_exprs_nig_k4 = [e for col_name in features_nig_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_nig_k4 = df_nig_k4.agg(*stats_exprs_nig_k4).collect()[0]

z_exprs_nig_k4 = {}
for col_name in features_nig_k4:
    mean_val = stats_row_nig_k4[f"{col_name}__mean"]
    std_val  = stats_row_nig_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_nig_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_nig_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_nig_k4 = df_nig_k4.withColumns(z_exprs_nig_k4)

features_z_nig_k4 = [f"{c}_z" for c in features_nig_k4]
agg_exprs_nig_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_nig_k4]

df_cluster_z_means_nig_k4 = (
    df_z_nig_k4
    .groupBy(F.col(f"`{CLUSTER_COL_NIG_K4}`"))
    .agg(*agg_exprs_nig_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_NIG_K4}`"))
)

display(df_cluster_z_means_nig_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agr. Nordic+Iso+GPS best_k4)
cluster_z_pd_nig_k4 = df_cluster_z_means_nig_k4.toPandas()

importance_pd_nig_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_nig_k4[c].max() - cluster_z_pd_nig_k4[c].min(),
        "mean_abs_z": cluster_z_pd_nig_k4[c].abs().mean()
    } for c in features_z_nig_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_nig_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agr. Nordic+Iso+GPS best_k4)
TOP_N_NIG_K4 = 30

top_vars_nig_k4   = importance_pd_nig_k4.head(TOP_N_NIG_K4)["variable"].tolist()
top_z_cols_nig_k4 = [f"{v}_z" for v in top_vars_nig_k4]
heatmap_data_nig_k4 = cluster_z_pd_nig_k4.set_index(CLUSTER_COL_NIG_K4)[top_z_cols_nig_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_nig_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_nig_k4)),
    labels=top_vars_nig_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_nig_k4.index)),
    labels=heatmap_data_nig_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_NIG_K4} (k={K_NIG_K4})")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### AGREGADO TODO

# COMMAND ----------

# DBTITLE 1,Carga columnas Agregado todo limpio vs original
cols_todo_limpio = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo_limpio").columns
cols_todo_orig   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo").columns

# COMMAND ----------

# DBTITLE 1,Mapeo columnas Agregado todo limpio → original
import difflib

set_limpio_todo = set(cols_todo_limpio)
set_orig_todo   = set(cols_todo_orig)

shared_todo      = sorted(set_limpio_todo & set_orig_todo)
only_limpio_todo = sorted(set_limpio_todo - set_orig_todo)
only_orig_todo   = sorted(set_orig_todo  - set_limpio_todo)

print(f"Compartidas : {len(shared_todo)}")
print(f"Solo limpio : {len(only_limpio_todo)}")
print(f"Solo original: {len(only_orig_todo)}")
print()

print("=" * 70)
print(f"{'Columna en limpio':<40}  {'Mejor match en original':<30}  Score")
print("=" * 70)
for col_l in only_limpio_todo:
    matches = difflib.get_close_matches(col_l, only_orig_todo, n=1, cutoff=0.0)
    best = matches[0] if matches else ""
    score = difflib.SequenceMatcher(None, col_l, best).ratio() if best else 0.0
    print(f"{col_l:<40}  {best:<30}  {score:.3f}")

print()
print("Columnas solo en original:")
for c in only_orig_todo:
    print(" ", c)

# COMMAND ----------

# DBTITLE 1,Buscar columna origen de best_k4 (Agregado todo)
id_col = "id_unico_deportista"

candidatas_k4_todo = [
    c for c in cols_todo_orig
    if re.search(r'[Kk]4', c) and c != id_col
    and not c.startswith("pct_") and not c.endswith("_missing")
]
print("Candidatas K4 en Agregado todo original:", candidatas_k4_todo)

df_limpio_todo_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo_limpio").select(id_col, "best_k4")
df_orig_todo_k4   = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo").select([id_col] + [F.col(f"`{c}`").alias(c) for c in candidatas_k4_todo])

df_joined_todo_k4 = df_limpio_todo_k4.join(df_orig_todo_k4, on=id_col, how="inner")
n_total_todo_k4 = df_joined_todo_k4.count()

results_todo_k4 = []
for c in candidatas_k4_todo:
    n_match = df_joined_todo_k4.filter(F.col("best_k4") == F.col(c)).count()
    results_todo_k4.append({
        "candidata": c,
        "n_total": n_total_todo_k4,
        "n_match": n_match,
        "pct_match": round(n_match / n_total_todo_k4 * 100, 2) if n_total_todo_k4 > 0 else 0
    })

results_todo_k4_pd = pd.DataFrame(results_todo_k4).sort_values("pct_match", ascending=False)
print(results_todo_k4_pd.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Config best_k4 (Agregado todo)
CLUSTER_COL_TODO_K4 = results_todo_k4_pd.iloc[0]["candidata"]
ID_COL_TODO_K4 = "id_unico_deportista"

df_todo_k4 = spark.read.table("dev_gold.pitchvision.clustersresults_agregadotodo")

_k_match_todo_k4 = re.search(r'[Kk](\d+)', CLUSTER_COL_TODO_K4)
K_TODO_K4 = int(_k_match_todo_k4.group(1)) if _k_match_todo_k4 else None

all_cluster_cols_todo_k4 = [
    f.name for f in df_todo_k4.schema.fields
    if f.name.startswith("cluster_") or f.name.startswith("UMAP_") or f.name.startswith("umap_")
]

cols_excluir_todo_k4 = list(set([
    ID_COL_TODO_K4, CLUSTER_COL_TODO_K4,
    "temporada_fuente", "first_test_date", "last_test_date"
] + all_cluster_cols_todo_k4))

features_todo_k4 = [
    field.name
    for field in df_todo_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_todo_k4
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"CLUSTER_COL: {CLUSTER_COL_TODO_K4}  |  K={K_TODO_K4}")
print(f"Nº features: {len(features_todo_k4)}")
for c in features_todo_k4:
    print(c)

# COMMAND ----------

# DBTITLE 1,Z-scores y medias por cluster (Agregado todo best_k4)
stats_exprs_todo_k4 = [e for col_name in features_todo_k4 for e in [
    F.avg(col_name).alias(f"{col_name}__mean"),
    F.stddev(col_name).alias(f"{col_name}__std")
]]
stats_row_todo_k4 = df_todo_k4.agg(*stats_exprs_todo_k4).collect()[0]

z_exprs_todo_k4 = {}
for col_name in features_todo_k4:
    mean_val = stats_row_todo_k4[f"{col_name}__mean"]
    std_val  = stats_row_todo_k4[f"{col_name}__std"]
    if std_val is not None and std_val != 0:
        z_exprs_todo_k4[f"{col_name}_z"] = (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
    else:
        z_exprs_todo_k4[f"{col_name}_z"] = F.lit(0.0)

df_z_todo_k4 = df_todo_k4.withColumns(z_exprs_todo_k4)

features_z_todo_k4 = [f"{c}_z" for c in features_todo_k4]
agg_exprs_todo_k4 = [F.count("*").alias("n_jugadores")] + [F.avg(c).alias(c) for c in features_z_todo_k4]

df_cluster_z_means_todo_k4 = (
    df_z_todo_k4
    .groupBy(F.col(f"`{CLUSTER_COL_TODO_K4}`"))
    .agg(*agg_exprs_todo_k4)
    .orderBy(F.col(f"`{CLUSTER_COL_TODO_K4}`"))
)

display(df_cluster_z_means_todo_k4)

# COMMAND ----------

# DBTITLE 1,Importancia de variables (Agregado todo best_k4)
cluster_z_pd_todo_k4 = df_cluster_z_means_todo_k4.toPandas()

importance_pd_todo_k4 = (
    pd.DataFrame([{
        "variable": c.replace("_z", ""),
        "range_between_clusters": cluster_z_pd_todo_k4[c].max() - cluster_z_pd_todo_k4[c].min(),
        "mean_abs_z": cluster_z_pd_todo_k4[c].abs().mean()
    } for c in features_z_todo_k4])
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_pd_todo_k4.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap perfil por cluster (Agregado todo best_k4)
TOP_N_TODO_K4 = 30

top_vars_todo_k4   = importance_pd_todo_k4.head(TOP_N_TODO_K4)["variable"].tolist()
top_z_cols_todo_k4 = [f"{v}_z" for v in top_vars_todo_k4]
heatmap_data_todo_k4 = cluster_z_pd_todo_k4.set_index(CLUSTER_COL_TODO_K4)[top_z_cols_todo_k4]

plt.figure(figsize=(16, 7))

plt.imshow(
    heatmap_data_todo_k4,
    aspect="auto"
)

plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars_todo_k4)),
    labels=top_vars_todo_k4,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data_todo_k4.index)),
    labels=heatmap_data_todo_k4.index
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(f"Perfil medio estandarizado por cluster | {CLUSTER_COL_TODO_K4} (k={K_TODO_K4})")

plt.tight_layout()
plt.show()
