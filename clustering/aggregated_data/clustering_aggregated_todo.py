# Databricks notebook source
# MAGIC %pip install umap-learn

# COMMAND ----------

# MAGIC %pip install hdbscan

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
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

# COMMAND ----------

ID_COL = "id_unico_deportista"

TABLE_COMPETITIVE        = "dev_gold.pitchvision.competitive_profile_agregado_por_jugador"
TABLE_GPS                = "dev_gold.pitchvision.gps_data_agregado_por_jugador"
TABLE_CMJ                = "dev_gold.pitchvision.cmj_agregado_por_jugador"
TABLE_ISOPRONE           = "dev_gold.pitchvision.isoprone_agregado_por_jugador"
TABLE_NORDIC             = "dev_gold.pitchvision.nordic_Agregado_por_jugador"
TABLE_SLJ                = "dev_gold.pitchvision.slj_agregado_por_jugador"
TABLE_ANTHROPOMETRY      = "dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering"
TABLE_INJURY             = "dev_gold.pitchvision.injury_agregado_por_jugador"

def read_and_suffix(table_name, suffix):
    df = spark.table(table_name)
    
    renamed_cols = [
        F.col(c).alias(c if c == ID_COL else f"{c}_{suffix}")
        for c in df.columns
    ]
    
    return df.select(*renamed_cols)

df_competitive   = read_and_suffix(TABLE_COMPETITIVE, "competitive")
df_gps           = read_and_suffix(TABLE_GPS, "gps")
df_cmj           = read_and_suffix(TABLE_CMJ, "cmj")
df_isoprone      = read_and_suffix(TABLE_ISOPRONE, "isoprone")
df_nordic        = read_and_suffix(TABLE_NORDIC, "nordic")
df_slj           = read_and_suffix(TABLE_SLJ, "slj")
df_anthropometry = read_and_suffix(TABLE_ANTHROPOMETRY, "anthropometry")
df_injury        = read_and_suffix(TABLE_INJURY, "injury")

# COMMAND ----------

from functools import reduce

dfs = [
    df_competitive,
    df_gps,
    df_cmj,
    df_isoprone,
    df_nordic,
    df_slj,
    df_anthropometry,
    df_injury
]

df_joined = reduce(lambda df1, df2: df1.join(df2, ID_COL, "inner"), dfs)
unique_players_count = df_joined.select(ID_COL).distinct().count()
unique_players_count

# COMMAND ----------

# Selecciona solo columnas numéricas relevantes para clustering, excluyendo identificadores, fechas y columnas categóricas
from pyspark.sql.types import NumericType

cols_to_exclude = [
    "n_sesiones_validas_gps",
    "n_temporadas_gps",
    "temporadas_fuente_cmj",
    "n_tests_isoprone_isoprone",
    "n_seasons_isoprone_isoprone",
    "first_isoprone_test_date_isoprone",
    "last_isoprone_test_date_isoprone",

    "id_unico_deportista",
    "temporada_fuente",
    "first_test_date",
    "last_test_date",
    "first_isoprone_test_date",
    "last_isoprone_test_date",
    "pct_torque_asym_pct_missing_isoprone",
    "pct_max_force_asym_pct_missing_isoprone",
    "pct_avg_force_asym_pct_missing_isoprone",
    "pct_impulse_asym_pct_missing_isoprone",
    "n_tests_agregados_nordic",
    "n_temporadas_agregadas_nordic",
    "temporada_fuente_nordic",
    "first_test_date_nordic",
    "last_test_date_nordic",
    "mdc_left_max_force_n_mean_nordic",
    "mdc_left_avg_force_n_mean_nordic",
    "mdc_left_impulse_ns_mean_nordic",
    "mdc_left_torque_nm_mean_nordic",
    "mdc_right_max_force_n_mean_nordic",
    "mdc_right_avg_force_n_mean_nordic",
    "mdc_right_impulse_ns_mean_nordic",
    "mdc_right_torque_nm_mean_nordic",
    "pct_torque_asym_pct_missing_nordic",
    "pct_max_force_asym_pct_missing_nordic",
    "pct_avg_force_asym_pct_missing_nordic",
    "pct_impulse_asym_pct_missing_nordic",
    "n_tests_agregados_slj",
    "n_temporadas_agregadas_slj",
    "temporada_fuente_slj",
    "first_test_date_slj",
    "last_test_date_slj",
    "n_trials_mean_slj",
    "weight_kg_mean_slj"
]

numeric_cols = [f.name for f in df_joined.schema.fields if isinstance(f.dataType, NumericType) and f.name not in cols_to_exclude]
df_clustering = df_joined.select("id_unico_deportista", *numeric_cols)

display(df_clustering)

# COMMAND ----------

# Columnas que NO deben entrar al clustering
ID_COL = "id_unico_deportista"
cols_excluir_cluster = [
    ID_COL,
]

features_cluster = [
    field.name
    for field in df_clustering.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

print(f"Nº variables iniciales para clustering: {len(features_cluster)}")

for c in features_cluster:
    print(c)

# COMMAND ----------

df_cluster_input = df_clustering

assembler = VectorAssembler(
    inputCols=features_cluster,
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline = Pipeline(stages=[assembler, scaler])

pipeline_model = pipeline.fit(df_cluster_input)

df_cluster_ready = (
    pipeline_model
    .transform(df_cluster_input)
    .select(
        ID_COL,
        "features_raw",
        "features_scaled",
        *features_cluster
    )
    .cache()
)

df_cluster_ready.count()

display(df_cluster_ready.limit(10))

# COMMAND ----------

df_kmeans_input = (
    df_cluster_ready
    .select(ID_COL, "features_scaled")
    .cache()
)

df_kmeans_input.count()

# Rango de k a probar
k_values = range(2, 10)

elbow_results = []

for k in k_values:
    print(f"Entrenando KMeans con k={k}...")
    
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features_scaled",
        predictionCol="cluster",
        maxIter=20,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_kmeans_input)
    
    # WSSSE: suma de distancias cuadradas dentro de cada cluster
    wssse = model.summary.trainingCost
    
    elbow_results.append((int(k), float(wssse)))
    
    print(f"k={k} | WSSSE={wssse:.2f}")

elbow_df = spark.createDataFrame(
    elbow_results,
    ["k", "wssse"]
)

display(elbow_df.orderBy("k"))

# COMMAND ----------

elbow_pd = elbow_df.toPandas().sort_values("k")

plt.figure(figsize=(8, 5))
plt.plot(elbow_pd["k"], elbow_pd["wssse"], marker="o")

plt.xlabel("Número de clusters (k)")
plt.ylabel("WSSSE")
plt.title("Método del codo - KMeans sobre tabla conjunta")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### KMeans + PCA

# COMMAND ----------

df_pca_input = (
    df_cluster_ready
    .select(ID_COL, "features_scaled")
    .cache()
)

df_pca_input.count()

# Número máximo de componentes a probar
# Usamos como máximo 30 para que el gráfico sea legible
n_features = len(features_cluster)
max_components = min(n_features, 50)

print(f"Nº variables originales: {n_features}")
print(f"Nº componentes PCA calculadas: {max_components}")

pca = PCA(
    k=max_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

pca_model = pca.fit(df_pca_input)

explained_variance = list(pca_model.explainedVariance)

pca_variance_pd = pd.DataFrame({
    "component": list(range(1, len(explained_variance) + 1)),
    "explained_variance": explained_variance
})

pca_variance_pd["cumulative_explained_variance"] = (
    pca_variance_pd["explained_variance"].cumsum()
)

display(pca_variance_pd)

# COMMAND ----------

plt.figure(figsize=(9, 5))

plt.plot(
    pca_variance_pd["component"],
    pca_variance_pd["cumulative_explained_variance"],
    marker="o"
)

plt.axhline(0.80, linestyle="--", label="80%")
plt.axhline(0.85, linestyle="--", label="85%")
plt.axhline(0.90, linestyle="--", label="90%")
plt.axhline(0.95, linestyle="--", label="95%")

plt.xlabel("Número de componentes PCA")
plt.ylabel("Varianza explicada acumulada")
plt.title("PCA - Varianza explicada acumulada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ID_COL = "id_unico_deportista"

N_COMPONENTS_LOADINGS = 48

# Reconstruir matriz escalada desde df_cluster_ready
pdf_vectors = df_cluster_ready.select(ID_COL, "features_scaled").toPandas()

X_scaled_loadings = np.vstack(
    pdf_vectors["features_scaled"].apply(lambda x: x.toArray())
)

feature_names = features_cluster

max_components_possible = min(
    N_COMPONENTS_LOADINGS,
    X_scaled_loadings.shape[0] - 1,
    X_scaled_loadings.shape[1]
)

print(f"Nº jugadores: {X_scaled_loadings.shape[0]}")
print(f"Nº variables: {X_scaled_loadings.shape[1]}")
print(f"Nº componentes calculadas: {max_components_possible}")

# Entrenar PCA una sola vez
pca_loadings_model = PCA(
    n_components=max_components_possible,
    random_state=42
)

X_pca_loadings = pca_loadings_model.fit_transform(X_scaled_loadings)

# En sklearn, components_ tiene shape: (n_components, n_features)
# Transponemos para tener filas = variables, columnas = PCs
pc_matrix = pca_loadings_model.components_.T

pc_cols = [
    f"PC{i}"
    for i in range(1, max_components_possible + 1)
]

loadings_all_pd = pd.DataFrame(
    pc_matrix,
    columns=pc_cols
)

loadings_all_pd["variable"] = feature_names

loadings_all_pd = loadings_all_pd[
    ["variable"] + pc_cols
]

# Añadir valores absolutos
for pc in pc_cols:
    loadings_all_pd[f"abs_{pc}"] = loadings_all_pd[pc].abs()

display(loadings_all_pd)

# COMMAND ----------

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.agregadotodo_loadingsPCA")

# COMMAND ----------

from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F

df_pca_input.count()

# Nº de features originales
n_features = len(features_cluster)

n_components_values = [31, 38, 48]
n_components_values = [n for n in n_components_values if n <= n_features]

# Valores de k a probar
k_values = [2, 3, 4, 5, 6, 7, 8]

results_pca_kmeans = []
models_pca_kmeans = {}
predictions_pca_kmeans = {}
pca_models = {}

for n_components in n_components_values:
    print(f"Calculando PCA con {n_components} componentes")
    
    pca = PCA(
        k=n_components,
        inputCol="features_scaled",
        outputCol=f"features_pca_{n_components}"
    )
    
    pca_model = pca.fit(df_pca_input)
    pca_models[n_components] = pca_model
    
    explained_variance_total = float(sum(pca_model.explainedVariance))
    
    df_pca_tmp = (
        pca_model
        .transform(df_pca_input)
        .select(ID_COL, f"features_pca_{n_components}")
        .cache()
    )
    
    df_pca_tmp.count()
    
    evaluator = ClusteringEvaluator(
        featuresCol=f"features_pca_{n_components}",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    
    for k in k_values:
        
        print(f"Entrenando KMeans | PCA={n_components} | k={k}")
        
        kmeans = KMeans(
            k=k,
            seed=42,
            featuresCol=f"features_pca_{n_components}",
            predictionCol="cluster",
            maxIter=50,
            tol=1e-4,
            initMode="k-means||"
        )
        
        model = kmeans.fit(df_pca_tmp)
        predictions = model.transform(df_pca_tmp)
        
        silhouette = evaluator.evaluate(predictions)
        wssse = model.summary.trainingCost
        
        cluster_sizes = {
            row["cluster"]: row["count"]
            for row in predictions.groupBy("cluster").count().orderBy("cluster").collect()
        }
        
        results_pca_kmeans.append((
            int(n_components),
            float(explained_variance_total),
            int(k),
            float(silhouette),
            float(wssse),
            str(cluster_sizes)
        ))
        
        models_pca_kmeans[(n_components, k)] = model
        predictions_pca_kmeans[(n_components, k)] = predictions
        
        print(
            f"PCA={n_components} | k={k} | "
            f"var_exp={explained_variance_total:.3f} | "
            f"silhouette={silhouette:.4f} | "
            f"WSSSE={wssse:.2f} | tamaños={cluster_sizes}"
        )
    
    df_pca_tmp.unpersist()

pca_kmeans_results_df = spark.createDataFrame(
    results_pca_kmeans,
    [
        "pca_components",
        "explained_variance",
        "k",
        "silhouette",
        "wssse",
        "cluster_sizes"
    ]
)

display(
    pca_kmeans_results_df
    .orderBy(F.desc("silhouette"))
)

# COMMAND ----------

# Crear df_resultados: una fila por jugador, columnas originales + una columna por cada combinación PCA/K con la label de cluster

from functools import reduce

# Lista de combinaciones PCA/K usadas en la celda anterior
combinaciones = list(predictions_pca_kmeans.keys())

# DataFrame base con todas las características originales
df_resultados = df_cluster_ready.select(ID_COL, *features_cluster)

# Para cada combinación, añadir columna de cluster
for n_components, k in combinaciones:
    pred = predictions_pca_kmeans[(n_components, k)].select(ID_COL, "cluster")
    col_name = f"cluster_PCA{n_components}_K{k}"
    pred = pred.withColumnRenamed("cluster", col_name)
    df_resultados = df_resultados.join(pred, on=ID_COL, how="left")

display(df_resultados)

# COMMAND ----------

ID_COL = "id_unico_deportista"

N_COMPONENTS = 48
K = 4

# Recuperar predicciones guardadas en la grid
df_pred_k4_pca15 = predictions_pca_kmeans[(N_COMPONENTS, K)]

# Columna PCA correspondiente
PCA_COL = f"features_pca_{N_COMPONENTS}"

# Pasar a pandas
plot_pd = (
    df_pred_k4_pca15
    .select(ID_COL, PCA_COL, "cluster")
    .toPandas()
)

# Extraer PC1 y PC2 del vector PCA(15)
plot_pd["PC1"] = plot_pd[PCA_COL].apply(lambda x: float(x[0]))
plot_pd["PC2"] = plot_pd[PCA_COL].apply(lambda x: float(x[1]))

# Plot
plt.figure(figsize=(8, 6))

for cluster_id in sorted(plot_pd["cluster"].unique()):
    subset = plot_pd[plot_pd["cluster"] == cluster_id]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA(15) + KMeans(k=4) visualizado en PC1 y PC2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

ID_COL = "id_unico_deportista"

N_COMPONENTS = 48
K = 4

# Predicciones de la configuración PCA(15) + KMeans(k=4)
df_pred_final = (
    predictions_pca_kmeans[(N_COMPONENTS, K)]
    .select(ID_COL, F.col("cluster").alias("cluster_k4_pca15"))
)

# Unir clusters con la tabla original de features
df_cluster_interp = (
    df_clustering
    .join(df_pred_final, on=ID_COL, how="inner")
)

display(df_cluster_interp.limit(10))

# COMMAND ----------

CLUSTER_COL = "cluster_k4_pca15"

cols_excluir_interp = [
    ID_COL,
    CLUSTER_COL,
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
]

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

print(f"Nº variables para interpretar: {len(features_interpretacion)}")

for c in features_interpretacion:
    print(c)

# COMMAND ----------

# Compute all means and stds in a single pass (1 Spark action instead of N)
stats_row = df_cluster_interp.agg(
    *[F.avg(c).alias(f"{c}__mean") for c in features_interpretacion],
    *[F.stddev(c).alias(f"{c}__std") for c in features_interpretacion]
).collect()[0]

# Apply all z-score columns at once (flat execution plan)
z_col_exprs = {}
for col_name in features_interpretacion:
    m = stats_row[f"{col_name}__mean"]
    s = stats_row[f"{col_name}__std"]
    if s is not None and s != 0:
        z_col_exprs[f"{col_name}_z"] = (F.col(col_name) - F.lit(m)) / F.lit(s)
    else:
        z_col_exprs[f"{col_name}_z"] = F.lit(0.0)

df_z = df_cluster_interp.withColumns(z_col_exprs)

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
plt.title(
    f"Perfil medio estandarizado por cluster | PCA({N_COMPONENTS}) + KMeans(k={K})"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## KMeans + UMAP

# COMMAND ----------

# Pasamos features_scaled de Spark a Pandas/Numpy
pdf_umap_input = (
    df_cluster_ready
    .select(ID_COL, "features_scaled")
    .toPandas()
)

X_scaled = np.vstack(
    pdf_umap_input["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_umap_input[ID_COL].values

print("Shape matriz UMAP:", X_scaled.shape)
print("Nº jugadores:", len(player_ids))

# COMMAND ----------

import umap
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score

SEED = 42

n_components_values = [2,  5]
n_neighbors_values = [5,15]
min_dist_values = [0.1]
k_values = [3, 4, 5, 6, 7]

results_umap_kmeans = []
umap_embeddings = {}
umap_kmeans_models = {}
umap_kmeans_labels = {}

for n_components in n_components_values:
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            
            print("=" * 80)
            print(
                f"Calculando UMAP | "
                f"n_components={n_components}, "
                f"n_neighbors={n_neighbors}, "
                f"min_dist={min_dist}"
            )
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=SEED
            )
            
            X_umap = reducer.fit_transform(X_scaled)
            
            umap_embeddings[(n_components, n_neighbors, min_dist)] = X_umap
            
            for k in k_values:
                
                print(f"Entrenando KMeans sobre UMAP | k={k}")
                
                km = SklearnKMeans(
                    n_clusters=k,
                    random_state=SEED,
                    n_init=20
                )
                
                labels = km.fit_predict(X_umap)
                
                silhouette = silhouette_score(
                    X_umap,
                    labels,
                    metric="euclidean"
                )
                
                inertia = km.inertia_
                
                unique, counts = np.unique(labels, return_counts=True)
                cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))
                
                results_umap_kmeans.append({
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "k": k,
                    "silhouette": float(silhouette),
                    "inertia": float(inertia),
                    "cluster_sizes": str(cluster_sizes)
                })
                
                umap_kmeans_models[(n_components, n_neighbors, min_dist, k)] = km
                umap_kmeans_labels[(n_components, n_neighbors, min_dist, k)] = labels
                
                print(
                    f"UMAP={n_components} | nn={n_neighbors} | "
                    f"min_dist={min_dist} | k={k} | "
                    f"silhouette={silhouette:.4f} | tamaños={cluster_sizes}"
                )

umap_kmeans_results_pd = (
    pd.DataFrame(results_umap_kmeans)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

display(umap_kmeans_results_pd)

# COMMAND ----------

# Encontrar la mejor configuración UMAP+KMeans según silhouette para cada k
best_labels_dict = {}
for k in k_values:
    df_k = umap_kmeans_results_pd[umap_kmeans_results_pd["k"] == k]
    best_idx = df_k["silhouette"].idxmax()
    best_row = umap_kmeans_results_pd.loc[best_idx]
    n_components = int(best_row["n_components"])
    n_neighbors = int(best_row["n_neighbors"])
    min_dist = float(best_row["min_dist"])
    labels = umap_kmeans_labels[(n_components, n_neighbors, min_dist, k)]
    col_name = f"UMAP_best_K{k}"
    best_labels_dict[col_name] = labels

# Crear DataFrame Pandas con id y labels para cada k
umap_labels_pd = pd.DataFrame({ID_COL: player_ids})
for col_name, labels in best_labels_dict.items():
    umap_labels_pd[col_name] = labels

# Convertir a Spark DataFrame
umap_labels_spark = spark.createDataFrame(umap_labels_pd)

# Añadir columnas de cluster UMAP a df_resultados
df_resultados = df_resultados.join(umap_labels_spark, on=ID_COL, how="left")

display(df_resultados)

# COMMAND ----------

import umap
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Mejor resultado global para k=4
# =========================

K_OBJETIVO = 4

best_row = (
    umap_kmeans_results_pd
    [umap_kmeans_results_pd["k"] == K_OBJETIVO]
    .sort_values("silhouette", ascending=False)
    .iloc[0]
)

best_n_components = int(best_row["n_components"])
best_n_neighbors = int(best_row["n_neighbors"])
best_min_dist = float(best_row["min_dist"])
best_silhouette = float(best_row["silhouette"])

print("Mejor configuración encontrada para k=4:")
print(best_row)

# =========================
# 2. Labels del mejor modelo real
# =========================

labels_best = umap_kmeans_labels[
    (best_n_components, best_n_neighbors, best_min_dist, K_OBJETIVO)
]

# =========================
# 3. UMAP(2) auxiliar para visualización
#    Mantengo nn y min_dist del mejor modelo
# =========================

reducer_vis = umap.UMAP(
    n_components=2,
    n_neighbors=best_n_neighbors,
    min_dist=best_min_dist,
    metric="euclidean",
    random_state=SEED
)

X_umap_vis_2 = reducer_vis.fit_transform(X_scaled)

# =========================
# 4. DataFrame para plot
# =========================

plot_pd = pd.DataFrame({
    "id_unico_deportista": player_ids,
    "UMAP1": X_umap_vis_2[:, 0],
    "UMAP2": X_umap_vis_2[:, 1],
    "cluster": labels_best
})

# =========================
# 5. Plot
# =========================

plt.figure(figsize=(8, 6))

for cluster_id in sorted(plot_pd["cluster"].unique()):
    subset = plot_pd[plot_pd["cluster"] == cluster_id]
    plt.scatter(
        subset["UMAP1"],
        subset["UMAP2"],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title(
    f"Mejor solución k=4 representada en UMAP(2)\n"
    f"Modelo real: UMAP({best_n_components}), "
    f"nn={best_n_neighbors}, min_dist={best_min_dist} | "
    f"Silhouette={best_silhouette:.3f}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

df_resultados.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clustersresults_agregadotodo")

# COMMAND ----------

from sklearn.decomposition import PCA
import hdbscan

SEED = 42

# Pocas combinaciones para que sea rápido
pca_components_values = [15, 20]

umap_configs = [
    {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 3, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 5, "n_neighbors": 15, "min_dist": 0.1},
]

min_cluster_size_values = [5, 10]
min_samples_values = [1, 5]

results_hdbscan = []
embeddings_hdbscan = {}
labels_hdbscan = {}
models_hdbscan = {}


def evaluar_hdbscan(X_reduced, labels):
    """
    Evalúa una solución HDBSCAN.
    El cluster -1 se considera ruido.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    pct_noise = n_noise / len(labels)

    mask_valid = labels != -1

    if n_clusters >= 2 and mask_valid.sum() > n_clusters:
        silhouette = silhouette_score(
            X_reduced[mask_valid],
            labels[mask_valid],
            metric="euclidean"
        )
    else:
        silhouette = np.nan

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))

    return n_clusters, n_noise, pct_noise, silhouette, cluster_sizes



for n_components in pca_components_values:
    
    print("=" * 80)
    print(f"Calculando PCA({n_components})")
    
    pca = PCA(
        n_components=n_components,
        random_state=SEED
    )
    
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    
    embeddings_hdbscan[("pca", n_components)] = X_pca
    
    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            
            print(
                f"HDBSCAN sobre PCA({n_components}) | "
                f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
            )
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            
            labels = clusterer.fit_predict(X_pca)
            
            n_clusters, n_noise, pct_noise, silhouette, cluster_sizes = evaluar_hdbscan(
                X_pca,
                labels
            )
            
            results_hdbscan.append({
                "method": "PCA",
                "n_components": n_components,
                "n_neighbors": None,
                "min_dist": None,
                "explained_variance": explained_variance,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "pct_noise": pct_noise,
                "silhouette_no_noise": silhouette,
                "cluster_sizes": str(cluster_sizes)
            })
            
            labels_hdbscan[
                ("pca", n_components, min_cluster_size, min_samples)
            ] = labels
            
            models_hdbscan[
                ("pca", n_components, min_cluster_size, min_samples)
            ] = clusterer
            
            print(
                f"clusters={n_clusters} | "
                f"ruido={pct_noise:.1%} | "
                f"silhouette={silhouette} | "
                f"tamaños={cluster_sizes}"
            )


# =========================
# UMAP + HDBSCAN
# =========================

for config in umap_configs:
    
    n_components = config["n_components"]
    n_neighbors = config["n_neighbors"]
    min_dist = config["min_dist"]
    
    print("=" * 80)
    print(
        f"Calculando UMAP({n_components}) | "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}"
    )
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=SEED
    )
    
    X_umap = reducer.fit_transform(X_scaled)
    
    embeddings_hdbscan[
        ("umap", n_components, n_neighbors, min_dist)
    ] = X_umap
    
    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            
            print(
                f"HDBSCAN sobre UMAP({n_components}) | "
                f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
            )
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            
            labels = clusterer.fit_predict(X_umap)
            
            n_clusters, n_noise, pct_noise, silhouette, cluster_sizes = evaluar_hdbscan(
                X_umap,
                labels
            )
            
            results_hdbscan.append({
                "method": "UMAP",
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "explained_variance": None,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "pct_noise": pct_noise,
                "silhouette_no_noise": silhouette,
                "cluster_sizes": str(cluster_sizes)
            })
            
            labels_hdbscan[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = labels
            
            models_hdbscan[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = clusterer
            
            print(
                f"clusters={n_clusters} | "
                f"ruido={pct_noise:.1%} | "
                f"silhouette={silhouette} | "
                f"tamaños={cluster_sizes}"
            )


# =========================
# RESULTADOS
# =========================

hdbscan_results_pd = (
    pd.DataFrame(results_hdbscan)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

display(hdbscan_results_pd)

# COMMAND ----------

# FILTRAR RESULTADOS 

max_noise = 0.15

hdbscan_valid = hdbscan_results_pd[
    (hdbscan_results_pd["n_clusters"] >= 2) &
    (hdbscan_results_pd["pct_noise"] <= max_noise) &
    (hdbscan_results_pd["silhouette_no_noise"].notna())
].copy()

hdbscan_valid = hdbscan_valid.sort_values(
    ["silhouette_no_noise", "pct_noise"],
    ascending=[False, True]
).reset_index(drop=True)

display(hdbscan_valid)
