# Databricks notebook source
# MAGIC %pip install umap-learn

# COMMAND ----------

# MAGIC %pip install hdbscan

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql.types import NumericType
import seaborn as sns
import umap
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import hdbscan
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import umap
from sklearn.cluster import KMeans

# COMMAND ----------

df = spark.table("dev_gold.pitchvision.gps_data_agregado_por_jugador")

print(df.count())
display(df.limit(10))

# COMMAND ----------

id_col = "id_unico_deportista"

# Columnas que NO queremos usar en el clustering
cols_excluir = [
    id_col,
    "n_sesiones_validas",
    "n_sesiones",
    "n_temporadas",
    "avg_participation_minutes",
    "avg_session_minutes",
    "pct_sprint_high_intensity_distance_imputed"
]

# Tipos numéricos válidos en PySpark
numeric_types = [
    "int",
    "bigint",
    "double",
    "float",
    "decimal",
    "long",
    "smallint",
    "tinyint"
]

# Todas las columnas numéricas menos las excluidas
features_cluster = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString().split("(")[0] in numeric_types
    and field.name not in cols_excluir
]

print(f"Número de variables usadas para clustering: {len(features_cluster)}")
print("Variables usadas:")
for c in features_cluster:
    print(c)

# Vectorizar y escalar
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

pipeline_preprocess = Pipeline(stages=[assembler, scaler])

preprocess_model = pipeline_preprocess.fit(df)
df_cluster_ready = preprocess_model.transform(df)

df_cluster_ready.select(id_col, "features_scaled").show(5, truncate=False)

# COMMAND ----------

df_cluster_ready = df_cluster_ready.select(
    "id_unico_deportista",
    "features_scaled"
).cache()

df_cluster_ready.count()

# COMMAND ----------

df_cluster_ready.is_cached

# COMMAND ----------

from pyspark.ml.clustering import KMeans as SparkKMeans

k_values = [2, 3, 4, 5, 6, 7, 8]

elbow_results = []

for k in k_values:
    kmeans = SparkKMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=5,       
        tol=1e-3,         # tolerancia más laxa
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_cluster_ready)
    wssse = model.summary.trainingCost
    
    elbow_results.append((k, float(wssse)))
    print(f"k={k} | WSSSE={wssse:.2f}")

# COMMAND ----------


elbow_df = spark.createDataFrame(
    elbow_results,
    ["k", "wssse"]
)

elbow_df.show()

# COMMAND ----------

elbow_pd = elbow_df.toPandas()
plt.figure(figsize=(8, 5))
plt.plot(elbow_pd["k"], elbow_pd["wssse"], marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("WSSSE")
plt.title("Método del codo (Elbow Method)")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline: KMeans

# COMMAND ----------

from pyspark.ml.clustering import KMeans as SparkKMeans

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_values = [3, 4, 5]
resultados_k = []

for k in k_values:
    kmeans = SparkKMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_cluster_ready)
    predictions = model.transform(df_cluster_ready)
    
    silhouette = evaluator.evaluate(predictions)
    wssse = model.summary.trainingCost
    
    n_por_cluster = (
        predictions
        .groupBy("cluster")
        .count()
        .orderBy("cluster")
        .collect()
    )
    
    cluster_sizes = {row["cluster"]: row["count"] for row in n_por_cluster}
    
    resultados_k.append((k, float(silhouette), float(wssse), str(cluster_sizes)))
    
    print(f"k={k} | silhouette={silhouette:.4f} | WSSSE={wssse:.2f} | tamaños={cluster_sizes}")

resultados_k_df = spark.createDataFrame(
    resultados_k,
    ["k", "silhouette", "wssse", "cluster_sizes"]
)

resultados_k_df.show(truncate=False)

# COMMAND ----------

df_pca_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input.count()

# 1. PCA a 2 dimensiones
pca = PCA(
    k=2,
    inputCol="features_scaled",
    outputCol="pca_features"
)

pca_model = pca.fit(df_pca_input)
df_pca = pca_model.transform(df_pca_input)

print("Varianza explicada por PC1 y PC2:")
print(pca_model.explainedVariance)

# 2. Probar y plotear k=3, 4 y 5 sobre el espacio PCA
k_values = [3, 4, 5]

for k in k_values:
    kmeans = SparkKMeans(
        featuresCol="features_scaled",   # clustering sobre todas las variables, NO sobre PCA
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_pca)
    df_pred = model.transform(df_pca)
    
    # Pasar a pandas para plotear
    pdf = (
        df_pred
        .select("id_unico_deportista", "cluster", "pca_features")
        .toPandas()
    )
    
    pdf["PC1"] = pdf["pca_features"].apply(lambda x: float(x[0]))
    pdf["PC2"] = pdf["pca_features"].apply(lambda x: float(x[1]))
    
    plt.figure(figsize=(8, 6))
    
    for cluster_id in sorted(pdf["cluster"].unique()):
        subset = pdf[pdf["cluster"] == cluster_id]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans k={k} proyectado en PCA 2D")
    plt.legend()
    plt.grid(True)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## KMeans + PCA

# COMMAND ----------

df_pca_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input.count()

# Número máximo de componentes a probar
# No puede ser mayor que el número de variables originales
n_features = len(features_cluster)

max_components = min(n_features, 20)  # puedes subirlo si tienes muchas variables

pca = PCA(
    k=max_components,
    inputCol="features_scaled",
    outputCol="pca_features"
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

pca_variance_pd

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(
    pca_variance_pd["component"],
    pca_variance_pd["explained_variance"],
    marker="o"
)
plt.xlabel("Componente principal")
plt.ylabel("Varianza explicada")
plt.title("Varianza explicada por cada componente PCA")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    pca_variance_pd["component"],
    pca_variance_pd["cumulative_explained_variance"],
    marker="o"
)
plt.axhline(0.80, linestyle="--", label="80%")
plt.axhline(0.90, linestyle="--", label="90%")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("Varianza explicada acumulada PCA")
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

FEATURES_COL = "features_scaled"
df = df_cluster_ready
recommended_components = 12

feature_names = features_cluster

print(f"Número de variables: {len(feature_names)}")
print(feature_names[:10])

loadings_all_pd = pd.DataFrame({
    "variable": feature_names
})

pca_models_explain = {}

for n_comp in range(1, recommended_components + 1):
    print(f"Entrenando PCA con {n_comp} componentes...")
    pca = PCA(
        k=n_comp,
        inputCol=FEATURES_COL,
        outputCol=f"features_pca_{n_comp}"
    )
    pca_model = pca.fit(df)
    pca_models_explain[n_comp] = pca_model
    pc_matrix = pca_model.pc.toArray()
    for pc_idx in range(n_comp):
        col_name = f"PCA_{n_comp}_PC{pc_idx + 1}"
        loadings_all_pd[col_name] = pc_matrix[:, pc_idx]

display(loadings_all_pd)

# COMMAND ----------

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.gps_loadingsPCA")

# COMMAND ----------

df_pca_kmeans_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_kmeans_input.count()

# Valores a probar
n_components_values = [5, 6, 8, 10]
k_values = [3, 4, 5]

evaluator = ClusteringEvaluator(
    featuresCol="features_pca",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

resultados = []

for n_comp in n_components_values:
    
    print(f"\nProbando PCA con {n_comp} componentes...")
    
    pca = PCA(
        k=n_comp,
        inputCol="features_scaled",
        outputCol="features_pca"
    )
    
    pca_model = pca.fit(df_pca_kmeans_input)
    df_pca_tmp = pca_model.transform(df_pca_kmeans_input).cache()
    
    explained_variance = float(sum(pca_model.explainedVariance))
    
    for k in k_values:
        
        kmeans = SparkKMeans(
            featuresCol="features_pca",
            predictionCol="cluster",
            k=k,
            seed=42,
            maxIter=50,
            tol=1e-4,
            initMode="k-means||"
        )
        
        model = kmeans.fit(df_pca_tmp)
        predictions = model.transform(df_pca_tmp)
        
        silhouette = evaluator.evaluate(predictions)
        wssse = model.summary.trainingCost
        
        cluster_sizes_rows = (
            predictions
            .groupBy("cluster")
            .count()
            .orderBy("cluster")
            .collect()
        )
        
        cluster_sizes = {
            row["cluster"]: row["count"]
            for row in cluster_sizes_rows
        }
        
        resultados.append((
            int(n_comp),
            int(k),
            float(explained_variance),
            float(silhouette),
            float(wssse),
            str(cluster_sizes)
        ))
        
        print(
            f"n_comp={n_comp} | k={k} | "
            f"var_exp={explained_variance:.4f} | "
            f"silhouette={silhouette:.4f} | "
            f"WSSSE={wssse:.2f} | "
            f"tamaños={cluster_sizes}"
        )
    
    df_pca_tmp.unpersist()

resultados_pca_kmeans_df = spark.createDataFrame(
    resultados,
    [
        "n_components",
        "k",
        "explained_variance",
        "silhouette",
        "wssse",
        "cluster_sizes"
    ]
)

resultados_pca_kmeans_df.orderBy(
    F.desc("silhouette")
).show(100, truncate=False)

# COMMAND ----------

resultados_pca_kmeans_df.orderBy(
    "n_components",
    "k"
).show(100, truncate=False)

# COMMAND ----------

from functools import reduce

# Lista de combinaciones de PCA y K ya probadas
combinaciones = [
    (5, 3), (5, 4), (5, 5),
    (6, 3), (6, 4), (6, 5),
    (8, 3), (8, 4), (8, 5),
    (10, 3), (10, 4), (10, 5)
]

dfs_clusters = []

for n_comp, k in combinaciones:
    pca = PCA(
        k=n_comp,
        inputCol="features_scaled",
        outputCol="features_pca"
    )
    pca_model = pca.fit(df_cluster_ready)
    df_pca_tmp = pca_model.transform(df_cluster_ready)
    
    kmeans = SparkKMeans(
        featuresCol="features_pca",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    model = kmeans.fit(df_pca_tmp)
    df_pred = model.transform(df_pca_tmp).select(
        "id_unico_deportista",
        F.col("cluster").alias(f"cluster_PCA{n_comp}_K{k}")
    )
    dfs_clusters.append(df_pred)

# Unir todas las columnas de cluster por id_unico_deportista
df_clusters_all = reduce(
    lambda df1, df2: df1.join(df2, on="id_unico_deportista", how="inner"),
    dfs_clusters
)

# Seleccionar solo las variables originales del DataFrame original
df_jugadores_originales = spark.table("dev_gold.pitchvision.gps_data_agregado_por_jugador").select(
    [id_col] + features_cluster
)

# Unir con los clusters
df_resultados = df_jugadores_originales.join(df_clusters_all, on="id_unico_deportista", how="inner")

display(df_resultados)

# COMMAND ----------

n_components_elegido = 5
k_elegido = 4

# Dataset base
df_plot_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_plot_input.count()

pca = PCA(
    k=n_components_elegido,
    inputCol="features_scaled",
    outputCol="features_pca"
)

pca_model = pca.fit(df_plot_input)
df_pca = pca_model.transform(df_plot_input)

print("Varianza explicada por componente:")
print(pca_model.explainedVariance)

print("Varianza explicada acumulada:")
print(float(sum(pca_model.explainedVariance)))

kmeans = SparkKMeans(
    featuresCol="features_pca",
    predictionCol="cluster",
    k=k_elegido,
    seed=42,
    maxIter=50,
    tol=1e-4,
    initMode="k-means||"
)

modelo_kmeans_pca = kmeans.fit(df_pca)
df_clusters_pca = modelo_kmeans_pca.transform(df_pca)

evaluator = ClusteringEvaluator(
    featuresCol="features_pca",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

silhouette = evaluator.evaluate(df_clusters_pca)
wssse = modelo_kmeans_pca.summary.trainingCost

print(f"k={k_elegido} | n_components={n_components_elegido}")
print(f"Silhouette: {silhouette:.4f}")
print(f"WSSSE: {wssse:.2f}")

df_clusters_pca.groupBy("cluster").count().orderBy("cluster").show()

pdf_plot = (
    df_clusters_pca
    .select("id_unico_deportista", "cluster", "features_pca")
    .toPandas()
)

pdf_plot["PC1"] = pdf_plot["features_pca"].apply(lambda x: float(x[0]))
pdf_plot["PC2"] = pdf_plot["features_pca"].apply(lambda x: float(x[1]))

plt.figure(figsize=(9, 6))

for cluster_id in sorted(pdf_plot["cluster"].unique()):
    subset = pdf_plot[pdf_plot["cluster"] == cluster_id]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    f"KMeans con PCA | k={k_elegido}, componentes={n_components_elegido}, "
    f"silhouette={silhouette:.3f}"
)
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## KMeans +UMAP

# COMMAND ----------

# Pasamos a pandas solo id + vector escalado
pdf_umap_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .toPandas()
)

# Convertir Vector de Spark a matriz numpy
X_scaled = np.vstack(
    pdf_umap_input["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_umap_input["id_unico_deportista"].values

print("Shape matriz clustering:", X_scaled.shape)

# COMMAND ----------

# Parámetros UMAP
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

X_umap_2d = umap_model.fit_transform(X_scaled)

resultados_umap = []

for k in [3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    
    clusters = kmeans.fit_predict(X_umap_2d)
    
    silhouette = silhouette_score(X_umap_2d, clusters)
    inertia = kmeans.inertia_
    
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    resultados_umap.append({
        "k": k,
        "silhouette": silhouette,
        "inertia": inertia,
        "cluster_sizes": cluster_sizes
    })
    
    print(
        f"k={k} | silhouette={silhouette:.4f} | "
        f"inertia={inertia:.2f} | tamaños={cluster_sizes}"
    )

# COMMAND ----------

k_values = [3, 4, 5]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

resultados_umap_plots = []

for ax, k in zip(axes, k_values):
    
    kmeans_umap = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    
    clusters = kmeans_umap.fit_predict(X_umap_2d)
    
    silhouette = silhouette_score(X_umap_2d, clusters)
    inertia = kmeans_umap.inertia_
    
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    resultados_umap_plots.append({
        "k": k,
        "silhouette": silhouette,
        "inertia": inertia,
        "cluster_sizes": cluster_sizes
    })
    
    pdf_plot = pd.DataFrame({
        "id_unico_deportista": player_ids,
        "UMAP1": X_umap_2d[:, 0],
        "UMAP2": X_umap_2d[:, 1],
        "cluster": clusters
    })
    
    for cluster_id in sorted(pdf_plot["cluster"].unique()):
        subset = pdf_plot[pdf_plot["cluster"] == cluster_id]
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )
    
    ax.set_title(
        f"UMAP + KMeans | k={k}\n"
        f"Silhouette={silhouette:.3f}"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# Tabla resumen
resultados_umap_plots_df = pd.DataFrame(resultados_umap_plots)
resultados_umap_plots_df

# COMMAND ----------

# Añadir etiquetas de cluster UMAP para k=3,4,5 como columnas a df_resultados

pdf_umap_labels_all = pd.DataFrame({"id_unico_deportista": player_ids})

for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_umap_2d)
    pdf_umap_labels_all[f"cluster_umap_k{k}"] = clusters

df_umap_labels_all = spark.createDataFrame(pdf_umap_labels_all)

df_resultados_umap = df_resultados.join(
    df_umap_labels_all,
    on="id_unico_deportista",
    how="left"
)

display(df_resultados_umap)

# COMMAND ----------

# Añadir etiquetas de cluster UMAP para k=3,4,5 como columnas a df_resultados

pdf_umap_labels_all = pd.DataFrame({"id_unico_deportista": player_ids})

for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_umap_2d)
    pdf_umap_labels_all[f"cluster_umap_k{k}"] = clusters

df_umap_labels_all = spark.createDataFrame(pdf_umap_labels_all)

df_resultados_umap = df_resultados.join(
    df_umap_labels_all,
    on="id_unico_deportista",
    how="left"
)

display(df_resultados_umap)

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a analizar más en detalle el resultado para k=4

# COMMAND ----------

# Crear pdf_umap_plot con k=4 sobre UMAP
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=20)
clusters_k4 = kmeans_k4.fit_predict(X_umap_2d)

pdf_umap_plot = pd.DataFrame({
    "id_unico_deportista": player_ids,
    "UMAP1": X_umap_2d[:, 0],
    "UMAP2": X_umap_2d[:, 1],
    "cluster": clusters_k4
})

df_clusters_k4 = spark.createDataFrame(
    pdf_umap_plot[["id_unico_deportista", "cluster"]]
)

# Unir cluster al dataframe agregado de jugadores
df_cluster_interpretacion = df.join(
    df_clusters_k4,
    on="id_unico_deportista",
    how="inner"
)

df_cluster_interpretacion.groupBy("cluster").agg(
    F.count("*").alias("n_jugadores")
).orderBy("cluster").show()

# COMMAND ----------


# Join original features from source table with cluster labels
df_cluster_interpretacion_full = spark.table("dev_gold.pitchvision.gps_data_agregado_por_jugador").select(
    [id_col] + features_cluster
).join(
    df_cluster_interpretacion.select("id_unico_deportista", "cluster"),
    on="id_unico_deportista",
    how="inner"
)

agg_exprs = [
    F.count("*").alias("n_jugadores")
]

for col_name in features_cluster:
    agg_exprs.append(
        F.avg(col_name).alias(f"{col_name}_mean")
    )

df_cluster_means = (
    df_cluster_interpretacion_full
    .groupBy("cluster")
    .agg(*agg_exprs)
    .orderBy("cluster")
)

display(df_cluster_means)

# COMMAND ----------

cluster_means_pd = df_cluster_means.toPandas()
cluster_means_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explicabilidad

# COMMAND ----------

# Join original features from source table with cluster labels
df_cluster_interpretacion_full = spark.table("dev_gold.pitchvision.gps_data_agregado_por_jugador").select(
    [id_col] + features_cluster
).join(
    df_cluster_interpretacion.select("id_unico_deportista", "cluster"),
    on="id_unico_deportista",
    how="inner"
)

df_z = df_cluster_interpretacion_full

# Calcular z-score de cada feature
for col_name in features_cluster:
    stats = df_cluster_interpretacion_full.agg(
        F.avg(col_name).alias("mean"),
        F.stddev(col_name).alias("std")
    ).collect()[0]
    
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    if std_val is not None and std_val != 0:
        df_z = df_z.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z = df_z.withColumn(
            f"{col_name}_z",
            F.lit(0.0)
        )

features_z = [f"{col}_z" for col in features_cluster]

agg_z_exprs = [
    F.count("*").alias("n_jugadores")
]

for col_name in features_z:
    agg_z_exprs.append(
        F.avg(col_name).alias(col_name)
    )

df_cluster_z_means = (
    df_z
    .groupBy("cluster")
    .agg(*agg_z_exprs)
    .orderBy("cluster")
)

df_cluster_z_means.show(truncate=False)

# COMMAND ----------

cluster_z_pd = df_cluster_z_means.toPandas()

# Columnas z-score
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
)

importance_pd.head(30)

# COMMAND ----------

# Nos quedamos con las variables más diferenciadoras
top_n = 20
top_vars = importance_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_pd.set_index("cluster")[top_z_cols]

plt.figure(figsize=(14, 6))
plt.imshow(heatmap_data, aspect="auto")
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
plt.title("Perfil medio estandarizado por cluster")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC El heatmap muestra el perfil medio estandarizado de cada cluster, es decir, en qué variables cada grupo está por encima o por debajo de la media global. El cluster 0 destaca claramente por valores altos en variables de alta intensidad, sprint, HSR, aceleraciones y carga, por lo que representa un perfil físico muy exigente. El cluster 1 aparece bajo en la mayoría de métricas explosivas e intensas, aunque con cierta distancia total media elevada, lo que podría indicar un perfil menos explosivo y más constante. El cluster 2 muestra un perfil más equilibrado, con buenos valores de aceleración, velocidad e intensidad, pero sin ser tan extremo como el cluster 0. Por último, el cluster 3 parece caracterizarse por mayor volumen total y carga, especialmente en distancia total y TSS, pero con menor peso relativo de sprint y HSR. En conjunto, los clusters parecen separar jugadores de alta intensidad, perfiles más aeróbicos/constantes, perfiles equilibrados y perfiles de mayor volumen de carga.

# COMMAND ----------

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

importance_pd.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC La características que más varía entre clusters es max_acceleration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## HDBSCAN

# COMMAND ----------

pdf_hdbscan_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .toPandas()
)

X_scaled = np.vstack(
    pdf_hdbscan_input["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_hdbscan_input["id_unico_deportista"].values

print("Shape:", X_scaled.shape)

# COMMAND ----------

min_cluster_sizes = [3, 4, 5, 6, 8, 10]
min_samples_values = [1, 2, 3, 5]

resultados_hdbscan = []

for min_cluster_size in min_cluster_sizes:
    for min_samples in min_samples_values:
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean"
        )
        
        labels = clusterer.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        pct_noise = n_noise / len(labels)
        
        # Silhouette solo si hay al menos 2 clusters y excluyendo ruido
        mask_valid = labels != -1
        
        if n_clusters >= 2 and mask_valid.sum() > n_clusters:
            silhouette = silhouette_score(
                X_scaled[mask_valid],
                labels[mask_valid]
            )
        else:
            silhouette = np.nan
        
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        resultados_hdbscan.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "pct_noise": pct_noise,
            "silhouette_no_noise": silhouette,
            "cluster_sizes": cluster_sizes
        })

resultados_hdbscan_pd = (
    pd.DataFrame(resultados_hdbscan)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
)

resultados_hdbscan_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### HDBSCAN + PCA

# COMMAND ----------

from sklearn.decomposition import PCA as SklearnPCA

n_components_values = [3, 4, 5, 6, 7, 8]
min_cluster_sizes = [2, 3, 4, 5, 6, 8, 10]
min_samples_values = [1, 2, 3, 5]

resultados_hdbscan_pca = []

for n_comp in n_components_values:
    
    pca = SklearnPCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean"
            )
            
            labels = clusterer.fit_predict(X_pca)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            pct_noise = n_noise / len(labels)
            
            mask_valid = labels != -1
            
            if n_clusters >= 2 and mask_valid.sum() > n_clusters:
                silhouette = silhouette_score(
                    X_pca[mask_valid],
                    labels[mask_valid]
                )
            else:
                silhouette = np.nan
            
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            resultados_hdbscan_pca.append({
                "n_components": n_comp,
                "explained_variance": explained_variance,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": int(n_noise),
                "pct_noise": pct_noise,
                "silhouette_no_noise": silhouette,
                "cluster_sizes": cluster_sizes
            })

resultados_hdbscan_pca_pd = (
    pd.DataFrame(resultados_hdbscan_pca)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
)

resultados_hdbscan_pca_pd.head(30)

# COMMAND ----------

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

X_umap_2d = umap_model.fit_transform(X_scaled)

min_cluster_sizes = [3, 4, 5, 6, 8, 10, 20]
min_samples_values = [1, 2, 3, 5]

resultados_hdbscan_umap = []

for min_cluster_size in min_cluster_sizes:
    for min_samples in min_samples_values:
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean"
        )
        
        labels = clusterer.fit_predict(X_umap_2d)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        pct_noise = n_noise / len(labels)
        
        mask_valid = labels != -1
        
        if n_clusters >= 2 and mask_valid.sum() > n_clusters:
            silhouette = silhouette_score(
                X_umap_2d[mask_valid],
                labels[mask_valid]
            )
        else:
            silhouette = np.nan
        
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        resultados_hdbscan_umap.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "pct_noise": pct_noise,
            "silhouette_no_noise": silhouette,
            "cluster_sizes": cluster_sizes
        })

resultados_hdbscan_umap_pd = (
    pd.DataFrame(resultados_hdbscan_umap)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
)

resultados_hdbscan_umap_pd.head(30)

# COMMAND ----------

min_cluster_size_elegido = 10
min_samples_elegido = 2

clusterer_final = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size_elegido,
    min_samples=min_samples_elegido,
    metric="euclidean"
)

labels_hdbscan = clusterer_final.fit_predict(X_umap_2d)

n_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
n_noise = np.sum(labels_hdbscan == -1)
pct_noise = n_noise / len(labels_hdbscan)

mask_valid = labels_hdbscan != -1

if n_clusters >= 2 and mask_valid.sum() > n_clusters:
    silhouette = silhouette_score(
        X_umap_2d[mask_valid],
        labels_hdbscan[mask_valid]
    )
else:
    silhouette = np.nan

print(f"N clusters: {n_clusters}")
print(f"Ruido: {n_noise} jugadores ({pct_noise:.2%})")
print(f"Silhouette sin ruido: {silhouette:.4f}")

pdf_hdbscan_plot = pd.DataFrame({
    "id_unico_deportista": player_ids,
    "UMAP1": X_umap_2d[:, 0],
    "UMAP2": X_umap_2d[:, 1],
    "cluster": labels_hdbscan
})

plt.figure(figsize=(9, 6))

for cluster_id in sorted(pdf_hdbscan_plot["cluster"].unique()):
    subset = pdf_hdbscan_plot[pdf_hdbscan_plot["cluster"] == cluster_id]
    
    label = "Ruido (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
    
    plt.scatter(
        subset["UMAP1"],
        subset["UMAP2"],
        label=label,
        alpha=0.75
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title(
    f"UMAP + HDBSCAN | min_cluster_size={min_cluster_size_elegido}, "
    f"min_samples={min_samples_elegido}"
)
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

df_hdbscan_clusters_pd = pdf_hdbscan_plot[
    ["id_unico_deportista", "UMAP1", "UMAP2", "cluster"]
].copy()

df_hdbscan_clusters = spark.createDataFrame(df_hdbscan_clusters_pd)

df_con_clusters_hdbscan = df.join(
    df_hdbscan_clusters,
    on="id_unico_deportista",
    how="left"
)

df_con_clusters_hdbscan.groupBy("cluster").agg(
    F.count("*").alias("n_jugadores")
).orderBy("cluster").show()

# COMMAND ----------

df_hdbscan_labels = df_hdbscan_clusters.select("id_unico_deportista", F.col("cluster").alias("cluster_hdbscan_umap"))

df_resultados_umap_hdbscan = df_resultados_umap.join(
    df_hdbscan_labels,
    on="id_unico_deportista",
    how="left"
)

display(df_resultados_umap_hdbscan)

# COMMAND ----------

# Join original features from source table with HDBSCAN cluster labels (excluding noise)
df_hdbscan_interpretacion = spark.table("dev_gold.pitchvision.gps_data_agregado_por_jugador").select(
    [id_col] + features_cluster
).join(
    df_con_clusters_hdbscan.filter(F.col("cluster") != -1).select("id_unico_deportista", "cluster"),
    on="id_unico_deportista",
    how="inner"
)

agg_exprs = [
    F.count("*").alias("n_jugadores")
]

for col_name in features_cluster:
    agg_exprs.append(
        F.avg(col_name).alias(f"{col_name}_mean")
    )

df_hdbscan_cluster_means = (
    df_hdbscan_interpretacion
    .groupBy("cluster")
    .agg(*agg_exprs)
    .orderBy("cluster")
)

display(df_hdbscan_cluster_means)

# COMMAND ----------

# Partimos de df_hdbscan_interpretacion, excluyendo ruido (-1)
df_z_hdbscan = df_hdbscan_interpretacion

# Calcular z-score de cada variable respecto a todos los jugadores asignados a cluster
for col_name in features_cluster:
    stats = df_hdbscan_interpretacion.agg(
        F.avg(col_name).alias("mean"),
        F.stddev(col_name).alias("std")
    ).collect()[0]
    
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    if std_val is not None and std_val != 0:
        df_z_hdbscan = df_z_hdbscan.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_z_hdbscan = df_z_hdbscan.withColumn(
            f"{col_name}_z",
            F.lit(0.0)
        )

features_z_hdbscan = [f"{col}_z" for col in features_cluster]

agg_z_exprs = [
    F.count("*").alias("n_jugadores")
]

for col_name in features_z_hdbscan:
    agg_z_exprs.append(
        F.avg(col_name).alias(col_name)
    )

df_hdbscan_cluster_z_means = (
    df_z_hdbscan
    .groupBy("cluster")
    .agg(*agg_z_exprs)
    .orderBy("cluster")
)

cluster_z_hdbscan_pd = df_hdbscan_cluster_z_means.toPandas()

cluster_z_hdbscan_pd

# COMMAND ----------

# Columnas z-score
z_cols = [c for c in cluster_z_hdbscan_pd.columns if c.endswith("_z")]

# Ranking previo para pintar solo las más diferenciadoras
importance_rows = []

for col in z_cols:
    values = cluster_z_hdbscan_pd[col]
    
    importance_rows.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_hdbscan_pd = (
    pd.DataFrame(importance_rows)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

# Pintar top variables
top_n = min(20, len(importance_hdbscan_pd))

top_vars = importance_hdbscan_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_hdbscan_pd.set_index("cluster")[top_z_cols]

plt.figure(figsize=(14, 6))
plt.imshow(heatmap_data, aspect="auto")
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
plt.title("HDBSCAN: perfil medio estandarizado por cluster")
plt.tight_layout()
plt.show()

# COMMAND ----------

importance_hdbscan_pd.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Igual que antes las variable que más sepaa es max_acceleration

# COMMAND ----------

# MAGIC %md
# MAGIC Nos quedamos con las dos agrupaciones en 4 clusteres que fueron las que mejores resultados dieron, tanto con KMeans como Hdbscan

# COMMAND ----------

display(df_resultados_umap_hdbscan)

# COMMAND ----------

display(df_con_clusters_hdbscan.head(10))

# COMMAND ----------

df_resultados_umap_hdbscan.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clustersresults_Gps")
