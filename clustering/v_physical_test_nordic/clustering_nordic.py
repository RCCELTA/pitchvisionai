# Databricks notebook source
# MAGIC %pip install hdbscan
# MAGIC %pip install umap-learn

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql.types import NumericType
import seaborn as sns
import umap as umap
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import hdbscan
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from collections import defaultdict
from sklearn.cluster import KMeans as SklearnKMeans
from pyspark.ml import Pipeline

# COMMAND ----------

df_cluster = spark.table("dev_gold.pitchvision.nordic_agregado_por_jugador")

# COMMAND ----------

cols_to_drop = [
    "mdc_left_max_force_n_mean",
    "mdc_left_avg_force_n_mean",
    "mdc_left_impulse_ns_mean",
    "mdc_left_torque_nm_mean",
    "mdc_right_max_force_n_mean",
    "mdc_right_avg_force_n_mean",
    "mdc_right_impulse_ns_mean",
    "mdc_right_torque_nm_mean",
    "mdc_left_max_force_n_mean",
    "mdc_left_avg_force_n_mean",
    "mdc_left_impulse_ns_mean",
    "mdc_left_torque_nm_mean",
    "mdc_right_max_force_n_mean",
    "left_repetitions_mean",
    "right_repetitions_mean"




]

df_cluster = df_cluster.drop(*cols_to_drop)
display(df_cluster.limit(10))

# COMMAND ----------

ID_COL = "id_unico_deportista"

# Columnas que NO queremos usar para clusterizar
cols_excluir_cluster = [
    ID_COL,
    "n_tests_agregados",
    "n_temporadas_agregadas",
    "temporada_fuente",
    "first_test_date",
    "last_test_date"
]

# Seleccionar variables numéricas candidatas para clustering
features_cluster = [
    field.name
    for field in df_cluster.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
]

print(f"Nº variables para clustering: {len(features_cluster)}")
for c in features_cluster:
    print(c)

# COMMAND ----------

# Vectorizar
assembler = VectorAssembler(
    inputCols=features_cluster,
    outputCol="features_raw"
)

# Escalar
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline = Pipeline(stages=[assembler, scaler])

pipeline_model = pipeline.fit(df_cluster)

df_cluster_ready = (
    pipeline_model
    .transform(df_cluster)
    .select(
        ID_COL,
        "features_raw",
        "features_scaled",
        *features_cluster,
        *[c for c in df_cluster.columns if c.startswith("pct_")],
        *[c for c in ["n_tests_agregados", "n_temporadas_agregadas", "temporada_fuente"] if c in df_cluster.columns]
    )
    .cache()
)

df_cluster_ready.count()

display(df_cluster_ready.limit(10))

# COMMAND ----------

null_check = df_cluster.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in features_cluster
])

display(null_check)

# COMMAND ----------

# Vectorizar
assembler = VectorAssembler(
    inputCols=features_cluster,
    outputCol="features_raw"
)

# Escalar
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline = Pipeline(stages=[assembler, scaler])

pipeline_model = pipeline.fit(df_cluster)

df_cluster_ready = (
    pipeline_model
    .transform(df_cluster)
    .select(
        ID_COL,
        "features_raw",
        "features_scaled",
        *features_cluster,
        *[c for c in df_cluster.columns if c.startswith("pct_")],
        *[c for c in ["n_tests_agregados", "n_temporadas_agregadas", "temporada_fuente"] if c in df_cluster.columns]
    )
    .cache()
)

df_cluster_ready.count()

display(df_cluster_ready.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Método del codo para el K óptimo

# COMMAND ----------

df_kmeans_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_kmeans_input.count()

# Valores de k a probar
k_values = range(2, 11)

elbow_results = []

for k in k_values:
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features_scaled",
        predictionCol="cluster",
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_kmeans_input)
    wssse = model.summary.trainingCost
    
    elbow_results.append((k, float(wssse)))
    
    print(f"k={k} | WSSSE={wssse:.2f}")

elbow_df = spark.createDataFrame(
    elbow_results,
    ["k", "wssse"]
)

display(elbow_df)

# COMMAND ----------

elbow_pd = elbow_df.toPandas().sort_values("k")

plt.figure(figsize=(8, 5))

plt.plot(
    elbow_pd["k"],
    elbow_pd["wssse"],
    marker="o"
)

plt.xlabel("Número de clusters (k)")
plt.ylabel("WSSSE")
plt.title("Método del codo - KMeans baseline")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC El primer candidato razonable es k=4.

# COMMAND ----------

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_values = [2, 3, 4, 5, 6, 7, 8]

results_kmeans = []
models_kmeans = {}
predictions_kmeans = {}

for k in k_values:
    print("-" * 80)
    print(f"Entrenando KMeans baseline k={k}")
    
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features_scaled",
        predictionCol="cluster",
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_kmeans_input)
    predictions = model.transform(df_kmeans_input)
    
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
    
    results_kmeans.append((
        int(k),
        float(silhouette),
        float(wssse),
        str(cluster_sizes)
    ))
    
    models_kmeans[k] = model
    predictions_kmeans[k] = predictions
    
    print(
        f"k={k} | silhouette={silhouette:.4f} | "
        f"WSSSE={wssse:.2f} | tamaños={cluster_sizes}"
    )

kmeans_results_df = spark.createDataFrame(
    results_kmeans,
    ["k", "silhouette", "wssse", "cluster_sizes"]
)

display(
    kmeans_results_df.orderBy(F.desc("silhouette"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### KMeans +PCA

# COMMAND ----------

# Dataset ya escalado
df_pca_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input.count()

# Número máximo de componentes
n_features = len(features_cluster)
max_components = min(n_features, 20)

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

plt.figure(figsize=(8, 5))

plt.plot(
    pca_variance_pd["component"],
    pca_variance_pd["explained_variance"],
    marker="o"
)

plt.xlabel("Componente principal")
plt.ylabel("Varianza explicada")
plt.title("Varianza explicada por componente PCA")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    pca_variance_pd["component"],
    pca_variance_pd["cumulative_explained_variance"],
    marker="o"
)

plt.axhline(0.80, linestyle="--", label="80%")
plt.axhline(0.90, linestyle="--", label="90%")
plt.axhline(0.95, linestyle="--", label="95%")

plt.xlabel("Número de componentes PCA")
plt.ylabel("Varianza explicada acumulada")
plt.title("Varianza explicada acumulada PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

FEATURES_COL = "features_scaled"
df = df_cluster_ready
recommended_components = 6

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

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.nordic_loadingsPCA")

# COMMAND ----------

# Ajusta según la curva de varianza explicada
n_components_values = [2, 3, 4, 6]
n_components_values = [n for n in n_components_values if n <= n_features]

k_values = [3, 4, 5, 6, 7]

results_pca_kmeans = []
models_pca_kmeans = {}
predictions_pca_kmeans = {}
pca_models = {}

for n_components in n_components_values:
    
    print("=" * 80)
    print(f"PCA con {n_components} componentes")
    
    pca = PCA(
        k=n_components,
        inputCol="features_scaled",
        outputCol=f"features_pca_{n_components}"
    )
    
    pca_model = pca.fit(df_pca_input)
    df_pca_tmp = pca_model.transform(df_pca_input).cache()
    df_pca_tmp.count()
    
    explained_variance_total = float(sum(pca_model.explainedVariance))
    pca_models[n_components] = pca_model
    
    evaluator = ClusteringEvaluator(
        featuresCol=f"features_pca_{n_components}",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    
    for k in k_values:
        
        print(f"Entrenando PCA({n_components}) + KMeans k={k}")
        
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
            f"n_comp={n_components} | k={k} | "
            f"var_exp={explained_variance_total:.4f} | "
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
    pca_kmeans_results_df.orderBy(F.desc("silhouette"))
)

# COMMAND ----------

# Crear un DataFrame con todas las características originales de cada jugador
df_resultados = df_cluster_ready.select(ID_COL, *features_cluster)

# Para cada combinación (n_components, k) de la celda anterior, añadir la columna de cluster
for n_components, k in predictions_pca_kmeans.keys():
    pred_df = predictions_pca_kmeans[(n_components, k)].select(ID_COL, "cluster")
    col_name = f"cluster_PCA{n_components}_K{k}"
    df_resultados = df_resultados.join(
        pred_df.withColumnRenamed("cluster", col_name),
        on=ID_COL,
        how="left"
    )

display(df_resultados)

# COMMAND ----------

# MAGIC %md
# MAGIC #### mejores resultados por k

# COMMAND ----------


w = Window.partitionBy("k").orderBy(F.desc("silhouette"))

best_pca_by_k_df = (
    pca_kmeans_results_df
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("k")
)

display(best_pca_by_k_df)

# COMMAND ----------

best_pca_by_k_pd = best_pca_by_k_df.toPandas().sort_values("k")

plt.figure(figsize=(9, 5))

plt.plot(
    best_pca_by_k_pd["k"],
    best_pca_by_k_pd["silhouette"],
    marker="o"
)

for _, row in best_pca_by_k_pd.iterrows():
    plt.text(
        row["k"],
        row["silhouette"],
        f"PCA {int(row['pca_components'])}",
        fontsize=9,
        ha="center",
        va="bottom"
    )

plt.xlabel("Número de clusters k")
plt.ylabel("Mejor silhouette")
plt.title("Mejor configuración PCA + KMeans por k")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

n_plots = len(best_pca_by_k_pd)
n_cols = 3
n_rows = int((n_plots + n_cols - 1) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

# Aplanar axes para manejar 1 fila o varias
if n_rows == 1:
    axes = axes.reshape(1, -1)

axes_flat = axes.flatten()

for idx, (_, row) in enumerate(best_pca_by_k_pd.iterrows()):
    
    ax = axes_flat[idx]
    
    n_components = int(row["pca_components"])
    k = int(row["k"])
    silhouette = float(row["silhouette"])
    explained_variance = float(row["explained_variance"])
    
    # Recuperar predictions ya calculadas
    # predictions_pca_kmeans[(n_components, k)] debe existir
    pred_df = predictions_pca_kmeans[(n_components, k)]
    
    pca_col = f"features_pca_{n_components}"
    
    pdf_plot = (
        pred_df
        .select(ID_COL, pca_col, "cluster")
        .toPandas()
    )
    
    # Extraer las dos primeras componentes para pintar
    pdf_plot["PC1"] = pdf_plot[pca_col].apply(lambda x: float(x[0]))
    pdf_plot["PC2"] = pdf_plot[pca_col].apply(lambda x: float(x[1]))
    
    for cluster_id in sorted(pdf_plot["cluster"].unique()):
        subset = pdf_plot[pdf_plot["cluster"] == cluster_id]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"k={k} | PCA={n_components} comp.\n"
        f"Silhouette={silhouette:.3f} | Var={explained_variance:.1%}"
    )
    ax.grid(True)
    ax.legend()

# Ocultar ejes sobrantes
for j in range(idx + 1, len(axes_flat)):
    axes_flat[j].axis("off")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Obtener los labels de cluster para k=4, PCA4
cluster_col = "cluster_PCA2_K4"

# Unir features originales con los labels de cluster
df_cluster_interpretacion_full = (
    spark.table("dev_gold.pitchvision.nordic_agregado_por_jugador")
    .select([ID_COL] + features_cluster)
    .join(
        df_resultados.select(ID_COL, cluster_col).withColumnRenamed(cluster_col, "cluster"),
        on=ID_COL,
        how="inner"
    )
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

agg_z_exprs = [F.count("*").alias("n_jugadores")]
for col_name in features_z:
    agg_z_exprs.append(F.avg(col_name).alias(col_name))

df_cluster_z_means = (
    df_z
    .groupBy("cluster")
    .agg(*agg_z_exprs)
    .orderBy("cluster")
)

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
)

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
plt.title("Perfil medio estandarizado por cluster (k=4, PCA4)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### KMeans + UMAP

# COMMAND ----------

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

# COMMAND ----------

from sklearn.cluster import KMeans

# Grid moderado
n_components_values = [2,  5]
n_neighbors_values = [5, 15, 30]
min_dist_values = [0.0, 0.1]
k_values = [3, 4, 5, 6, 7]

results_umap_kmeans = []

umap_embeddings = {}
models_umap_kmeans = {}
predictions_umap_kmeans = {}

for n_components in n_components_values:
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            
            print(
                f"UMAP | n_components={n_components}, "
                f"n_neighbors={n_neighbors}, min_dist={min_dist}"
            )
            
            umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=42
            )
            
            X_umap = umap_model.fit_transform(X_scaled)
            
            umap_embeddings[
                (n_components, n_neighbors, min_dist)
            ] = X_umap
            
            for k in k_values:
                
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=20
                )
                
                clusters = kmeans.fit_predict(X_umap)
                
                silhouette = silhouette_score(X_umap, clusters)
                inertia = kmeans.inertia_
                
                unique, counts = np.unique(clusters, return_counts=True)
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
                
                models_umap_kmeans[
                    (n_components, n_neighbors, min_dist, k)
                ] = kmeans
                
                predictions_umap_kmeans[
                    (n_components, n_neighbors, min_dist, k)
                ] = clusters
                
                print(
                    f"n_comp={n_components} | nn={n_neighbors} | "
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

top_n = 15

plot_df = umap_kmeans_results_pd.head(top_n).copy()

plot_df["config"] = (
    "comp=" + plot_df["n_components"].astype(str)
    + " | nn=" + plot_df["n_neighbors"].astype(str)
    + " | md=" + plot_df["min_dist"].astype(str)
    + " | k=" + plot_df["k"].astype(str)
)

plt.figure(figsize=(11, 6))

plt.barh(
    plot_df["config"][::-1],
    plot_df["silhouette"][::-1]
)

plt.xlabel("Silhouette")
plt.ylabel("Configuración")
plt.title("Top configuraciones UMAP + KMeans")
plt.grid(axis="x")
plt.tight_layout()
plt.show()

# COMMAND ----------

best_umap_by_k = (
    umap_kmeans_results_pd
    .sort_values("silhouette", ascending=False)
    .groupby("k")
    .head(1)
    .sort_values("k")
    .reset_index(drop=True)
)

display(best_umap_by_k)

# COMMAND ----------

n_plots = len(best_umap_by_k)

n_cols = 3
n_rows = int((n_plots + n_cols - 1) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(6 * n_cols, 5 * n_rows)
)

# Asegurar que axes sea iterable
if n_rows == 1:
    axes = axes.reshape(1, -1)

axes_flat = axes.flatten()

best_umap_plots = []

for idx, (_, row) in enumerate(best_umap_by_k.iterrows()):
    
    ax = axes_flat[idx]
    
    n_components = int(row["n_components"])
    n_neighbors = int(row["n_neighbors"])
    min_dist = float(row["min_dist"])
    k = int(row["k"])
    silhouette = float(row["silhouette"])
    
    # Recuperar embedding y predicciones
    X_umap_best = umap_embeddings[
        (n_components, n_neighbors, min_dist)
    ]
    
    clusters_best = predictions_umap_kmeans[
        (n_components, n_neighbors, min_dist, k)
    ]
    
    # Guardar resumen
    best_umap_plots.append({
        "k": k,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "silhouette": silhouette,
        "cluster_sizes": row["cluster_sizes"]
    })
    
    # Dataframe para plot
    pdf_plot = pd.DataFrame({
        "id_unico_deportista": player_ids,
        "UMAP1": X_umap_best[:, 0],
        "UMAP2": X_umap_best[:, 1],
        "cluster": clusters_best
    })
    
    for cluster_id in sorted(pdf_plot["cluster"].unique()):
        subset = pdf_plot[pdf_plot["cluster"] == cluster_id]
        
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            label=f"Cluster {cluster_id}",
            alpha=0.75
        )
    
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"k={k} | UMAP comp={n_components}, nn={n_neighbors}, md={min_dist}\n"
        f"Silhouette={silhouette:.3f}"
    )
    ax.grid(True)
    ax.legend()

# Ocultar subplots vacíos
for j in range(idx + 1, len(axes_flat)):
    axes_flat[j].axis("off")

plt.tight_layout()
plt.show()

best_umap_plots_pd = pd.DataFrame(best_umap_plots)
display(best_umap_plots_pd)

# COMMAND ----------

from pyspark.sql.functions import lit

df_resultados_umap = df_resultados

for _, row in best_umap_by_k.iterrows():
    n_components = int(row["n_components"])
    n_neighbors = int(row["n_neighbors"])
    min_dist = float(row["min_dist"])
    k = int(row["k"])
    label_col = f"cluster_UMAP{n_components}_K{k}"
    
    clusters = predictions_umap_kmeans[(n_components, n_neighbors, min_dist, k)]
    labels_pd = pd.DataFrame({
        ID_COL: player_ids,
        label_col: clusters
    })
    labels_spark = spark.createDataFrame(labels_pd)
    
    df_resultados_umap = (
        df_resultados_umap
        .join(labels_spark, on=ID_COL, how="left")
    )

df_resultados = df_resultados_umap
display(df_resultados)

# COMMAND ----------

df_resultados.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("dev_gold.pitchvision.clustersresults_nordic")

# COMMAND ----------

# MAGIC %md
# MAGIC ### HDBSCAN

# COMMAND ----------

import numpy as np
import pandas as pd
import hdbscan
import umap

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# PCA
pca_components_values = [2, 3, 5, 8]

# UMAP
umap_configs = [
    {"n_components": 2, "n_neighbors": 5, "min_dist": 0.0},
    {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 3, "n_neighbors": 10, "min_dist": 0.1},
    {"n_components": 5, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 5, "n_neighbors": 30, "min_dist": 0.1},
]

# HDBSCAN
min_cluster_size_values = [5, 10, 15]
min_samples_values = [1, 5]

results_hdbscan = []

embeddings_hdbscan = {}
labels_hdbscan = {}
models_hdbscan = {}

def evaluar_hdbscan(X_reduced, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    pct_noise = n_noise / len(labels)

    mask_valid = labels != -1

    if n_clusters >= 2 and mask_valid.sum() > n_clusters:
        silhouette = silhouette_score(
            X_reduced[mask_valid],
            labels[mask_valid]
        )
    else:
        silhouette = np.nan

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))

    return n_clusters, n_noise, pct_noise, silhouette, cluster_sizes

for n_components in pca_components_values:
    
    print("=" * 100)
    print(f"PCA + HDBSCAN | n_components={n_components}")

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)
    explained_variance = float(pca.explained_variance_ratio_.sum())

    embeddings_hdbscan[("pca", n_components)] = X_pca

    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:

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
                f"PCA({n_components}) | mcs={min_cluster_size} | "
                f"ms={min_samples} | clusters={n_clusters} | "
                f"ruido={pct_noise:.1%} | silhouette={silhouette}"
            )

for config in umap_configs:
    
    n_components = config["n_components"]
    n_neighbors = config["n_neighbors"]
    min_dist = config["min_dist"]

    print("=" * 100)
    print(
        f"UMAP + HDBSCAN | n_components={n_components}, "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}"
    )

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42
    )

    X_umap = umap_model.fit_transform(X_scaled)

    embeddings_hdbscan[
        ("umap", n_components, n_neighbors, min_dist)
    ] = X_umap

    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:

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
                f"UMAP(comp={n_components}, nn={n_neighbors}, md={min_dist}) | "
                f"mcs={min_cluster_size} | ms={min_samples} | "
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette}"
            )


# =====================================================
# Resultados ordenados
# =====================================================

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

max_noise = 0.20

hdbscan_valid = (
    hdbscan_results_pd
    .copy()
)

# Filtrar configuraciones válidas
hdbscan_valid = hdbscan_valid[
    (hdbscan_valid["pct_noise"] < max_noise) &
    (hdbscan_valid["n_clusters"] >= 2) &
    (hdbscan_valid["silhouette_no_noise"].notna())
]

# Mejor configuración para cada número de clusters
best_hdbscan_by_k = (
    hdbscan_valid
    .sort_values("silhouette_no_noise", ascending=False)
    .groupby("n_clusters", as_index=False)
    .head(1)
    .sort_values("n_clusters")
    .reset_index(drop=True)
)

display(best_hdbscan_by_k)

# COMMAND ----------

# MAGIC %md
# MAGIC No hay ningun resultado con menos de un 20% de ruido para 2 < k < 7

# COMMAND ----------

# Opción A: todos los mejores por nº de clusters
best_results_to_plot = best_hdbscan_by_k.copy()


n_plots = len(best_results_to_plot)

n_cols = 3
n_rows = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(6 * n_cols, 5 * n_rows)
)

if n_rows == 1:
    axes = np.array([axes])

axes_flat = axes.flatten()

for idx, (_, row) in enumerate(best_results_to_plot.iterrows()):
    
    ax = axes_flat[idx]
    
    method = row["method"]
    n_components = int(row["n_components"])
    min_cluster_size = int(row["min_cluster_size"])
    min_samples = int(row["min_samples"])
    n_clusters = int(row["n_clusters"])
    pct_noise = float(row["pct_noise"])
    silhouette = float(row["silhouette_no_noise"])
    
    if method == "PCA":
        # Recuperar embedding PCA y etiquetas
        X_plot = embeddings_hdbscan[
            ("pca", n_components)
        ]
        
        labels = labels_hdbscan[
            ("pca", n_components, min_cluster_size, min_samples)
        ]
        
        title_extra = f"PCA({n_components})"
        
    elif method == "UMAP":
        n_neighbors = int(row["n_neighbors"])
        min_dist = float(row["min_dist"])
        
        # Recuperar embedding UMAP y etiquetas
        X_plot = embeddings_hdbscan[
            ("umap", n_components, n_neighbors, min_dist)
        ]
        
        labels = labels_hdbscan[
            ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
        ]
        
        title_extra = f"UMAP({n_components}), nn={n_neighbors}, md={min_dist}"
    
    else:
        raise ValueError(f"Método no reconocido: {method}")
    
    # Usamos las dos primeras dimensiones
    plot_pd = pd.DataFrame({
        "id_unico_deportista": player_ids,
        "X1": X_plot[:, 0],
        "X2": X_plot[:, 1],
        "cluster": labels
    })
    
    for cluster_id in sorted(plot_pd["cluster"].unique()):
        subset = plot_pd[plot_pd["cluster"] == cluster_id]
        
        label = "Ruido (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
        
        ax.scatter(
            subset["X1"],
            subset["X2"],
            label=label,
            alpha=0.75
        )
    
    ax.set_xlabel("Dimensión 1")
    ax.set_ylabel("Dimensión 2")
    ax.set_title(
        f"HDBSCAN | {n_clusters} clusters\n"
        f"{title_extra} | mcs={min_cluster_size}, ms={min_samples}\n"
        f"Silhouette={silhouette:.3f}, ruido={pct_noise:.1%}"
    )
    ax.grid(True)
    ax.legend(fontsize=8)

# Ocultar subplots vacíos
for j in range(idx + 1, len(axes_flat)):
    axes_flat[j].axis("off")

plt.tight_layout()
plt.show()
