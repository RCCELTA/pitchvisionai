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

df = spark.table("dev_gold.pitchvision.agregado_physical_test_cleaned")
df.display()

# COMMAND ----------

ID_COL = "id_unico_deportista"

# Columnas que no deben entrar al clustering
cols_excluir_cluster = [
    ID_COL,
    "temporada_fuente",
]


features_cluster = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster
]

print(f"Número de variables para clustering CMJ: {len(features_cluster)}")
for c in features_cluster:
    print(c)

# COMMAND ----------

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

pipeline_model = pipeline.fit(df)

df_cluster_ready = (
    pipeline_model
    .transform(df)
    .select(
        ID_COL,
        "features_raw",
        "features_scaled",
        *features_cluster
    )
    .cache()
)

df_cluster_ready.count()

display(df_cluster_ready.limit(5))

# COMMAND ----------

# Rango de k a probar
k_values = range(2, 10)

elbow_results = []

# Cachear para acelerar
df_cluster_ready_fast = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_cluster_ready_fast.count()

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
    
    model = kmeans.fit(df_cluster_ready_fast)
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
plt.title("CMJ - Método del codo KMeans")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Baseline: KMeans

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
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features_scaled",
        predictionCol="cluster",
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )

    model = kmeans.fit(df_cluster_ready)
    predictions = model.transform(df_cluster_ready)

    silhouette = evaluator.evaluate(predictions)
    wssse = model.summary.trainingCost

    cluster_sizes = (
        predictions
        .groupBy("cluster")
        .count()
        .orderBy("cluster")
        .collect()
    )

    cluster_sizes_dict = {
        row["cluster"]: row["count"]
        for row in cluster_sizes
    }

    results_kmeans.append((
        k,
        float(silhouette),
        float(wssse),
        str(cluster_sizes_dict)
    ))

    models_kmeans[k] = model
    predictions_kmeans[k] = predictions

    print(
        f"k={k} | silhouette={silhouette:.4f} | "
        f"WSSSE={wssse:.2f} | tamaños={cluster_sizes_dict}"
    )

kmeans_results_df = spark.createDataFrame(
    results_kmeans,
    ["k", "silhouette", "wssse", "cluster_sizes"]
)

display(kmeans_results_df.orderBy(F.desc("silhouette")))

# COMMAND ----------

# MAGIC %md
# MAGIC # KMeans + PCA

# COMMAND ----------

df_pca_input = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input.count()

# Número máximo de componentes
n_features = len(features_cluster)
max_components = min(n_features, 18)

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
plt.title("CMJ - Varianza explicada por componente PCA")
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
plt.title("CMJ - Varianza explicada acumulada PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

n_components_values = [2, 3, 4, 5, 6, 8, 10, 15]
n_components_values = [n for n in n_components_values if n <= n_features]

k_values = [2, 3, 4, 5, 6]

results_pca_kmeans = []
models_pca_kmeans = {}
predictions_pca_kmeans = {}
pca_models = {}

for n_components in n_components_values:
    
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
        
        cluster_sizes = (
            predictions
            .groupBy("cluster")
            .count()
            .orderBy("cluster")
            .collect()
        )
        
        cluster_sizes_dict = {
            row["cluster"]: row["count"]
            for row in cluster_sizes
        }
        
        results_pca_kmeans.append((
            int(n_components),
            float(explained_variance_total),
            int(k),
            float(silhouette),
            float(wssse),
            str(cluster_sizes_dict)
        ))
        
        models_pca_kmeans[(n_components, k)] = model
        predictions_pca_kmeans[(n_components, k)] = predictions
        
        print(
            f"n_comp={n_components} | k={k} | "
            f"var_exp={explained_variance_total:.4f} | "
            f"silhouette={silhouette:.4f} | "
            f"WSSSE={wssse:.2f} | tamaños={cluster_sizes_dict}"
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

pca_kmeans_results_pd = pca_kmeans_results_df.toPandas()

plt.figure(figsize=(9, 5))

for n_comp in sorted(pca_kmeans_results_pd["pca_components"].unique()):
    subset = (
        pca_kmeans_results_pd[
            pca_kmeans_results_pd["pca_components"] == n_comp
        ]
        .sort_values("k")
    )
    
    plt.plot(
        subset["k"],
        subset["silhouette"],
        marker="o",
        label=f"PCA {n_comp}"
    )

plt.xlabel("Número de clusters k")
plt.ylabel("Silhouette")
plt.title("CMJ - PCA + KMeans")
plt.legend(title="Componentes")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Obtener predicciones para k=4 y 2 componentes PCA
n_components_plot = 2
k_plot = 4

df_pca_input_2 = (
    df_cluster_ready
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

pca_model_2 = PCA(
    k=n_components_plot,
    inputCol="features_scaled",
    outputCol="features_pca_2"
).fit(df_pca_input_2)

df_pca_2 = pca_model_2.transform(df_pca_input_2)

kmeans_2_4 = KMeans(
    k=k_plot,
    seed=42,
    featuresCol="features_pca_2",
    predictionCol="cluster",
    maxIter=50,
    tol=1e-4,
    initMode="k-means||"
).fit(df_pca_2)

df_pca_2_pred = kmeans_2_4.transform(df_pca_2)

# Calcular silhouette
evaluator_plot = ClusteringEvaluator(
    featuresCol="features_pca_2",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)
silhouette_plot = evaluator_plot.evaluate(df_pca_2_pred)

# Convertir a Pandas para graficar
from pyspark.ml.functions import vector_to_array

plot_pd = (
    df_pca_2_pred
    .withColumn("pca_array", vector_to_array(F.col("features_pca_2")))
    .select(
        "id_unico_deportista",
        F.col("pca_array").getItem(0).alias("PC1"),
        F.col("pca_array").getItem(1).alias("PC2"),
        "cluster"
    )
    .toPandas()
)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=plot_pd,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10",
    s=60,
    alpha=0.8
)
plt.title(f"KMeans (k=4) sobre 2 componentes PCA\nSilhouette: {silhouette_plot:.4f}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Añadir predicciones de cluster a los datos originales
df_result = (
    df
    .join(
        df_pca_2_pred.select(ID_COL, "cluster"),
        on=ID_COL,
        how="left"
    )
)


# COMMAND ----------

df_result.display()

# COMMAND ----------

# DBTITLE 1,Heatmap perfil estandarizado por cluster
# Calcular media de cada variable por cluster
cluster_means = (
    df_result
    .groupBy("cluster")
    .agg(*[F.avg(c).alias(c) for c in features_cluster])
    .orderBy("cluster")
    .toPandas()
)

# Estandarizar (z-score) por columna para comparar entre clusters
cluster_z_pd = cluster_means.copy()
for c in features_cluster:
    mean_val = cluster_z_pd[c].mean()
    std_val = cluster_z_pd[c].std()
    cluster_z_pd[f"{c}_z"] = (
        (cluster_z_pd[c] - mean_val) / std_val if std_val > 0 else 0
    )

# Calcular importancia: varianza del z-score entre clusters
importance = []
for c in features_cluster:
    col_z = f"{c}_z"
    var_z = cluster_z_pd[col_z].var()
    importance.append({"variable": c, "importance": var_z})

importance_pd = (
    pd.DataFrame(importance)
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# Top N variables más diferenciadoras
top_n = 20
top_vars = importance_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_pd.set_index("cluster")[top_z_cols]

plt.figure(figsize=(14, 6))
plt.imshow(heatmap_data, aspect="auto", cmap="viridis")
plt.colorbar(label="Media estandarizada por cluster")

plt.xticks(
    ticks=range(len(top_vars)),
    labels=top_vars,
    rotation=90
)

plt.yticks(
    ticks=range(len(heatmap_data.index)),
    labels=[f"Cluster {i}" for i in heatmap_data.index]
)

plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title("Perfil medio estandarizado por cluster (Top 20 variables más diferenciadoras)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Cell 19
# Obtener el centroide de cada cluster (en el espacio PCA de 2 componentes)
centroids = kmeans_2_4.clusterCenters()

centroids_pd = pd.DataFrame(
    centroids,
    columns=["PC1", "PC2"]
)
centroids_pd["cluster"] = centroids_pd.index

# Aproximar centroides en el espacio original usando la matriz PC inversa
pc_matrix = pca_model_2.pc.toArray()  # shape: (n_features, 2)
centroids_array = np.array(centroids)  # shape: (n_clusters, 2)

# Inverse transform: X_approx = X_pca @ PC^T
centroids_original_array = centroids_array @ pc_matrix.T

centroids_original_pd = pd.DataFrame(
    centroids_original_array,
    columns=features_cluster
)
centroids_original_pd["cluster"] = range(len(centroids_original_pd))
display(centroids_original_pd)

# COMMAND ----------

df_result.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clusters_aggregatedphysical_tests_k4_pca2")

# COMMAND ----------

# MAGIC %md
# MAGIC # Explicabilidad PCA

# COMMAND ----------

N_COMPONENTS_EXPLAIN = 2

pca_explain = PCA(
    k=N_COMPONENTS_EXPLAIN,
    inputCol="features_scaled",
    outputCol="features_pca_2_explain"
)

pca_explain_model = pca_explain.fit(df_cluster_ready)

# Matriz de coeficientes/loadings
# Filas = variables originales
# Columnas = componentes principales
pc_matrix = pca_explain_model.pc.toArray()

explained_variance = pca_explain_model.explainedVariance.toArray()

print("Varianza explicada por componente:")
for i, v in enumerate(explained_variance, start=1):
    print(f"PC{i}: {v:.4f}")

print(f"\nVarianza explicada acumulada: {explained_variance.sum():.4f}")

# COMMAND ----------

feature_names = features_cluster

loadings_pd = pd.DataFrame(
    pc_matrix,
    columns=[f"PC{i}" for i in range(1, N_COMPONENTS_EXPLAIN + 1)]
)

loadings_pd["variable"] = feature_names

loadings_pd = loadings_pd[
    ["variable", "PC1", "PC2"]
]

# Añadir valor absoluto para ordenar por importancia
loadings_pd["abs_PC1"] = loadings_pd["PC1"].abs()
loadings_pd["abs_PC2"] = loadings_pd["PC2"].abs()

display(loadings_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC # UMAP

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

print("Shape matriz CMJ:", X_scaled.shape)

# COMMAND ----------

# Configuraciones moderadas de UMAP
n_components_values = [2, 5]
n_neighbors_values = [5, 10]
min_dist_values = [0.0, 0.1]

# Clusters a probar
k_values = [2, 3, 4, 5, 6]

results_umap_kmeans = []

umap_embeddings = {}
models_umap_kmeans = {}
predictions_umap_kmeans = {}

for n_components in n_components_values:
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            
            print("=" * 100)
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
                
                kmeans = SklearnKMeans(
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

k_values_plot = [3, 4, 5]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

best_configs = []

for ax, k in zip(axes, k_values_plot):
    
    best_row = (
        umap_kmeans_results_pd
        .query("k == @k")
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    
    n_components_best = int(best_row["n_components"])
    n_neighbors_best = int(best_row["n_neighbors"])
    min_dist_best = float(best_row["min_dist"])
    silhouette_best = float(best_row["silhouette"])
    
    X_umap_best = umap_embeddings[
        (n_components_best, n_neighbors_best, min_dist_best)
    ]
    
    clusters_best = predictions_umap_kmeans[
        (n_components_best, n_neighbors_best, min_dist_best, k)
    ]
    
    best_configs.append({
        "k": k,
        "n_components": n_components_best,
        "n_neighbors": n_neighbors_best,
        "min_dist": min_dist_best,
        "silhouette": silhouette_best,
        "cluster_sizes": best_row["cluster_sizes"]
    })
    
    pdf_plot = pd.DataFrame({
        ID_COL: player_ids,
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
        f"k={k} | comp={n_components_best}, "
        f"nn={n_neighbors_best}, md={min_dist_best}\n"
        f"Silhouette={silhouette_best:.3f}"
    )
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

best_configs_pd = pd.DataFrame(best_configs)
display(best_configs_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC # HDBSCAN

# COMMAND ----------

# PCA
pca_components_values = [2, 3, 5]

# UMAP
umap_configs = [
    {"n_components": 2, "n_neighbors": 5, "min_dist": 0.0},
    {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 5, "n_neighbors": 15, "min_dist": 0.1},
]

# HDBSCAN
min_cluster_size_values = [5, 10, 15]
min_samples_values = [1, 5]

results_hdbscan = []

embeddings_hdbscan = {}
labels_hdbscan = {}
models_hdbscan = {}


from sklearn.decomposition import PCA as SklearnPCA


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

    pca = SklearnPCA(
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
                f"PCA({n_components}) | min_cluster_size={min_cluster_size} | "
                f"min_samples={min_samples} | clusters={n_clusters} | "
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
                f"min_cluster_size={min_cluster_size} | min_samples={min_samples} | "
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette}"
            )

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

# Queremos soluciones con 3, 4 y 5 clusters
n_clusters_plot = [3, 4, 5]

# Filtrar configuraciones válidas
valid_hdbscan_configs = (
    hdbscan_results_pd
    .query("pct_noise < 0.15")
    .query("n_clusters in @n_clusters_plot")
    .sort_values(["n_clusters", "silhouette_no_noise"], ascending=[True, False])
)

print(valid_hdbscan_configs)
display(valid_hdbscan_configs)

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

best_hdbscan_configs = []

for ax, n_clusters_target in zip(axes, n_clusters_plot):
    
    subset_configs = valid_hdbscan_configs[
        valid_hdbscan_configs["n_clusters"] == n_clusters_target
    ]
    
    if subset_configs.empty:
        ax.set_title(f"No hay configuración válida\npara {n_clusters_target} clusters")
        ax.axis("off")
        continue
    
    # Mejor configuración según silhouette
    best_row = subset_configs.sort_values(
        "silhouette_no_noise",
        ascending=False
    ).iloc[0]
    
    method = best_row["method"]
    n_components = int(best_row["n_components"])
    min_cluster_size = int(best_row["min_cluster_size"])
    min_samples = int(best_row["min_samples"])
    silhouette = float(best_row["silhouette_no_noise"])
    pct_noise = float(best_row["pct_noise"])
    
    if method == "PCA":
        # Recuperar embedding y labels
        X_plot = embeddings_hdbscan[
            ("pca", n_components)
        ]
        
        labels = labels_hdbscan[
            ("pca", n_components, min_cluster_size, min_samples)
        ]
        
        title_extra = f"PCA({n_components})"
        
    elif method == "UMAP":
        n_neighbors = int(best_row["n_neighbors"])
        min_dist = float(best_row["min_dist"])
        
        X_plot = embeddings_hdbscan[
            ("umap", n_components, n_neighbors, min_dist)
        ]
        
        labels = labels_hdbscan[
            ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
        ]
        
        title_extra = f"UMAP({n_components}), nn={n_neighbors}, md={min_dist}"
    
    # Guardar resumen
    best_hdbscan_configs.append({
        "n_clusters": n_clusters_target,
        "method": method,
        "n_components": n_components,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "pct_noise": pct_noise,
        "silhouette_no_noise": silhouette,
        "cluster_sizes": best_row["cluster_sizes"]
    })
    
    # Si el embedding tiene más de 2 dimensiones, visualizamos las dos primeras
    x_axis = X_plot[:, 0]
    y_axis = X_plot[:, 1]
    
    pdf_plot = pd.DataFrame({
        "id_unico_deportista": player_ids,
        "X1": x_axis,
        "X2": y_axis,
        "cluster": labels
    })
    
    for cluster_id in sorted(pdf_plot["cluster"].unique()):
        subset = pdf_plot[pdf_plot["cluster"] == cluster_id]
        
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
        f"HDBSCAN | {n_clusters_target} clusters\n"
        f"{title_extra} | mcs={min_cluster_size}, ms={min_samples}\n"
        f"Silhouette={silhouette:.3f}, ruido={pct_noise:.1%}"
    )
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

best_hdbscan_configs_pd = pd.DataFrame(best_hdbscan_configs)
display(best_hdbscan_configs_pd)

# COMMAND ----------


