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

df_isoprone_player = spark.table("dev_gold.pitchvision.isoprone_agregado_por_jugador")

# COMMAND ----------

ID_COL = "id_unico_deportista"

cols_excluir_cluster = [
    ID_COL,
    "n_tests_isoprone",
    "n_seasons_isoprone",
    "first_isoprone_test_date",
    "last_isoprone_test_date",
    "left_repetitions_mean",
    "left_repetitions_p90",
    "right_repetitions_mean",
    "right_repetitions_p90"

]

features_cluster_isoprone = [
    field.name
    for field in df_isoprone_player.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster
    and not field.name.startswith("pct_")
]

print(f"Número de variables para clustering: {len(features_cluster_isoprone)}")

for c in features_cluster_isoprone:
    print(c)

# COMMAND ----------

null_check = df_isoprone_player.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in features_cluster_isoprone
])

display(null_check)

# COMMAND ----------

# Vectorizar y escalar
assembler = VectorAssembler(
    inputCols=features_cluster_isoprone,
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline_iso = Pipeline(stages=[assembler, scaler])

pipeline_model_iso = pipeline_iso.fit(df_isoprone_player)

df_cluster_ready_iso = (
    pipeline_model_iso
    .transform(df_isoprone_player)
    .select(ID_COL, "features_raw", "features_scaled", *features_cluster_isoprone)
    .cache()
)

df_cluster_ready_iso.count()
display(df_cluster_ready_iso.limit(5))

# COMMAND ----------

k_values = range(2, 8)

elbow_results_iso = []

for k in k_values:
    kmeans = KMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_cluster_ready_iso)
    wssse = model.summary.trainingCost
    
    elbow_results_iso.append((k, float(wssse)))
    
    print(f"k={k} | WSSSE={wssse:.2f}")

elbow_iso_df = spark.createDataFrame(
    elbow_results_iso,
    ["k", "wssse"]
)

display(elbow_iso_df)

# COMMAND ----------

elbow_iso_pd = elbow_iso_df.toPandas().sort_values("k")

plt.figure(figsize=(8, 5))
plt.plot(
    elbow_iso_pd["k"],
    elbow_iso_pd["wssse"],
    marker="o"
)

plt.xlabel("Número de clusters (k)")
plt.ylabel("WSSSE")
plt.title("Método del codo - KMeans Isoprone")
plt.grid(True)
plt.show()

# COMMAND ----------

df_kmeans_baseline_input = (
    df_cluster_ready_iso
    .select("id_unico_deportista", "features_scaled", *features_cluster_isoprone)
    .cache()
)

df_kmeans_baseline_input.count()

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_values = [2, 3, 4, 5, 6, 7, 8]

results_kmeans_baseline = []
models_kmeans_baseline = {}
predictions_kmeans_baseline = {}

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

    model = kmeans.fit(df_kmeans_baseline_input)
    predictions = model.transform(df_kmeans_baseline_input)

    silhouette = evaluator.evaluate(predictions)
    training_cost = model.summary.trainingCost

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

    results_kmeans_baseline.append((
        k,
        float(silhouette),
        float(training_cost),
        str(cluster_sizes_dict)
    ))

    models_kmeans_baseline[k] = model
    predictions_kmeans_baseline[k] = predictions

    print(
        f"k={k} | silhouette={silhouette:.4f} | "
        f"WSSSE={training_cost:.2f} | tamaños={cluster_sizes_dict}"
    )

kmeans_baseline_results_df = spark.createDataFrame(
    results_kmeans_baseline,
    ["k", "silhouette", "wssse", "cluster_sizes"]
).orderBy(F.desc("silhouette"))

display(kmeans_baseline_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PCA + KMeans

# COMMAND ----------

df_pca_input_iso = (
    df_cluster_ready_iso
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input_iso.count()

# Número máximo de componentes a probar
n_features_iso = len(features_cluster_isoprone)
max_components = min(n_features_iso, 20)

pca = PCA(
    k=max_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

pca_model = pca.fit(df_pca_input_iso)

explained_variance = list(pca_model.explainedVariance)

pca_variance_iso_pd = pd.DataFrame({
    "component": list(range(1, len(explained_variance) + 1)),
    "explained_variance": explained_variance
})

pca_variance_iso_pd["cumulative_explained_variance"] = (
    pca_variance_iso_pd["explained_variance"].cumsum()
)

display(pca_variance_iso_pd)

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.plot(
    pca_variance_iso_pd["component"],
    pca_variance_iso_pd["cumulative_explained_variance"],
    marker="o"
)

plt.axhline(0.80, linestyle="--", label="80%")
plt.axhline(0.90, linestyle="--", label="90%")
plt.axhline(0.95, linestyle="--", label="95%")

plt.xlabel("Número de componentes PCA")
plt.ylabel("Varianza explicada acumulada")
plt.title("Isoprone - Varianza explicada acumulada PCA")
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

n_components_values = [3,  5, 7, 10]
n_components_values = [n for n in n_components_values if n <= n_features_iso]

k_values = [2, 3, 4, 5, 6, 7]

results_pca_kmeans_iso = []
models_pca_kmeans_iso = {}
predictions_pca_kmeans_iso = {}

for n_components in n_components_values:
    
    print("=" * 80)
    print(f"PCA con {n_components} componentes")
    
    pca = PCA(
        k=n_components,
        inputCol="features_scaled",
        outputCol=f"features_pca_{n_components}"
    )
    
    pca_model = pca.fit(df_pca_input_iso)
    df_pca_tmp = pca_model.transform(df_pca_input_iso).cache()
    df_pca_tmp.count()
    
    explained_variance_total = float(sum(pca_model.explainedVariance))
    
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
        training_cost = model.summary.trainingCost
        
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
        
        results_pca_kmeans_iso.append((
            int(n_components),
            float(explained_variance_total),
            int(k),
            float(silhouette),
            float(training_cost),
            str(cluster_sizes_dict)
        ))
        
        models_pca_kmeans_iso[(n_components, k)] = model
        predictions_pca_kmeans_iso[(n_components, k)] = predictions
        
        print(
            f"n_comp={n_components} | k={k} | "
            f"var_exp={explained_variance_total:.4f} | "
            f"silhouette={silhouette:.4f} | "
            f"WSSSE={training_cost:.2f} | tamaños={cluster_sizes_dict}"
        )
    
    df_pca_tmp.unpersist()

pca_kmeans_results_iso_df = spark.createDataFrame(
    results_pca_kmeans_iso,
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
    pca_kmeans_results_iso_df.orderBy(F.desc("silhouette"))
)

# COMMAND ----------

pca_kmeans_results_iso_pd = pca_kmeans_results_iso_df.toPandas()

plt.figure(figsize=(9, 5))

for n_comp in sorted(pca_kmeans_results_iso_pd["pca_components"].unique()):
    subset = pca_kmeans_results_iso_pd[
        pca_kmeans_results_iso_pd["pca_components"] == n_comp
    ].sort_values("k")
    
    plt.plot(
        subset["k"],
        subset["silhouette"],
        marker="o",
        label=f"PCA {n_comp}"
    )

plt.xlabel("Número de clusters k")
plt.ylabel("Silhouette")
plt.title("Isoprone - PCA + KMeans")
plt.legend(title="Componentes")
plt.grid(True)
plt.show()

# COMMAND ----------

# Crear DataFrame base con todas las características por jugador
df_resultados = df_cluster_ready_iso.select(ID_COL, *features_cluster_isoprone)

# Agregar labels de clustering por cada prueba realizada
# Asume que predictions_pca_kmeans_iso contiene los DataFrames Spark de predicciones

for n_components, k in predictions_pca_kmeans_iso.keys():
    pred_df = predictions_pca_kmeans_iso[(n_components, k)].select(
        ID_COL,
        F.col("cluster").alias(f"cluster_PCA{n_components}_K{k}")
    )
    df_resultados = df_resultados.join(pred_df, on=ID_COL, how="left")

display(df_resultados)

# COMMAND ----------

# DBTITLE 1,Cell 16
FEATURES_COL = "features_scaled"
df = df_cluster_ready_iso
recommended_components = 10

feature_names = features_cluster_isoprone

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

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.isoprone_loadingsPCA")

# COMMAND ----------

N_COMPONENTS = 3
K_FINAL = 4
pca_3 = PCA(
    k=N_COMPONENTS,
    inputCol="features_scaled",
    outputCol="features_pca_3"
)

pca_3_model = pca_3.fit(df_cluster_ready_iso)
df_pca_3 = pca_3_model.transform(df_cluster_ready_iso)

print("Varianza explicada por componente:")
print(pca_3_model.explainedVariance)

print("Varianza explicada acumulada:")
print(float(sum(pca_3_model.explainedVariance)))

kmeans_pca_3_k4 = KMeans(
    k=K_FINAL,
    seed=42,
    featuresCol="features_pca_3",
    predictionCol="cluster",
    maxIter=50,
    tol=1e-4,
    initMode="k-means||"
)

model_pca_3_k4 = kmeans_pca_3_k4.fit(df_pca_3)
df_pca_3_k4 = model_pca_3_k4.transform(df_pca_3)
evaluator = ClusteringEvaluator(
    featuresCol="features_pca_3",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

silhouette = evaluator.evaluate(df_pca_3_k4)
wssse = model_pca_3_k4.summary.trainingCost

print(f"Silhouette PCA(3) + KMeans k=4: {silhouette:.4f}")
print(f"WSSSE PCA(3) + KMeans k=4: {wssse:.2f}")

display(
    df_pca_3_k4
    .groupBy("cluster")
    .count()
    .orderBy("cluster")
)
pdf_plot = (
    df_pca_3_k4
    .select(ID_COL, "features_pca_3", "cluster")
    .toPandas()
)

pdf_plot["PC1"] = pdf_plot["features_pca_3"].apply(lambda x: float(x[0]))
pdf_plot["PC2"] = pdf_plot["features_pca_3"].apply(lambda x: float(x[1]))
pdf_plot["PC3"] = pdf_plot["features_pca_3"].apply(lambda x: float(x[2]))

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
    f"Isoprone | PCA(3) + KMeans k=4 | "
    f"Silhouette={silhouette:.3f}"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## UMAP + `KMeans`

# COMMAND ----------

pdf_umap_input_iso = (
    df_cluster_ready_iso
    .select(ID_COL, "features_scaled")
    .toPandas()
)

# Convertir vector Spark a matriz numpy
X_scaled_iso = np.vstack(
    pdf_umap_input_iso["features_scaled"].apply(lambda x: x.toArray())
)

player_ids_iso = pdf_umap_input_iso[ID_COL].values

print("Shape matriz UMAP:", X_scaled_iso.shape)

# COMMAND ----------

# Configuraciones UMAP a probar
n_neighbors_values = [5, 10, 15, 25]
min_dist_values = [0.0, 0.1]

# Número de clusters a probar
k_values = [2, 3, 4, 5, 6, 7]

results_umap_kmeans_iso = []

umap_embeddings_iso = {}
models_umap_kmeans_iso = {}
predictions_umap_kmeans_iso = {}

for n_neighbors in n_neighbors_values:
    for min_dist in min_dist_values:
        print(f"UMAP | n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=42
        )
        
        X_umap = umap_model.fit_transform(X_scaled_iso)
        
        umap_embeddings_iso[(n_neighbors, min_dist)] = X_umap
        
        for k in k_values:
            print("-" * 80)
            print(f"Entrenando UMAP + KMeans | k={k}")
            
            kmeans = SklearnKMeans(
                n_clusters=k,
                random_state=42,
                n_init=20
            )
            
            clusters = kmeans.fit_predict(X_umap)
            
            silhouette = silhouette_score(X_umap, clusters)
            inertia = kmeans.inertia_
            
            unique, counts = np.unique(clusters, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            results_umap_kmeans_iso.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "k": k,
                "silhouette": float(silhouette),
                "inertia": float(inertia),
                "cluster_sizes": str(cluster_sizes)
            })
            
            models_umap_kmeans_iso[(n_neighbors, min_dist, k)] = kmeans
            predictions_umap_kmeans_iso[(n_neighbors, min_dist, k)] = clusters
            
            print(
                f"n_neighbors={n_neighbors} | min_dist={min_dist} | k={k} | "
                f"silhouette={silhouette:.4f} | inertia={inertia:.2f} | "
                f"tamaños={cluster_sizes}"
            )

umap_kmeans_results_iso_pd = (
    pd.DataFrame(results_umap_kmeans_iso)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

umap_kmeans_results_iso_pd.head(30)

# COMMAND ----------

display(umap_kmeans_results_iso_pd)

# COMMAND ----------

# Añadir a df_resultados una columna con los labels del mejor resultado UMAP+KMeans para cada K según silhouette

k_values_umap = umap_kmeans_results_iso_pd["k"].unique()
df_resultados_umap = df_resultados

for k in sorted(k_values_umap):
    best_row = (
        umap_kmeans_results_iso_pd
        .query("k == @k")
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    n_neighbors_best = int(best_row.get("n_neighbors", best_row.get("n_neighbors", 5)))
    min_dist_best = float(best_row.get("min_dist", 0.0))
    n_components_best = int(best_row.get("n_components", 2)) if "n_components" in best_row else 2

    # Recuperar los clusters predichos
    key = (
        (n_components_best, n_neighbors_best, min_dist_best, k)
        if (n_components_best, n_neighbors_best, min_dist_best, k) in predictions_umap_kmeans_iso
        else (n_neighbors_best, min_dist_best, k)
    )
    clusters_best = predictions_umap_kmeans_iso[key]

    # DataFrame pandas con id y cluster
    pdf_labels = pd.DataFrame({
        ID_COL: player_ids_iso,
        f"umap_k{k}_best": clusters_best
    })

    # Convertir a Spark DataFrame y unir
    df_labels = spark.createDataFrame(pdf_labels)
    df_resultados_umap = df_resultados_umap.join(df_labels, on=ID_COL, how="left")

display(df_resultados_umap)

# COMMAND ----------

k_values_plot = [3, 4, 5]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

best_configs = []

for ax, k in zip(axes, k_values_plot):
    
    # Mejor configuración para ese k según silhouette
    best_row = (
        umap_kmeans_results_iso_pd
        .query("k == @k")
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    
    n_neighbors_best = int(best_row["n_neighbors"])
    min_dist_best = float(best_row["min_dist"])
    silhouette_best = float(best_row["silhouette"])
    
    # Recuperar embedding y clusters ya calculados
    X_umap_best = umap_embeddings_iso[(n_neighbors_best, min_dist_best)]
    clusters_best = predictions_umap_kmeans_iso[
        (n_neighbors_best, min_dist_best, k)
    ]
    
    # Guardar resumen
    best_configs.append({
        "k": k,
        "n_neighbors": n_neighbors_best,
        "min_dist": min_dist_best,
        "silhouette": silhouette_best,
        "cluster_sizes": best_row["cluster_sizes"]
    })
    
    # DataFrame para plot
    pdf_plot = pd.DataFrame({
        "id_unico_deportista": player_ids_iso,
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
        f"k={k} | nn={n_neighbors_best}, min_dist={min_dist_best}\n"
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
# MAGIC ### Explicabilidad

# COMMAND ----------

df_resultados_umap.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clustersresults_isoprone")

# COMMAND ----------

cols_excluir_interp = [
    ID_COL,
    "cluster_k4",
    "n_tests_isoprone",
    "n_seasons_isoprone",
    "first_isoprone_test_date",
    "last_isoprone_test_date"
]

features_interpretacion_iso = [
    field.name
    for field in df_pca_3_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_interp
    and not field.name.startswith("pct_")
]

df_iso_z = df_pca_3_k4

for col_name in features_interpretacion_iso:
    stats = df_pca_3_k4.agg(
        F.avg(col_name).alias("mean"),
        F.stddev(col_name).alias("std")
    ).collect()[0]
    
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    if std_val is not None and std_val != 0:
        df_iso_z = df_iso_z.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val)
        )
    else:
        df_iso_z = df_iso_z.withColumn(
            f"{col_name}_z",
            F.lit(0.0)
        )

features_z_iso = [f"{c}_z" for c in features_interpretacion_iso]

agg_exprs = [
    F.count("*").alias("n_jugadores")
]

for c in features_z_iso:
    agg_exprs.append(
        F.avg(c).alias(c)
    )

df_iso_cluster_z_means = (
    df_iso_z
    .groupBy("cluster")
    .agg(*agg_exprs)
    .orderBy("cluster")
)

display(df_iso_cluster_z_means)

# COMMAND ----------

cluster_z_iso_pd = df_iso_cluster_z_means.toPandas()

z_cols = [c for c in cluster_z_iso_pd.columns if c.endswith("_z")]

importance_rows = []

for col in z_cols:
    values = cluster_z_iso_pd[col]
    
    importance_rows.append({
        "variable": col.replace("_z", ""),
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_iso_pd = (
    pd.DataFrame(importance_rows)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_iso_pd.head(20))

# COMMAND ----------

top_n = 20

top_vars = importance_iso_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_iso_pd.set_index("cluster")[top_z_cols]

plt.figure(figsize=(14, 6))

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
    "Isoprone - Perfil medio estandarizado por cluster | "
    "PCA(3) KMeans k=4"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### HDBSCAN

# COMMAND ----------

from sklearn.decomposition import PCA as SklearnPCA

# PCA
pca_components_values = [2, 3, 5]

# UMAP
umap_configs = [
    {"n_components": 2, "n_neighbors": 5, "min_dist": 0.0},
    {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 2, "n_neighbors": 30, "min_dist": 0.1},
]

# HDBSCAN
min_cluster_size_values = [5, 10, 15]
min_samples_values = [1, 5]

results_hdbscan_iso = []

embeddings_hdbscan_iso = {}
labels_hdbscan_iso = {}
models_hdbscan_iso = {}

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

    X_pca = pca.fit_transform(X_scaled_iso)

    explained_variance = float(pca.explained_variance_ratio_.sum())

    embeddings_hdbscan_iso[("pca", n_components)] = X_pca

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

            results_hdbscan_iso.append({
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

            labels_hdbscan_iso[
                ("pca", n_components, min_cluster_size, min_samples)
            ] = labels

            models_hdbscan_iso[
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

    X_umap = umap_model.fit_transform(X_scaled_iso)

    embeddings_hdbscan_iso[
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

            results_hdbscan_iso.append({
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

            labels_hdbscan_iso[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = labels

            models_hdbscan_iso[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = clusterer

            print(
                f"UMAP(nn={n_neighbors}, md={min_dist}) | "
                f"min_cluster_size={min_cluster_size} | min_samples={min_samples} | "
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette}"
            )


hdbscan_results_iso_pd = (
    pd.DataFrame(results_hdbscan_iso)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

display(hdbscan_results_iso_pd)

# COMMAND ----------

max_pct_noise = 0.20

best_by_k = (
    hdbscan_results_iso_pd
    .query("pct_noise <= @max_pct_noise")
    .sort_values("silhouette_no_noise", ascending=False)
    .groupby("n_clusters")
    .head(1)
    .reset_index(drop=True)
)

display(best_by_k)
