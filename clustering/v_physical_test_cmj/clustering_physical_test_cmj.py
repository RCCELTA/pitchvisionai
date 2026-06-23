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

df_cmj_cluster = spark.table("dev_gold.pitchvision.cmj_agregado_por_jugador")

# COMMAND ----------

df_cmj_cluster.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminamos el peso para que no sesgue la clusterización

# COMMAND ----------

ID_COL = "id_unico_deportista"

# Columnas que no deben entrar al clustering
cols_excluir_cluster_cmj = [
    ID_COL,
    "temporada_fuente",
    "weight_kg"
]


features_cluster_cmj = [
    field.name
    for field in df_cmj_cluster.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster_cmj
]

print(f"Número de variables para clustering CMJ: {len(features_cluster_cmj)}")
for c in features_cluster_cmj:
    print(c)

# COMMAND ----------

assembler_cmj = VectorAssembler(
    inputCols=features_cluster_cmj,
    outputCol="features_raw"
)

scaler_cmj = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

pipeline_cmj = Pipeline(stages=[assembler_cmj, scaler_cmj])

pipeline_model_cmj = pipeline_cmj.fit(df_cmj_cluster)

df_cluster_ready_cmj = (
    pipeline_model_cmj
    .transform(df_cmj_cluster)
    .select(
        ID_COL,
        "features_raw",
        "features_scaled",
        *features_cluster_cmj
    )
    .cache()
)

df_cluster_ready_cmj.count()

display(df_cluster_ready_cmj.limit(5))

# COMMAND ----------

# Rango de k a probar
k_values = range(2, 8)

elbow_results_cmj = []

# Cachear para acelerar
df_cluster_ready_cmj_fast = (
    df_cluster_ready_cmj
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_cluster_ready_cmj_fast.count()

for k in k_values:
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features_scaled",
        predictionCol="cluster",
        maxIter=20,
        tol=1e-4,
        initMode="k-means||"
    )
    
    model = kmeans.fit(df_cluster_ready_cmj_fast)
    wssse = model.summary.trainingCost
    
    elbow_results_cmj.append((k, float(wssse)))
    
    print(f"k={k} | WSSSE={wssse:.2f}")

elbow_cmj_df = spark.createDataFrame(
    elbow_results_cmj,
    ["k", "wssse"]
)

display(elbow_cmj_df)

# COMMAND ----------

elbow_cmj_pd = elbow_cmj_df.toPandas().sort_values("k")

plt.figure(figsize=(8, 5))
plt.plot(
    elbow_cmj_pd["k"],
    elbow_cmj_pd["wssse"],
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
# MAGIC ## Baseline: KMeans

# COMMAND ----------

evaluator_cmj = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_values = [2, 3, 4, 5, 6, 7, 8]

results_kmeans_cmj = []
models_kmeans_cmj = {}
predictions_kmeans_cmj = {}

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

    model = kmeans.fit(df_cluster_ready_cmj)
    predictions = model.transform(df_cluster_ready_cmj)

    silhouette = evaluator_cmj.evaluate(predictions)
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

    results_kmeans_cmj.append((
        k,
        float(silhouette),
        float(wssse),
        str(cluster_sizes_dict)
    ))

    models_kmeans_cmj[k] = model
    predictions_kmeans_cmj[k] = predictions

    print(
        f"k={k} | silhouette={silhouette:.4f} | "
        f"WSSSE={wssse:.2f} | tamaños={cluster_sizes_dict}"
    )

kmeans_cmj_results_df = spark.createDataFrame(
    results_kmeans_cmj,
    ["k", "silhouette", "wssse", "cluster_sizes"]
)

display(kmeans_cmj_results_df.orderBy(F.desc("silhouette")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## KMeans + PCA

# COMMAND ----------

df_pca_input_cmj = (
    df_cluster_ready_cmj
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

df_pca_input_cmj.count()

# Número máximo de componentes
n_features_cmj = len(features_cluster_cmj)
max_components = min(n_features_cmj, 20)

pca_cmj = PCA(
    k=max_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

pca_cmj_model = pca_cmj.fit(df_pca_input_cmj)

explained_variance = list(pca_cmj_model.explainedVariance)

pca_variance_cmj_pd = pd.DataFrame({
    "component": list(range(1, len(explained_variance) + 1)),
    "explained_variance": explained_variance
})

pca_variance_cmj_pd["cumulative_explained_variance"] = (
    pca_variance_cmj_pd["explained_variance"].cumsum()
)

display(pca_variance_cmj_pd)

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.plot(
    pca_variance_cmj_pd["component"],
    pca_variance_cmj_pd["explained_variance"],
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
    pca_variance_cmj_pd["component"],
    pca_variance_cmj_pd["cumulative_explained_variance"],
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

# DBTITLE 1,Cell 15
FEATURES_COL = "features_scaled"
df = df_cluster_ready_cmj
recommended_components = min(10, len(features_cluster_cmj))

feature_names = features_cluster_cmj

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

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.cmj_loadingsPCA")

# COMMAND ----------

n_components_values = [4, 5, 6, 7, 10]
n_components_values = [n for n in n_components_values if n <= n_features_cmj]

k_values = [2, 3, 4, 5, 6, 7]

results_pca_kmeans_cmj = []
models_pca_kmeans_cmj = {}
predictions_pca_kmeans_cmj = {}
pca_models_cmj = {}

for n_components in n_components_values:
    
    print(f"PCA con {n_components} componentes")
    
    pca = PCA(
        k=n_components,
        inputCol="features_scaled",
        outputCol=f"features_pca_{n_components}"
    )
    
    pca_model = pca.fit(df_pca_input_cmj)
    df_pca_tmp = pca_model.transform(df_pca_input_cmj).cache()
    df_pca_tmp.count()
    
    explained_variance_total = float(sum(pca_model.explainedVariance))
    pca_models_cmj[n_components] = pca_model
    
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
        
        results_pca_kmeans_cmj.append((
            int(n_components),
            float(explained_variance_total),
            int(k),
            float(silhouette),
            float(wssse),
            str(cluster_sizes_dict)
        ))
        
        models_pca_kmeans_cmj[(n_components, k)] = model
        predictions_pca_kmeans_cmj[(n_components, k)] = predictions
        
        print(
            f"n_comp={n_components} | k={k} | "
            f"var_exp={explained_variance_total:.4f} | "
            f"silhouette={silhouette:.4f} | "
            f"WSSSE={wssse:.2f} | tamaños={cluster_sizes_dict}"
        )
    
    df_pca_tmp.unpersist()

pca_kmeans_results_cmj_df = spark.createDataFrame(
    results_pca_kmeans_cmj,
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
    pca_kmeans_results_cmj_df.orderBy(F.desc("silhouette"))
)

# COMMAND ----------

from functools import reduce

# Lista de combinaciones de PCA y K probadas
pca_k_combos = list(models_pca_kmeans_cmj.keys())

# DataFrame base con todas las variables originales
df_resultados = df_cmj_cluster

for n_components, k in pca_k_combos:
    pred_df = (
        predictions_pca_kmeans_cmj[(n_components, k)]
        .select(ID_COL, F.col("cluster").alias(f"cluster_PCA{n_components}_K{k}"))
    )
    df_resultados = df_resultados.join(pred_df, on=ID_COL, how="left")

display(df_resultados)

# COMMAND ----------

pca_kmeans_results_cmj_pd = pca_kmeans_results_cmj_df.toPandas()

plt.figure(figsize=(9, 5))

for n_comp in sorted(pca_kmeans_results_cmj_pd["pca_components"].unique()):
    subset = (
        pca_kmeans_results_cmj_pd[
            pca_kmeans_results_cmj_pd["pca_components"] == n_comp
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

# Obtener predicciones para k=4 y 4 componentes PCA
n_components_plot = 4
k_plot = 4

df_pca_input_cmj_4 = (
    df_cluster_ready_cmj
    .select("id_unico_deportista", "features_scaled")
    .cache()
)

pca_model_4 = PCA(
    k=n_components_plot,
    inputCol="features_scaled",
    outputCol="features_pca_4"
).fit(df_pca_input_cmj_4)

df_pca_4 = pca_model_4.transform(df_pca_input_cmj_4)

kmeans_4_4 = KMeans(
    k=k_plot,
    seed=42,
    featuresCol="features_pca_4",
    predictionCol="cluster",
    maxIter=50,
    tol=1e-4,
    initMode="k-means||"
).fit(df_pca_4)

df_pca_4_pred = kmeans_4_4.transform(df_pca_4)

# Calcular silhouette
evaluator_plot = ClusteringEvaluator(
    featuresCol="features_pca_4",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)
silhouette_plot = evaluator_plot.evaluate(df_pca_4_pred)

# Convertir a Pandas para graficar
from pyspark.ml.functions import vector_to_array

plot_pd = (
    df_pca_4_pred
    .withColumn("pca_array", vector_to_array(F.col("features_pca_4")))
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
plt.title(f"KMeans (k=4) sobre 4 componentes PCA\nSilhouette: {silhouette_plot:.4f}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Cell 21
# Crear DataFrame con features originales + cluster del mejor modelo PCA(4) KMeans k=4
df_pca_4_k4 = (
    df_pca_4_pred
    .select(ID_COL, "cluster")
    .join(df_cluster_ready_cmj, on=ID_COL, how="inner")
)

cols_excluir_interp = [
    ID_COL,
    "cluster"
]

features_interpretacion_iso = [
    field.name
    for field in df_pca_4_k4.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_interp
    and not field.name.startswith("pct_")
]

df_iso_z = df_pca_4_k4

for col_name in features_interpretacion_iso:
    stats = df_pca_4_k4.agg(
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

agg_exprs = [F.count("*").alias("n_jugadores")]
for c in features_z_iso:
    agg_exprs.append(F.avg(c).alias(c))

df_iso_cluster_z_means = (
    df_iso_z
    .groupBy("cluster")
    .agg(*agg_exprs)
    .orderBy("cluster")
)

display(df_iso_cluster_z_means)

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

top_n = 20
top_vars = importance_iso_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_iso_pd.set_index("cluster")[top_z_cols]

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
    labels=[f"{i}" for i in heatmap_data.index]
)
plt.xlabel("Variables")
plt.ylabel("Cluster")
plt.title(
    "CMJ - Perfil medio estandarizado por cluster | "
    "PCA(4) KMeans k=4"
)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMAP

# COMMAND ----------

pdf_umap_input_cmj = (
    df_cluster_ready_cmj
    .select(ID_COL, "features_scaled")
    .toPandas()
)

X_scaled_cmj = np.vstack(
    pdf_umap_input_cmj["features_scaled"].apply(lambda x: x.toArray())
)

player_ids_cmj = pdf_umap_input_cmj[ID_COL].values

print("Shape matriz CMJ:", X_scaled_cmj.shape)

# COMMAND ----------

# Configuraciones moderadas de UMAP
n_components_values = [2, 5]
n_neighbors_values = [5, 10]
min_dist_values = [0.0, 0.1]

# Clusters a probar
k_values = [2, 3, 4, 5, 6]

results_umap_kmeans_cmj = []

umap_embeddings_cmj = {}
models_umap_kmeans_cmj = {}
predictions_umap_kmeans_cmj = {}

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
            
            X_umap = umap_model.fit_transform(X_scaled_cmj)
            
            umap_embeddings_cmj[
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
                
                results_umap_kmeans_cmj.append({
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "k": k,
                    "silhouette": float(silhouette),
                    "inertia": float(inertia),
                    "cluster_sizes": str(cluster_sizes)
                })
                
                models_umap_kmeans_cmj[
                    (n_components, n_neighbors, min_dist, k)
                ] = kmeans
                
                predictions_umap_kmeans_cmj[
                    (n_components, n_neighbors, min_dist, k)
                ] = clusters
                
                print(
                    f"n_comp={n_components} | nn={n_neighbors} | "
                    f"min_dist={min_dist} | k={k} | "
                    f"silhouette={silhouette:.4f} | tamaños={cluster_sizes}"
                )

umap_kmeans_results_cmj_pd = (
    pd.DataFrame(results_umap_kmeans_cmj)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

display(umap_kmeans_results_cmj_pd)

# COMMAND ----------

k_values_plot = [3, 4, 5]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

best_configs_cmj = []

for ax, k in zip(axes, k_values_plot):
    
    best_row = (
        umap_kmeans_results_cmj_pd
        .query("k == @k")
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    
    n_components_best = int(best_row["n_components"])
    n_neighbors_best = int(best_row["n_neighbors"])
    min_dist_best = float(best_row["min_dist"])
    silhouette_best = float(best_row["silhouette"])
    
    X_umap_best = umap_embeddings_cmj[
        (n_components_best, n_neighbors_best, min_dist_best)
    ]
    
    clusters_best = predictions_umap_kmeans_cmj[
        (n_components_best, n_neighbors_best, min_dist_best, k)
    ]
    
    best_configs_cmj.append({
        "k": k,
        "n_components": n_components_best,
        "n_neighbors": n_neighbors_best,
        "min_dist": min_dist_best,
        "silhouette": silhouette_best,
        "cluster_sizes": best_row["cluster_sizes"]
    })
    
    pdf_plot = pd.DataFrame({
        ID_COL: player_ids_cmj,
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

best_configs_cmj_pd = pd.DataFrame(best_configs_cmj)
display(best_configs_cmj_pd)

# COMMAND ----------

# Añadir a df_resultados el mejor cluster UMAP para cada k según silhouette

best_umap_rows = (
    umap_kmeans_results_cmj_pd
    .groupby("k")
    .apply(lambda x: x.loc[x["silhouette"].idxmax()])
    .reset_index(drop=True)
)

for idx, row in best_umap_rows.iterrows():
    n_components = int(row["n_components"])
    n_neighbors = int(row["n_neighbors"])
    min_dist = float(row["min_dist"])
    k = int(row["k"])
    clusters = predictions_umap_kmeans_cmj[(n_components, n_neighbors, min_dist, k)]
    
    pdf_umap = pd.DataFrame({
        ID_COL: player_ids_cmj,
        f"cluster_UMAP{n_components}_K{k}_nn{n_neighbors}_md{min_dist}": clusters
    })
    
    df_resultados = df_resultados.join(
        spark.createDataFrame(pdf_umap),
        on=ID_COL,
        how="left"
    )

display(df_resultados)

# COMMAND ----------

# MAGIC %md
# MAGIC ### HDBSCAN

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

results_hdbscan_cmj = []

embeddings_hdbscan_cmj = {}
labels_hdbscan_cmj = {}
models_hdbscan_cmj = {}


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

    X_pca = pca.fit_transform(X_scaled_cmj)
    explained_variance = float(pca.explained_variance_ratio_.sum())

    embeddings_hdbscan_cmj[("pca", n_components)] = X_pca

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

            results_hdbscan_cmj.append({
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

            labels_hdbscan_cmj[
                ("pca", n_components, min_cluster_size, min_samples)
            ] = labels

            models_hdbscan_cmj[
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

    X_umap = umap_model.fit_transform(X_scaled_cmj)

    embeddings_hdbscan_cmj[
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

            results_hdbscan_cmj.append({
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

            labels_hdbscan_cmj[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = labels

            models_hdbscan_cmj[
                ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
            ] = clusterer

            print(
                f"UMAP(comp={n_components}, nn={n_neighbors}, md={min_dist}) | "
                f"min_cluster_size={min_cluster_size} | min_samples={min_samples} | "
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette}"
            )

hdbscan_results_cmj_pd = (
    pd.DataFrame(results_hdbscan_cmj)
    .sort_values(
        ["silhouette_no_noise", "pct_noise"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

display(hdbscan_results_cmj_pd)

# COMMAND ----------

# Queremos soluciones con 3, 4 y 5 clusters
n_clusters_plot = [3, 4, 5]

# Filtrar configuraciones válidas
valid_hdbscan_configs = (
    hdbscan_results_cmj_pd
    .query("pct_noise < 0.15")
    .query("n_clusters in @n_clusters_plot")
    .sort_values(["n_clusters", "silhouette_no_noise"], ascending=[True, False])
)

display(valid_hdbscan_configs)

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

best_hdbscan_configs_cmj = []

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
        X_plot = embeddings_hdbscan_cmj[
            ("pca", n_components)
        ]
        
        labels = labels_hdbscan_cmj[
            ("pca", n_components, min_cluster_size, min_samples)
        ]
        
        title_extra = f"PCA({n_components})"
        
    elif method == "UMAP":
        n_neighbors = int(best_row["n_neighbors"])
        min_dist = float(best_row["min_dist"])
        
        X_plot = embeddings_hdbscan_cmj[
            ("umap", n_components, n_neighbors, min_dist)
        ]
        
        labels = labels_hdbscan_cmj[
            ("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)
        ]
        
        title_extra = f"UMAP({n_components}), nn={n_neighbors}, md={min_dist}"
    
    # Guardar resumen
    best_hdbscan_configs_cmj.append({
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
        "id_unico_deportista": player_ids_cmj,
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

best_hdbscan_configs_cmj_pd = pd.DataFrame(best_hdbscan_configs_cmj)
display(best_hdbscan_configs_cmj_pd)

# COMMAND ----------

# Añadir a df_resultados las labels de las 2 configuraciones ploteadas arriba (Cell 29)
for config in best_hdbscan_configs_cmj[:2]:
    method = config["method"]
    n_components = int(config["n_components"])
    min_cluster_size = int(config["min_cluster_size"])
    min_samples = int(config["min_samples"])
    if method == "PCA":
        labels = labels_hdbscan_cmj[("pca", n_components, min_cluster_size, min_samples)]
        col_name = f"cluster_HDBSCAN_PCA{n_components}_mcs{min_cluster_size}_ms{min_samples}"
    elif method == "UMAP":
        n_neighbors = int(valid_hdbscan_configs.query(
            (f"method == 'UMAP' and n_components == {n_components} and "
             f"min_cluster_size == {min_cluster_size} and min_samples == {min_samples}")
        )["n_neighbors"].iloc[0])
        min_dist = float(valid_hdbscan_configs.query(
            (f"method == 'UMAP' and n_components == {n_components} and "
             f"min_cluster_size == {min_cluster_size} and min_samples == {min_samples}")
        )["min_dist"].iloc[0])
        labels = labels_hdbscan_cmj[("umap", n_components, n_neighbors, min_dist, min_cluster_size, min_samples)]
        col_name = f"cluster_HDBSCAN_UMAP{n_components}_nn{n_neighbors}_md{min_dist}_mcs{min_cluster_size}_ms{min_samples}"
    pdf_labels = pd.DataFrame({
        ID_COL: player_ids_cmj,
        col_name: labels
    })
    df_resultados = df_resultados.join(
        spark.createDataFrame(pdf_labels),
        on=ID_COL,
        how="left"
    )

display(df_resultados)

# COMMAND ----------

df_resultados.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.clustersresults_cmj")
