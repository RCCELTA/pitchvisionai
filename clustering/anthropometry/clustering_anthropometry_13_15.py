# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
pip install umap-learn

# COMMAND ----------

pip install hdbscan

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grupo 13-15

# COMMAND ----------

df = spark.table("dev_gold.pitchvision.anthropometry_13_15")

print(df.count())
display(df.limit(10))

# COMMAND ----------

null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC Tenemos pocos nulos en esta tabla ya que eliminamos las variables que tenian problemas en el paso de análisis previo. Para los pocos nulos que quedan usaremos la mediana para imputar.

# COMMAND ----------

numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

medians = df.select([
    F.expr(f'percentile_approx({col}, 0.5)').alias(col) for col in numeric_cols
]).collect()[0].asDict()

df_imputed = df
for col in numeric_cols:
    median_value = medians[col]
    df_imputed = df_imputed.withColumn(
        col,
        F.when(F.col(col).isNull(), F.lit(median_value)).otherwise(F.col(col))
    )

display(df_imputed)

# COMMAND ----------

# DBTITLE 1,Winsorización conservadora global
# WINSORIZACIÓN CONSERVADORA GLOBAL
# Regla: Q1 - 3*IQR / Q3 + 3*IQR

IQR_FACTOR = 3.0

exclude_cols = [
    "id_unico_deportista",
    "age_years",
    "test_date",
    "season",
    "is_restricted"
]

numeric_cols_to_winsorize = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Columnas numéricas a winsorizar: {len(numeric_cols_to_winsorize)}")

# 1. Calcular Q1, Q3 globales para todas las columnas en una sola pasada
bound_exprs = []
for c in numeric_cols_to_winsorize:
    bound_exprs.append(F.expr(f"percentile_approx(`{c}`, 0.25)").alias(f"{c}__q1"))
    bound_exprs.append(F.expr(f"percentile_approx(`{c}`, 0.75)").alias(f"{c}__q3"))

bounds_row = df_imputed.agg(*bound_exprs).collect()[0]

# 2. Aplicar winsorización con un solo select
select_exprs = []
for col_name in df_imputed.columns:
    if col_name in numeric_cols_to_winsorize:
        q1 = bounds_row[f"{col_name}__q1"]
        q3 = bounds_row[f"{col_name}__q3"]
        iqr = q3 - q1 if (q1 is not None and q3 is not None) else 0
        lower = q1 - IQR_FACTOR * iqr if iqr > 0 else None
        upper = q3 + IQR_FACTOR * iqr if iqr > 0 else None

        if lower is not None and upper is not None:
            winsorized = (
                F.when(F.col(col_name) < F.lit(lower), F.lit(lower))
                .when(F.col(col_name) > F.lit(upper), F.lit(upper))
                .otherwise(F.col(col_name))
            )
            select_exprs.append(winsorized.alias(col_name))
        else:
            select_exprs.append(F.col(col_name))
    else:
        select_exprs.append(F.col(col_name))

df_winsorized = df_imputed.select(*select_exprs)

print("Winsorización conservadora global aplicada.")
print(f"Filas finales: {df_winsorized.count():,}")

display(df_winsorized.limit(20))

# COMMAND ----------

# Seleccionar columnas numéricas para escalar
numeric_cols_for_scaling = [
    field.name
    for field in df_winsorized.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

# VectorAssembler para unir las columnas numéricas
assembler = VectorAssembler(inputCols=numeric_cols_for_scaling, outputCol="features_unscaled")
df_vector = assembler.transform(df_winsorized)

# StandardScaler para escalar los datos
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

display(df_scaled.select(*exclude_cols, "features"))

# COMMAND ----------

window_spec = Window.partitionBy("id_unico_deportista").orderBy(F.col("test_date").desc())

df_latest = (
    df_scaled
    .withColumn("row_num", F.row_number().over(window_spec))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

display(df_latest)

# COMMAND ----------

df_scaled = df_latest

# COMMAND ----------

# DBTITLE 1,Baseline clustering - KMeans
# BASELINE CLUSTERING - KMEANS

FEATURES_COL = "features"
PREDICTION_COL = "prediction"

K_VALUES = [2, 3, 4, 5, 6, 7, 8]
SEED = 42

# Evaluador silhouette
evaluator = ClusteringEvaluator(
    featuresCol=FEATURES_COL,
    predictionCol=PREDICTION_COL,
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

results = []
models_kmeans = {}
predictions_kmeans = {}

for k in K_VALUES:
    print(f"Entrenando KMeans con k={k}")

    kmeans = KMeans(
        k=k,
        seed=SEED,
        featuresCol=FEATURES_COL,
        predictionCol=PREDICTION_COL
    )

    model = kmeans.fit(df_scaled)
    predictions = model.transform(df_scaled)

    silhouette = evaluator.evaluate(predictions)
    training_cost = model.summary.trainingCost

    cluster_sizes = (
        predictions
        .groupBy(PREDICTION_COL)
        .agg(F.count("*").alias("num_jugadores"))
        .orderBy(PREDICTION_COL)
    )

    results.append((
        k,
        float(silhouette),
        float(training_cost)
    ))

    models_kmeans[k] = model
    predictions_kmeans[k] = predictions

    print(f"k={k} | silhouette={silhouette:.4f} | training_cost={training_cost:.2f}")
    display(cluster_sizes)

# Tabla resumen de resultados
kmeans_results_df = spark.createDataFrame(
    results,
    ["k", "silhouette", "training_cost"]
).orderBy("k")

print("\nRESULTADOS KMEANS")
display(kmeans_results_df)

# COMMAND ----------

# Gráfico del método del codo para KMeans (sin reducción de dimensionalidad)
ks = [r[0] for r in results]
costs = [r[2] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(ks, costs, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Costo de entrenamiento (inertia)")
plt.title("Método del codo - KMeans")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

BEST_K = 3

df_kmeans_best = predictions_kmeans[BEST_K]

display(
    df_kmeans_best
    .groupBy("prediction")
    .agg(F.count("*").alias("num_jugadores"))
    .orderBy("prediction")
)

# COMMAND ----------

# DBTITLE 1,Perfil de clusters
# Perfil de clusters con las columnas numéricas originales
cluster_profile = (
    df_kmeans_best
    .groupBy("prediction")
    .agg(
        F.count("*").alias("num_jugadores"),
        *[
            F.round(F.avg(c), 2).alias(f"avg_{c}")
            for c in numeric_cols_for_scaling
        ]
    )
    .orderBy("prediction")
)

display(cluster_profile)

# COMMAND ----------

# DBTITLE 1,Visualización PCA 2D de clusters
# PCA a 2 componentes sobre las mismas features usadas en clustering
pca = PCA(
    k=2,
    inputCol=FEATURES_COL,
    outputCol="pca_features"
)

pca_model = pca.fit(df_kmeans_best)
df_pca = pca_model.transform(df_kmeans_best)

# Varianza explicada
explained_variance = pca_model.explainedVariance.toArray()

print("Varianza explicada por PCA:")
print(f"PC1: {explained_variance[0]:.4f} ({explained_variance[0] * 100:.2f}%)")
print(f"PC2: {explained_variance[1]:.4f} ({explained_variance[1] * 100:.2f}%)")
print(f"Acumulada: {explained_variance.sum():.4f} ({explained_variance.sum() * 100:.2f}%)")

# Convertir vector PCA en columnas PC1 y PC2
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")

df_pca_plot = (
    df_pca
    .withColumn("pca_array", to_array(F.col("pca_features")))
    .withColumn("PC1", F.col("pca_array")[0])
    .withColumn("PC2", F.col("pca_array")[1])
    .drop("pca_array")
)

# Pasar a pandas para pintar
plot_pd = (
    df_pca_plot
    .select("id_unico_deportista", "PC1", "PC2", PREDICTION_COL)
    .toPandas()
)

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    plot_pd["PC1"],
    plot_pd["PC2"],
    c=plot_pd[PREDICTION_COL],
    cmap="viridis",
    alpha=0.75
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Visualización 2D de clusters KMeans con PCA - k={BEST_K}")
plt.grid(True)
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reducción de la dimensionalidad
# MAGIC ### PCA

# COMMAND ----------

# DBTITLE 1,Análisis varianza explicada PCA
# ANÁLISIS DE VARIANZA EXPLICADA POR PCA

PCA_MAX_COMPONENTS = min(40, len(numeric_cols_for_scaling))

pca_tmp = PCA(
    k=PCA_MAX_COMPONENTS,
    inputCol=FEATURES_COL,
    outputCol="features_pca_tmp"
)

pca_tmp_model = pca_tmp.fit(df_scaled)

explained_variance = pca_tmp_model.explainedVariance.toArray()
cum_var = np.cumsum(explained_variance)

for i, cv in enumerate(cum_var, start=1):
    print(f"{i} componentes -> varianza acumulada: {cv*100:.2f}%")

target_variance = 0.95

recommended_components = int(np.argmax(cum_var >= target_variance) + 1)

print(f"\nComponentes recomendados para {target_variance*100:.0f}% de varianza: {recommended_components}")

# COMMAND ----------

feature_names = numeric_cols_for_scaling

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

    pca_model = pca.fit(df_scaled)
    pca_models_explain[n_comp] = pca_model

    pc_matrix = pca_model.pc.toArray()

    for pc_idx in range(n_comp):
        col_name = f"PCA_{n_comp}_PC{pc_idx + 1}"
        loadings_all_pd[col_name] = pc_matrix[:, pc_idx]

display(loadings_all_pd)

# COMMAND ----------

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.anthropometry_13_15_loadingsPCA")

# COMMAND ----------

# DBTITLE 1,Grid Search PCA + KMeans
# GRID SEARCH: PCA + KMEANS

PCA_COMPONENTS_VALUES = [9, 12, 16]
K_VALUES = [2, 3, 4, 5, 6, 7, 8]

results_pca_kmeans = []
models_pca_kmeans = {}
predictions_pca_kmeans = {}

for n_components in PCA_COMPONENTS_VALUES:
    print("=" * 100)
    print(f"Aplicando PCA con {n_components} componentes")

    pca = PCA(
        k=n_components,
        inputCol=FEATURES_COL,
        outputCol=f"features_pca_{n_components}"
    )

    pca_model = pca.fit(df_scaled)
    df_pca = pca_model.transform(df_scaled)

    explained_variance = pca_model.explainedVariance.toArray()
    explained_variance_total = float(explained_variance.sum())

    evaluator_pca = ClusteringEvaluator(
        featuresCol=f"features_pca_{n_components}",
        predictionCol=PREDICTION_COL,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    for k in K_VALUES:
        print(f"  PCA({n_components}) + KMeans k={k}")

        kmeans = KMeans(
            k=k,
            seed=SEED,
            featuresCol=f"features_pca_{n_components}",
            predictionCol=PREDICTION_COL
        )

        model = kmeans.fit(df_pca)
        predictions = model.transform(df_pca)

        silhouette = evaluator_pca.evaluate(predictions)
        training_cost = model.summary.trainingCost

        results_pca_kmeans.append((
            n_components,
            explained_variance_total,
            k,
            float(silhouette),
            float(training_cost)
        ))

        models_pca_kmeans[(n_components, k)] = model
        predictions_pca_kmeans[(n_components, k)] = predictions

        print(f"    silhouette={silhouette:.4f} | cost={training_cost:.2f}")

# Tabla resumen ordenada por silhouette
pca_kmeans_results_df = spark.createDataFrame(
    results_pca_kmeans,
    ["pca_components", "explained_variance", "k", "silhouette", "training_cost"]
).orderBy(F.col("silhouette").desc())

print("\nRESULTADOS PCA + KMEANS ORDENADOS POR SILHOUETTE")
display(pca_kmeans_results_df)

# COMMAND ----------

ID_COL = "id_unico_deportista"

# Obtener todas las combinaciones probadas
combinaciones = list(predictions_pca_kmeans.keys())

# Seleccionar columnas de características originales (sin columnas vectorizadas ni predicciones)
exclude_cols = [
    "features_age_scaled",
    "features_pca_5",
    "features_pca_10",
    "features_pca_15",
    "features_pca_19",
    "features_pca_25",
    "features_pca_tmp",
    "prediction"
]
caracteristicas_cols = [
    c for c in df.columns
    if c not in exclude_cols
]

# DataFrame base de características
df_base = df.select(ID_COL, *caracteristicas_cols)

# Añadir una columna por cada combinación (n_components, k) con el cluster asignado
dfs_with_preds = [df_base]

for n_components, k in combinaciones:
    pred_df = (
        predictions_pca_kmeans[(n_components, k)]
        .select(ID_COL, F.col("prediction").alias(f"cluster_PCA{n_components}_K{k}"))
    )
    dfs_with_preds.append(pred_df)

from functools import reduce

df_resultados = reduce(
    lambda left, right: left.join(right, on=ID_COL, how="left"),
    dfs_with_preds
)

display(df_resultados)

# COMMAND ----------

# DBTITLE 1,Visualización y perfil por k
# Encontrar el mejor resultado (máxima silueta) para cada k
best_per_k = defaultdict(lambda: None)
for row in pca_kmeans_results_df.collect():
    k = row['k']
    silhouette = row['silhouette']
    if best_per_k[k] is None or silhouette > best_per_k[k]['silhouette']:
        best_per_k[k] = row.asDict()

# Ordenar por k
best_results = [best_per_k[k] for k in sorted(best_per_k.keys())]

# Extraer datos para plot resumen
ks = [r['k'] for r in best_results]
silhouettes = [r['silhouette'] for r in best_results]
pca_components = [r['pca_components'] for r in best_results]
explained_variances = [r['explained_variance'] for r in best_results]

# Plot resumen: mejor silhouette por k
plt.figure(figsize=(8, 5))
plt.plot(ks, silhouettes, marker='o')
for k, s, n, v in zip(ks, silhouettes, pca_components, explained_variances):
    plt.text(k, s, f"PCA:{n}\nVar:{v*100:.1f}%", ha='center', va='bottom', fontsize=8)
plt.xlabel("k (número de clusters)")
plt.ylabel("Mejor Silhouette")
plt.title("Mejor Silhouette por k (PCA+KMeans)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Paleta de colores discretos bien diferenciados
CLUSTER_PALETTE = [
    "#E63946", "#1D3557", "#2A9D8F", "#E9C46A",
    "#F77F00", "#7209B7", "#06D6A0", "#EF476F"
]

# Para cada k: plotear clusters + mostrar perfil
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")

for result in best_results:
    k = result['k']
    n_comp = result['pca_components']
    sil = result['silhouette']
    var_exp = result['explained_variance']

    print("=" * 80)
    print(f"k={k} | Mejor config: PCA={n_comp} | Silhouette={sil:.4f} | Varianza explicada={var_exp*100:.1f}%")
    print("=" * 80)

    df_k = predictions_pca_kmeans[(n_comp, k)]

    # Extraer PC1/PC2 para visualización
    df_k_plot = (
        df_k
        .withColumn("pca_array", to_array(F.col(f"features_pca_{n_comp}")))
        .withColumn("PC1", F.col("pca_array")[0])
        .withColumn("PC2", F.col("pca_array")[1])
        .drop("pca_array")
    )

    plot_pd = (
        df_k_plot
        .select("id_unico_deportista", "PC1", "PC2", "prediction")
        .toPandas()
    )

    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(plot_pd["prediction"].unique()):
        mask = plot_pd["prediction"] == cluster_id
        plt.scatter(
            plot_pd.loc[mask, "PC1"],
            plot_pd.loc[mask, "PC2"],
            color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
            label=f"Cluster {cluster_id}",
            alpha=0.75,
            edgecolors="w",
            linewidths=0.3
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Clusters (PCA={n_comp}, k={k}) | Silhouette={sil:.4f}")
    plt.legend(title="Cluster", loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Perfil: número de jugadores y media por variable
    cluster_profile = (
        df_k
        .groupBy("prediction")
        .agg(
            F.count("*").alias("num_jugadores"),
            *[F.round(F.avg(c), 2).alias(f"avg_{c}") for c in numeric_cols_for_scaling]
        )
        .orderBy("prediction")
    )

    display(cluster_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMAP

# COMMAND ----------

# DBTITLE 0,Grid Search UMAP + KMeans
# Extraer features a numpy
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")
X = np.array(
    df_scaled
    .select(to_array(F.col(FEATURES_COL)).alias("arr"))
    .collect()
).squeeze()

print(f"Shape de X: {X.shape}")

# Configuraciones de UMAP a probar
UMAP_CONFIGS = [
    {"n_components": 2,  "n_neighbors": 10, "min_dist": 0.0},
    {"n_components": 2,  "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 5,  "n_neighbors": 10, "min_dist": 0.0},
    {"n_components": 5,  "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 10, "n_neighbors": 15, "min_dist": 0.0},
    {"n_components": 10, "n_neighbors": 15, "min_dist": 0.1},
   
]

K_VALUES = [2, 3, 4, 5, 6, 7, 8]

results_umap_kmeans = []
umap_embeddings = {}
umap_kmeans_models = {}
umap_kmeans_labels = {}

for cfg in UMAP_CONFIGS:
    n_comp = cfg["n_components"]
    n_neigh = cfg["n_neighbors"]
    m_dist = cfg["min_dist"]
    cfg_key = (n_comp, n_neigh, m_dist)

    print("=" * 80)
    print(f"UMAP: n_components={n_comp}, n_neighbors={n_neigh}, min_dist={m_dist}")

    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_neigh,
        min_dist=m_dist,
        metric="euclidean",
        random_state=SEED
    )

    X_umap = reducer.fit_transform(X)
    umap_embeddings[cfg_key] = X_umap

    for k in K_VALUES:
        km = SklearnKMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X_umap)

        sil = silhouette_score(X_umap, labels, metric="euclidean")

        results_umap_kmeans.append({
            "umap_components": n_comp,
            "n_neighbors": n_neigh,
            "min_dist": m_dist,
            "k": k,
            "silhouette": round(sil, 4)
        })

        umap_kmeans_models[(*cfg_key, k)] = km
        umap_kmeans_labels[(*cfg_key, k)] = labels

        print(f"  k={k} | silhouette={sil:.4f}")

# Tabla resumen
results_umap_kmeans_pd = (
    pd.DataFrame(results_umap_kmeans)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

print("\nRESULTADOS UMAP + KMEANS (TOP 15)")
display(results_umap_kmeans_pd.head(15))

# COMMAND ----------

# DBTITLE 1,Visualización y perfil UMAP+KMeans
# Encontrar el mejor resultado (máxima silueta) para cada k en UMAP+KMeans
best_umap_per_k = {}
for idx, row in results_umap_kmeans_pd.iterrows():
    k = row['k']
    sil = row['silhouette']
    if k not in best_umap_per_k or sil > best_umap_per_k[k]['silhouette']:
        best_umap_per_k[k] = row

best_umap_results = [best_umap_per_k[k] for k in sorted(best_umap_per_k.keys())]

# Paleta de colores
CLUSTER_PALETTE = [
    "#E63946", "#1D3557", "#2A9D8F", "#E9C46A",
    "#F77F00", "#7209B7", "#06D6A0", "#EF476F"
]

for result in best_umap_results:
    n_comp = result['umap_components']
    n_neigh = result['n_neighbors']
    m_dist = result['min_dist']
    k = result['k']
    sil = result['silhouette']

    cfg_key = (n_comp, n_neigh, m_dist)
    X_umap = umap_embeddings[cfg_key]
    labels = umap_kmeans_labels[(*cfg_key, k)]

    print("=" * 80)
    print(f"UMAP: n_components={n_comp}, n_neighbors={n_neigh}, min_dist={m_dist} | k={k} | silhouette={sil:.4f}")
    print("=" * 80)

    # Visualización 2D en las dos primeras componentes
    plt.figure(figsize=(9, 6))
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
            label=f"Cluster {cluster_id}",
            alpha=0.75,
            edgecolors="w",
            linewidths=0.3
        )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"Clusters UMAP+KMeans (k={k}) | Silhouette={sil:.4f}")
    plt.legend(title="Cluster", loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tabla de medias por cluster usando variables reales sin normalizar y con nombre de variable
    original_pd = df_scaled.select(numeric_cols_for_scaling).toPandas()
    df_cluster = original_pd.copy()
    df_cluster['cluster'] = labels
    cluster_means = df_cluster.groupby('cluster')[numeric_cols_for_scaling].mean().round(2)
    cluster_sizes = df_cluster['cluster'].value_counts().sort_index()
    cluster_means.insert(0, 'num_jugadores', cluster_sizes)
    display(cluster_means)

# COMMAND ----------

# Añadir etiquetas de cada prueba UMAP+KMeans a df_resultados

dfs_with_umap_preds = [df_resultados]

for cfg in UMAP_CONFIGS:
    n_comp = cfg["n_components"]
    n_neigh = cfg["n_neighbors"]
    m_dist = cfg["min_dist"]
    cfg_key = (n_comp, n_neigh, m_dist)
    for k in K_VALUES:
        labels = umap_kmeans_labels[(*cfg_key, k)]
        col_name = f"cluster_UMAP{n_comp}_NN{n_neigh}_MD{str(m_dist).replace('.','p')}_K{k}"
        # Extraer IDs en el mismo orden que X
        player_ids = (
            df_scaled
            .select("id_unico_deportista")
            .toPandas()["id_unico_deportista"]
            .values
        )
        df_labels = spark.createDataFrame(
            pd.DataFrame({
                "id_unico_deportista": player_ids,
                col_name: labels.astype(int)
            })
        )
        dfs_with_umap_preds.append(df_labels)

from functools import reduce

df_resultados_umap = reduce(
    lambda left, right: left.join(right, on="id_unico_deportista", how="left"),
    dfs_with_umap_preds
)

display(df_resultados_umap)

# COMMAND ----------

df_resultados_umap.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clustersresultsanthropometry_13_15")

# COMMAND ----------

# Guardar el DataFrame de resultados para k=4 y UMAP con 2 componentes
umap_cfg_key = (2, 10, 0.0)
k_umap = 4

labels = umap_kmeans_labels[(*umap_cfg_key, k_umap)]

# Extraer IDs en el mismo orden que X (usado para UMAP/KMeans)
player_ids = (
    df_scaled
    .select("id_unico_deportista")
    .toPandas()["id_unico_deportista"]
    .values
)

# Crear DataFrame con ID + cluster y hacer join (método seguro)
df_labels = spark.createDataFrame(
    pd.DataFrame({
        "id_unico_deportista": player_ids,
        "cluster_umap_k4": labels.astype(int)
    })
)

df_umap_save = df_scaled.join(df_labels, on="id_unico_deportista", how="inner")

# COMMAND ----------

display(df_umap_save)

# COMMAND ----------

# DBTITLE 1,Sección HDBSCAN
# MAGIC %md
# MAGIC ### HDBSCAN

# COMMAND ----------

# DBTITLE 1,Baseline HDBSCAN (sin reducción)
# BASELINE HDBSCAN (sin reducción de dimensionalidad)
# Directamente sobre los datos escalados

from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

HDBSCAN_MIN_CLUSTER_SIZES = [4, 5, 7, 10]
HDBSCAN_MIN_SAMPLES = [1, 2, 3, 5]

results_hdbscan_raw = []
hdbscan_raw_labels = {}

print(f"Shape de X: {X.shape}")
print(f"Probando HDBSCAN directo sobre {X.shape[1]} features escaladas")
print(f"min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZES}")
print(f"min_samples: {HDBSCAN_MIN_SAMPLES}\n")

for min_cs in HDBSCAN_MIN_CLUSTER_SIZES:
    for min_s in HDBSCAN_MIN_SAMPLES:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric="euclidean"
        )

        labels = clusterer.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = n_noise / len(labels) * 100

        valid_mask = labels >= 0
        if n_clusters >= 2 and valid_mask.sum() > n_clusters:
            sil = silhouette_score(X[valid_mask], labels[valid_mask], metric="euclidean")
        else:
            sil = -1.0

        results_hdbscan_raw.append({
            "reduction": "ninguna",
            "min_cluster_size": min_cs,
            "min_samples": min_s,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(noise_pct, 1),
            "silhouette": round(sil, 4)
        })

        hdbscan_raw_labels[(min_cs, min_s)] = labels

        print(f"min_cluster_size={min_cs:>2}, min_samples={min_s:>2} | "
              f"clusters={n_clusters} | ruido={noise_pct:.1f}% | silhouette={sil:.4f}")

# Tabla resumen
results_hdbscan_raw_pd = (
    pd.DataFrame(results_hdbscan_raw)
    .query("n_clusters >= 2")
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

print(f"\nCombinaciones válidas (>=2 clusters): {len(results_hdbscan_raw_pd)}")
if len(results_hdbscan_raw_pd) > 0:
    print("\nRESULTADOS HDBSCAN SIN REDUCCIÓN (TOP 10)")
    display(results_hdbscan_raw_pd.head(10))
else:
    print("Ninguna configuración produjo >=2 clusters en alta dimensionalidad.")
    print("Se recomienda aplicar reducción de dimensionalidad (PCA/UMAP) antes de HDBSCAN.")

# COMMAND ----------

# DBTITLE 1,Grid Search PCA + HDBSCAN
# GRID SEARCH: PCA + HDBSCAN
PCA_COMPONENTS_VALUES_HDBSCAN = [ 9, 12, 16]
HDBSCAN_MIN_CLUSTER_SIZES = [4, 5, 7, 10]
HDBSCAN_MIN_SAMPLES = [1, 2, 3, 5, 7]

results_pca_hdbscan = []
pca_hdbscan_labels = {}
pca_embeddings_hdbscan = {}

for n_components in PCA_COMPONENTS_VALUES_HDBSCAN:
    print("=" * 80)
    print(f"Aplicando PCA con {n_components} componentes")

    pca = PCA(
        k=n_components,
        inputCol=FEATURES_COL,
        outputCol=f"features_pca_hdb_{n_components}"
    )

    pca_model = pca.fit(df_scaled)
    df_pca = pca_model.transform(df_scaled)

    explained_var = float(pca_model.explainedVariance.toArray().sum())

    # Extraer a numpy
    X_pca = np.array(
        df_pca
        .select(to_array(F.col(f"features_pca_hdb_{n_components}")).alias("arr"))
        .collect()
    ).squeeze()

    pca_embeddings_hdbscan[n_components] = X_pca

    for min_cs in HDBSCAN_MIN_CLUSTER_SIZES:
        for min_s in HDBSCAN_MIN_SAMPLES:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cs,
                min_samples=min_s,
                metric="euclidean"
            )

            labels = clusterer.fit_predict(X_pca)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = n_noise / len(labels) * 100

            valid_mask = labels >= 0
            if n_clusters >= 2 and valid_mask.sum() > n_clusters:
                sil = silhouette_score(X_pca[valid_mask], labels[valid_mask], metric="euclidean")
            else:
                sil = -1.0

            results_pca_hdbscan.append({
                "pca_components": n_components,
                "explained_variance": round(explained_var, 4),
                "min_cluster_size": min_cs,
                "min_samples": min_s,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": round(noise_pct, 1),
                "silhouette": round(sil, 4)
            })

            pca_hdbscan_labels[(n_components, min_cs, min_s)] = labels

    print(f"  Varianza explicada: {explained_var*100:.1f}% | Combinaciones probadas: {len(HDBSCAN_MIN_CLUSTER_SIZES) * len(HDBSCAN_MIN_SAMPLES)}")

# Tabla resumen
results_pca_hdbscan_pd = (
    pd.DataFrame(results_pca_hdbscan)
    .query("n_clusters >= 2")
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

print(f"\nCombinaciones válidas (>=2 clusters): {len(results_pca_hdbscan_pd)}")
if len(results_pca_hdbscan_pd) > 0:
    print("\nRESULTADOS PCA + HDBSCAN (TOP 20)")
    display(results_pca_hdbscan_pd.head(20))
else:
    print("Ninguna configuración PCA+HDBSCAN produjo >=2 clusters.")

# COMMAND ----------

# DBTITLE 0,Grid Search UMAP + HDBSCAN
# GRID SEARCH: UMAP + HDBSCAN
# Reutilizamos los embeddings UMAP ya calculados
# Hiperparámetros ajustados para n=42 muestras

HDBSCAN_MIN_CLUSTER_SIZES = [2, 3, 4, 5, 7, 10]
HDBSCAN_MIN_SAMPLES = [1, 2, 3, 5, 7]

results_umap_hdbscan = []
umap_hdbscan_labels = {}

for cfg_key, X_umap in umap_embeddings.items():
    n_comp, n_neigh, m_dist = cfg_key

    for min_cs in HDBSCAN_MIN_CLUSTER_SIZES:
        for min_s in HDBSCAN_MIN_SAMPLES:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cs,
                min_samples=min_s,
                metric="euclidean"
            )

            labels = clusterer.fit_predict(X_umap)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = n_noise / len(labels) * 100

            # Silhouette solo si hay >= 2 clusters y no todo es ruido
            valid_mask = labels >= 0
            if n_clusters >= 2 and valid_mask.sum() > n_clusters:
                sil = silhouette_score(X_umap[valid_mask], labels[valid_mask], metric="euclidean")
            else:
                sil = -1.0

            results_umap_hdbscan.append({
                "umap_components": n_comp,
                "n_neighbors": n_neigh,
                "min_dist": m_dist,
                "min_cluster_size": min_cs,
                "min_samples": min_s,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": round(noise_pct, 1),
                "silhouette": round(sil, 4)
            })

            umap_hdbscan_labels[(*cfg_key, min_cs, min_s)] = labels

    print(f"UMAP({n_comp}, nn={n_neigh}, md={m_dist}) completado")

# Tabla resumen
results_umap_hdbscan_pd = (
    pd.DataFrame(results_umap_hdbscan)
    .query("n_clusters >= 2")
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

print(f"\nCombinaciones válidas (>=2 clusters): {len(results_umap_hdbscan_pd)}")
if len(results_umap_hdbscan_pd) > 0:
    print("\nRESULTADOS UMAP + HDBSCAN (TOP 20)")
    display(results_umap_hdbscan_pd.head(20))
else:
    print("Ninguna configuración UMAP+HDBSCAN produjo >=2 clusters.")

# COMMAND ----------

# DBTITLE 1,Visualización y perfil HDBSCAN
# VISUALIZACIÓN Y PERFIL HDBSCAN - COMPARATIVA DE LOS 3 ENFOQUES
# 1. Sin reducción (raw)
# 2. PCA + HDBSCAN
# 3. UMAP + HDBSCAN

def get_best_per_nclusters(df_results):
    """Devuelve dict {n_clusters: best_row} por silhouette."""
    best = {}
    for idx, row in df_results.iterrows():
        nc = row['n_clusters']
        sil = row['silhouette']
        if nc not in best or sil > best[nc]['silhouette']:
            best[nc] = row
    return best

# Datos originales en pandas para perfiles (mismo número de filas que X/labels)
original_pd = df_scaled.select(numeric_cols_for_scaling).toPandas()

# Color llamativo para ruido
NOISE_COLOR = "#FF00FF"  # magenta

# 1. SIN REDUCCIÓN
print("#" * 80)
print("# HDBSCAN SIN REDUCCIÓN DE DIMENSIONALIDAD")
print("#" * 80)

if len(results_hdbscan_raw_pd) > 0:
    best_raw = get_best_per_nclusters(results_hdbscan_raw_pd)
    best_raw_list = [best_raw[nc] for nc in sorted(best_raw.keys())]

    # Para visualización 2D proyectamos con UMAP
    reducer_2d_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
    X_plot_raw = reducer_2d_raw.fit_transform(X)

    for result in best_raw_list:
        min_cs = int(result['min_cluster_size'])
        min_s = int(result['min_samples'])
        nc = int(result['n_clusters'])
        sil = result['silhouette']
        noise_pct = result['noise_pct']
        if noise_pct > 30:
            continue
        labels = hdbscan_raw_labels[(min_cs, min_s)]

        print("=" * 80)
        print(f"n_clusters={nc} | min_cluster_size={min_cs}, min_samples={min_s} | "
              f"Silhouette={sil:.4f} | Ruido={noise_pct:.1f}%")
        print("=" * 80)

        plt.figure(figsize=(9, 6))
        unique_labels = sorted(set(labels))
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if cluster_id == -1:
                plt.scatter(X_plot_raw[mask, 0], X_plot_raw[mask, 1],
                            color=NOISE_COLOR, label="Ruido", alpha=0.3, s=15, marker="x")
            else:
                plt.scatter(X_plot_raw[mask, 0], X_plot_raw[mask, 1],
                            color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
                            label=f"Cluster {cluster_id}", alpha=0.75, edgecolors="w", linewidths=0.3)
        plt.xlabel("UMAP1 (proyección)")
        plt.ylabel("UMAP2 (proyección)")
        plt.title(f"HDBSCAN Raw (nc={nc}) | Sil={sil:.4f} | Ruido={noise_pct:.1f}%")
        plt.legend(title="Cluster", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Perfil
        df_prof = original_pd.copy()
        df_prof['cluster'] = labels
        df_valid = df_prof[df_prof['cluster'] >= 0]
        profile = df_valid.groupby('cluster')[numeric_cols_for_scaling].mean().round(2)
        profile.insert(0, 'num_jugadores', df_valid.groupby('cluster').size())
        display(profile)
else:
    print("No se encontraron configuraciones válidas con >=2 clusters en raw.")

# 2. PCA + HDBSCAN
print("\n" + "#" * 80)
print("# PCA + HDBSCAN")
print("#" * 80)

if len(results_pca_hdbscan_pd) > 0:
    best_pca_hdb = get_best_per_nclusters(results_pca_hdbscan_pd)
    best_pca_hdb_list = [best_pca_hdb[nc] for nc in sorted(best_pca_hdb.keys())]

    for result in best_pca_hdb_list:
        n_comp = int(result['pca_components'])
        min_cs = int(result['min_cluster_size'])
        min_s = int(result['min_samples'])
        nc = int(result['n_clusters'])
        sil = result['silhouette']
        noise_pct = result['noise_pct']
        var_exp = result['explained_variance']
        if noise_pct > 30:
            continue
        labels = pca_hdbscan_labels[(n_comp, min_cs, min_s)]

        print("=" * 80)
        print(f"n_clusters={nc} | PCA={n_comp} (var={var_exp*100:.1f}%) | "
              f"min_cs={min_cs}, min_s={min_s} | Sil={sil:.4f} | Ruido={noise_pct:.1f}%")
        print("=" * 80)

        X_pca = pca_embeddings_hdbscan[n_comp]
        if n_comp == 2:
            X_plot = X_pca
        else:
            reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
            X_plot = reducer_2d.fit_transform(X_pca)

        plt.figure(figsize=(9, 6))
        unique_labels = sorted(set(labels))
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if cluster_id == -1:
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                            color=NOISE_COLOR, label="Ruido", alpha=0.3, s=15, marker="x")
            else:
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                            color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
                            label=f"Cluster {cluster_id}", alpha=0.75, edgecolors="w", linewidths=0.3)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"PCA({n_comp})+HDBSCAN (nc={nc}) | Sil={sil:.4f} | Ruido={noise_pct:.1f}%")
        plt.legend(title="Cluster", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Perfil
        df_prof = original_pd.copy()
        df_prof['cluster'] = labels
        df_valid = df_prof[df_prof['cluster'] >= 0]
        profile = df_valid.groupby('cluster')[numeric_cols_for_scaling].mean().round(2)
        profile.insert(0, 'num_jugadores', df_valid.groupby('cluster').size())
        display(profile)
else:
    print("No se encontraron configuraciones válidas con >=2 clusters en PCA+HDBSCAN.")

# 3. UMAP + HDBSCAN
print("\n" + "#" * 80)
print("# UMAP + HDBSCAN")
print("#" * 80)

if len(results_umap_hdbscan_pd) > 0:
    best_umap_hdb = get_best_per_nclusters(results_umap_hdbscan_pd)
    best_umap_hdb_list = [best_umap_hdb[nc] for nc in sorted(best_umap_hdb.keys())]

    for result in best_umap_hdb_list:
        n_comp = int(result['umap_components'])
        n_neigh = int(result['n_neighbors'])
        m_dist = result['min_dist']
        min_cs = int(result['min_cluster_size'])
        min_s = int(result['min_samples'])
        nc = int(result['n_clusters'])
        sil = result['silhouette']
        noise_pct = result['noise_pct']
        if noise_pct > 30:
            continue

        cfg_key = (n_comp, n_neigh, m_dist)
        labels = umap_hdbscan_labels[(*cfg_key, min_cs, min_s)]

        print("=" * 80)
        print(f"n_clusters={nc} | UMAP({n_comp}, nn={n_neigh}, md={m_dist}) | "
              f"min_cs={min_cs}, min_s={min_s} | Sil={sil:.4f} | Ruido={noise_pct:.1f}%")
        print("=" * 80)

        X_umap = umap_embeddings[cfg_key]
        if n_comp == 2:
            X_plot = X_umap
        else:
            reducer_2d = umap.UMAP(n_components=2, n_neighbors=n_neigh, min_dist=m_dist, random_state=SEED)
            X_plot = reducer_2d.fit_transform(X_umap)

        plt.figure(figsize=(9, 6))
        unique_labels = sorted(set(labels))
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if cluster_id == -1:
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                            color=NOISE_COLOR, label="Ruido", alpha=0.3, s=15, marker="x")
            else:
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                            color=CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)],
                            label=f"Cluster {cluster_id}", alpha=0.75, edgecolors="w", linewidths=0.3)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title(f"UMAP({n_comp})+HDBSCAN (nc={nc}) | Sil={sil:.4f} | Ruido={noise_pct:.1f}%")
        plt.legend(title="Cluster", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Perfil
        df_prof = original_pd.copy()
        df_prof['cluster'] = labels
        df_valid = df_prof[df_prof['cluster'] >= 0]
        profile = df_valid.groupby('cluster')[numeric_cols_for_scaling].mean().round(2)
        profile.insert(0, 'num_jugadores', df_valid.groupby('cluster').size())
        display(profile)
else:
    print("No se encontraron configuraciones válidas con >=2 clusters en UMAP+HDBSCAN.")
