# Databricks notebook source
# DBTITLE 1,Install dependencias
# MAGIC %pip install umap-learn hdbscan

# COMMAND ----------

# DBTITLE 1,Imports
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
from pyspark.sql.types import NumericType
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import hdbscan
import umap as umap
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from collections import defaultdict
from sklearn.cluster import KMeans as SklearnKMeans
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC ## Clustering antropometría - Excluyendo menores (13-18 años)
# MAGIC
# MAGIC Este notebook replica el análisis de clustering antropométrico pero **excluyendo a los jugadores con edad entre 13 y 18 años**, centrándose en jugadores adultos (>18).

# COMMAND ----------

# DBTITLE 1,Carga y filtrado de datos
# Carga de datos desde la vista completa y exclusión de menores (13-18 años)
df_full = spark.table("dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering")

print(f"Filas totales en v_anthropometry: {df_full.count():,}")

# Excluir jugadores entre 13 y 18 años
df = df_full.filter(~((F.col("age_years") >= 13) & (F.col("age_years") <= 18)))

print(f"Filas tras excluir edad 13-18: {df.count():,}")
print(f"Rango de edad restante: {df.select(F.min('age_years'), F.max('age_years')).collect()[0]}")
display(df.limit(10))

# COMMAND ----------

display(df.select("age_group").distinct())

# COMMAND ----------

# DBTITLE 1,Conteo de nulos
null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
display(null_counts)

# COMMAND ----------

age_scaled_feature_cols = [
    c for c in df.columns
    if c.endswith("_age_scaled")
]



# COMMAND ----------

# BASELINE CLUSTERING - KMEANS

ID_COL = "id_unico_deportista"

FEATURES_COL = "features_age_scaled"
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

    model = kmeans.fit(df)
    predictions = model.transform(df)

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

# Gráfico del codo para determinar el k óptimo usando inertia_ (coste de entrenamiento) de KMeans Spark

ks = []
costs = []

for k in K_VALUES:
    model = models_kmeans[k]
    ks.append(k)
    costs.append(model.summary.trainingCost)

plt.figure(figsize=(8, 5))
plt.plot(ks, costs, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (trainingCost)")
plt.title("Método del codo para K óptimo (KMeans Spark)")
plt.xticks(ks)
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

BEST_K = 2

df_kmeans_best = predictions_kmeans[BEST_K]

display(
    df_kmeans_best
    .groupBy("prediction")
    .agg(F.count("*").alias("num_jugadores"))
    .orderBy("prediction")
)

# COMMAND ----------

display(
    df_kmeans_best
    .groupBy("prediction", "age_group")
    .agg(F.count("*").alias("num_jugadores"))
    .orderBy("prediction", "age_group")
)

# COMMAND ----------

# Predicciones del modelo KMeans elegido
df_kmeans_best = predictions_kmeans[BEST_K]

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
    .select(ID_COL, "age_group", "PC1", "PC2", PREDICTION_COL)
    .toPandas()
)

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    plot_pd["PC1"],
    plot_pd["PC2"],
    c=plot_pd[PREDICTION_COL],
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

# DBTITLE 1,Sección PCA
# MAGIC %md
# MAGIC ## Reducción de la dimensionalidad
# MAGIC ### PCA

# COMMAND ----------

FEATURES_COL = "features_age_scaled"

PCA_MAX_COMPONENTS = 40

pca_tmp = PCA(
    k=PCA_MAX_COMPONENTS,
    inputCol=FEATURES_COL,
    outputCol="features_pca_tmp"
)

pca_tmp_model = pca_tmp.fit(df)

explained_variance = pca_tmp_model.explainedVariance.toArray()
cum_var = np.cumsum(explained_variance)

for i, cv in enumerate(cum_var, start=1):
    print(f"{i} componentes -> varianza acumulada: {cv*100:.2f}%")

target_variance = 0.80

recommended_components = int(np.argmax(cum_var >= target_variance) + 1)

print(f"\nComponentes recomendados para {target_variance*100:.0f}% de varianza: {recommended_components}")

# COMMAND ----------

FEATURES_COL = "features_age_scaled"
df = df
recommended_components = 30
feature_names = [
    field["name"]
    for field in df.schema[FEATURES_COL].metadata.get("ml_attr", {}).get("attrs", {}).get("numeric", [])
]

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

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.anthropometry_excluyendomenores_loadingsPCA")

# COMMAND ----------

# Componentes PCA a probar
PCA_COMPONENTS_VALUES = [17, 22, 30]

# Número de clusters a probar
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

    pca_model = pca.fit(df)
    df_pca = pca_model.transform(df)

    explained_variance = pca_model.explainedVariance.toArray()
    explained_variance_total = float(explained_variance.sum())


    evaluator = ClusteringEvaluator(
        featuresCol=f"features_pca_{n_components}",
        predictionCol=PREDICTION_COL,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    for k in K_VALUES:
        print("-" * 80)
        print(f"Entrenando PCA({n_components}) + KMeans k={k}")

        kmeans = KMeans(
            k=k,
            seed=SEED,
            featuresCol=f"features_pca_{n_components}",
            predictionCol=PREDICTION_COL
        )

        model = kmeans.fit(df_pca)
        predictions = model.transform(df_pca)

        silhouette = evaluator.evaluate(predictions)
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



pca_kmeans_results_df = spark.createDataFrame(
    results_pca_kmeans,
    [
        "pca_components",
        "explained_variance",
        "k",
        "silhouette",
        "training_cost"
    ]
).orderBy(
    F.col("silhouette").desc()
)

print("\nRESULTADOS PCA + KMEANS ORDENADOS POR SILHOUETTE")
display(pca_kmeans_results_df)

# COMMAND ----------

# Crear df_resultados con una fila por jugador y columnas de labels para cada combinación PCAx_Kx

df_resultados = df.select(ID_COL, *df.columns)  # Incluye todas las variables originales

for n_components in PCA_COMPONENTS_VALUES:
    for k in K_VALUES:
        col_name = f"cluster_PCA{n_components}_K{k}"
        labels = predictions_pca_kmeans[(n_components, k)].select(ID_COL, PREDICTION_COL)
        df_resultados = (
            df_resultados
            .join(labels.withColumnRenamed(PREDICTION_COL, col_name), on=ID_COL, how="left")
        )

display(df_resultados)

# COMMAND ----------

# DBTITLE 1,Sección UMAP
# MAGIC %md
# MAGIC ### UMAP

# COMMAND ----------

# Convertimos vector Spark a array
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")

df_umap_input = (
    df
    .withColumn("features_array", to_array(F.col(FEATURES_COL)))
    .select(
        ID_COL,
        "age_group",
        "features_array"
    )
)

umap_pd = df_umap_input.toPandas()

X = np.array(umap_pd["features_array"].tolist())

print(X.shape)

# COMMAND ----------

from sklearn.cluster import KMeans as SklearnKMeans

UMAP_COMPONENTS_VALUES = [2, 5, 10]
K_VALUES = [2, 3, 4, 5, 6, 7, 8]

results_umap_kmeans = []

umap_embeddings = {}
umap_kmeans_models = {}
umap_kmeans_labels = {}

for n_components in UMAP_COMPONENTS_VALUES:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED
    )

    X_umap = reducer.fit_transform(X)
    umap_embeddings[n_components] = X_umap

    for k in K_VALUES:
        print(f"Entrenando KMeans sobre UMAP({n_components}) con k={k}")

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

        results_umap_kmeans.append({
            "umap_components": n_components,
            "k": k,
            "silhouette": silhouette
        })

        umap_kmeans_models[(n_components, k)] = km
        umap_kmeans_labels[(n_components, k)] = labels

results_umap_kmeans_pd = (
    pd.DataFrame(results_umap_kmeans)
    .sort_values("silhouette", ascending=False)
)

display(results_umap_kmeans_pd)

# COMMAND ----------

# Añadir los labels de KMeans+UMAP al DataFrame df_resultados

for n_components in UMAP_COMPONENTS_VALUES:
    for k in K_VALUES:
        col_name = f"cluster_UMAP{n_components}_K{k}"
        labels = umap_kmeans_labels[(n_components, k)]
        df_resultados = df_resultados.withColumn(col_name, F.lit(None))

# Drop vector columns before converting to pandas (Arrow can't serialize DenseVector)
from pyspark.ml.linalg import VectorUDT
vector_cols = [f.name for f in df_resultados.schema.fields if isinstance(f.dataType, VectorUDT)]
df_resultados_pd = df_resultados.drop(*vector_cols).toPandas()

for n_components in UMAP_COMPONENTS_VALUES:
    for k in K_VALUES:
        col_name = f"cluster_UMAP{n_components}_K{k}"
        labels = umap_kmeans_labels[(n_components, k)]
        df_resultados_pd[col_name] = labels

df_resultados = spark.createDataFrame(df_resultados_pd)
display(df_resultados)

# COMMAND ----------

BEST_UMAP_COMPONENTS = 2 
BEST_K_UMAP = 4

labels_best = umap_kmeans_labels[(BEST_UMAP_COMPONENTS, BEST_K_UMAP)]

# Para visualizar siempre usamos el UMAP 2D creado antes
umap_pd["umap_kmeans_cluster"] = labels_best

# Añadir coordenadas 2D desde el embedding de 2 componentes
X_umap_2d = umap_embeddings[2]
umap_pd["UMAP1"] = X_umap_2d[:, 0]
umap_pd["UMAP2"] = X_umap_2d[:, 1]

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=umap_pd,
    x="UMAP1",
    y="UMAP2",
    hue="umap_kmeans_cluster",
    palette="tab10",
    alpha=0.8,
    s=70
)

plt.title(f"UMAP visualización 2D - clusters UMAP({BEST_UMAP_COMPONENTS}) + KMeans(k={BEST_K_UMAP})")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

df_resultados.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.clustersresults_anthropometry_excluyendomenores")

# COMMAND ----------

# DBTITLE 1,Sección HDBSCAN
# MAGIC %md
# MAGIC ### HDBSCAN

# COMMAND ----------

from sklearn.decomposition import PCA as SklearnPCA

# COMPARATIVA REDUCCIÓN DIMENSIONAL + HDBSCAN: PCA / UMAP + HDBSCAN
# Reducciones PCA a probar
PCA_COMPONENTS_VALUES = [2, 5, 10, 15, 20, 30]

# Reducciones UMAP a probar
UMAP_COMPONENTS_VALUES = [2, 5, 10]
UMAP_N_NEIGHBORS_VALUES = [10, 15, 30]
UMAP_MIN_DIST_VALUES = [0.0, 0.1, 0.3]

# Configuraciones HDBSCAN
MIN_CLUSTER_SIZE_VALUES = [5, 10, 15, 20, 25]
MIN_SAMPLES_VALUES = [1, 3, 5, 10]
CLUSTER_SELECTION_METHODS = ["eom", "leaf"]

# 2. Crear embeddings PCA y UMAP

embeddings = {}

# PCA
for n_components in PCA_COMPONENTS_VALUES:
    print(f"Calculando PCA({n_components})...")
    
    pca_model = SklearnPCA(
        n_components=n_components,
        random_state=SEED
    )
    
    X_pca = pca_model.fit_transform(X)
    
    embeddings[("PCA", n_components, None, None)] = {
        "X_reduced": X_pca,
        "explained_variance": float(np.sum(pca_model.explained_variance_ratio_))
    }

# UMAP
for n_components in UMAP_COMPONENTS_VALUES:
    for n_neighbors in UMAP_N_NEIGHBORS_VALUES:
        for min_dist in UMAP_MIN_DIST_VALUES:
            
            print(
                f"Calculando UMAP("
                f"components={n_components}, "
                f"n_neighbors={n_neighbors}, "
                f"min_dist={min_dist})..."
            )
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=SEED
            )
            
            X_umap = reducer.fit_transform(X)
            
            embeddings[("UMAP", n_components, n_neighbors, min_dist)] = {
                "X_reduced": X_umap,
                "explained_variance": None
            }

# 3. Probar HDBSCAN sobre cada embedding
# ============================================================

results = []
models_hdbscan_all = {}
labels_hdbscan_all = {}
embeddings_all = {}

for embedding_key, embedding_info in embeddings.items():
    
    reduction_method, n_components, n_neighbors, min_dist = embedding_key
    X_reduced = embedding_info["X_reduced"]
    explained_variance = embedding_info["explained_variance"]
    
    embeddings_all[embedding_key] = X_reduced
    
    print("=" * 100)
    print(
        f"Probando HDBSCAN sobre {reduction_method} | "
        f"components={n_components} | "
        f"n_neighbors={n_neighbors} | "
        f"min_dist={min_dist}"
    )
    
    for min_cluster_size in MIN_CLUSTER_SIZE_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            for cluster_selection_method in CLUSTER_SELECTION_METHODS:
                
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_method=cluster_selection_method
                )
                
                labels = hdb.fit_predict(X_reduced)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int(np.sum(labels == -1))
                pct_noise = round(n_noise / len(labels) * 100, 2)
                
                mask_clustered = labels != -1
                
                if n_clusters >= 2 and mask_clustered.sum() > 1:
                    silhouette = silhouette_score(
                        X_reduced[mask_clustered],
                        labels[mask_clustered],
                        metric="euclidean"
                    )
                else:
                    silhouette = None
                
                # Tamaño mínimo/máximo de clusters reales
                real_labels = labels[labels != -1]
                
                if len(real_labels) > 0:
                    unique_real, counts_real = np.unique(real_labels, return_counts=True)
                    min_cluster_found = int(counts_real.min())
                    max_cluster_found = int(counts_real.max())
                else:
                    min_cluster_found = None
                    max_cluster_found = None
                
                result = {
                    "reduction_method": reduction_method,
                    "n_components": n_components,
                    "explained_variance": explained_variance,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_method": cluster_selection_method,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "pct_noise": pct_noise,
                    "min_cluster_found": min_cluster_found,
                    "max_cluster_found": max_cluster_found,
                    "silhouette_sin_ruido": silhouette
                }
                
                results.append(result)
                
                model_key = (
                    reduction_method,
                    n_components,
                    n_neighbors,
                    min_dist,
                    min_cluster_size,
                    min_samples,
                    cluster_selection_method
                )
                
                models_hdbscan_all[model_key] = hdb
                labels_hdbscan_all[model_key] = labels

# 4. Tabla de resultados
results_hdbscan_all_pd = pd.DataFrame(results)

# Ordenamos priorizando silhouette alto, poco ruido y número razonable de clusters
results_hdbscan_all_pd = results_hdbscan_all_pd.sort_values(
    ["silhouette_sin_ruido", "pct_noise", "n_clusters"],
    ascending=[False, True, True],
    na_position="last"
)

display(results_hdbscan_all_pd)

# COMMAND ----------

filtered_hdbscan_results = (
    results_hdbscan_all_pd[
        (results_hdbscan_all_pd["n_clusters"] >= 2) &
        (results_hdbscan_all_pd["n_clusters"] <= 8) &
        (results_hdbscan_all_pd["silhouette_sin_ruido"] > 0.6) &
        (results_hdbscan_all_pd["pct_noise"] < 20)
    ]
    .sort_values(
        ["silhouette_sin_ruido", "pct_noise"],
        ascending=[False, True]
    )
)

display(filtered_hdbscan_results)

# COMMAND ----------

K_MIN = 2
K_MAX = 8
MAX_NOISE_PCT = 20

# 1. Filtramos resultados válidos
valid_results = (
    results_hdbscan_all_pd[
        (results_hdbscan_all_pd["n_clusters"] >= K_MIN) &
        (results_hdbscan_all_pd["n_clusters"] <= K_MAX) &
        (results_hdbscan_all_pd["silhouette_sin_ruido"].notna()) &
        (results_hdbscan_all_pd["pct_noise"] < MAX_NOISE_PCT)
    ]
    .copy()
)

# 2. Para cada número de clusters, nos quedamos con la mejor silhouette
best_by_k = (
    valid_results
    .sort_values(
        ["n_clusters", "silhouette_sin_ruido", "pct_noise"],
        ascending=[True, False, True]
    )
    .groupby("n_clusters")
    .head(1)
    .sort_values("n_clusters")
)

display(best_by_k)

# 3. Función auxiliar para reconstruir keys
def clean_optional_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x

# 4. Plot para cada k
for _, row in best_by_k.iterrows():

    reduction_method = row["reduction_method"]
    n_components = int(row["n_components"])
    n_neighbors = clean_optional_value(row["n_neighbors"])
    min_dist = clean_optional_value(row["min_dist"])

    min_cluster_size = int(row["min_cluster_size"])
    min_samples = int(row["min_samples"])
    cluster_selection_method = row["cluster_selection_method"]

    n_clusters = int(row["n_clusters"])
    silhouette = row["silhouette_sin_ruido"]
    pct_noise = row["pct_noise"]

    embedding_key = (
        reduction_method,
        n_components,
        n_neighbors,
        min_dist
    )

    label_key = (
        reduction_method,
        n_components,
        n_neighbors,
        min_dist,
        min_cluster_size,
        min_samples,
        cluster_selection_method
    )

    X_reduced = embeddings_all[embedding_key]
    labels = labels_hdbscan_all[label_key]

    plot_pd = umap_pd[[ID_COL, "age_group"]].copy()
    plot_pd["dim1"] = X_reduced[:, 0]
    plot_pd["dim2"] = X_reduced[:, 1]
    plot_pd["cluster"] = labels.astype(str)

    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=plot_pd,
        x="dim1",
        y="dim2",
        hue="cluster",
        palette="tab10",
        alpha=0.85,
        s=70
    )

    plt.title(
        f"Mejor HDBSCAN para {n_clusters} clusters\n"
        f"{reduction_method}({n_components}) | "
        f"neighbors={n_neighbors} | min_dist={min_dist} | "
        f"min_cluster_size={min_cluster_size} | min_samples={min_samples} | "
        f"method={cluster_selection_method}\n"
        f"silhouette={silhouette:.3f} | ruido={pct_noise:.2f}%"
    )

    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
