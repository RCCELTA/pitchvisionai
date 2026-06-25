# Databricks notebook source
# DBTITLE 1,Limpiar caché ML (ejecutar solo si hay overflow)
# Ejecutar esta celda SOLO si aparece ML_CACHE_SIZE_OVERFLOW_EXCEPTION.
# Reinicia el interprete Python y libera toda la cache de modelos ML.
# Despues volver a ejecutar celdas 5, 6, 7, 18 y 20 antes de continuar.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install umap-learn

# COMMAND ----------

# MAGIC %pip install hdbscan

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering general

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: KMeans

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


# COMMAND ----------

df = spark.table("dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering")

print(df.count())
display(df.limit(10))

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

# Extraer valores de k y training_cost del DataFrame de resultados
k_values = [row['k'] for row in kmeans_results_df.collect()]
training_costs = [row['training_cost'] for row in kmeans_results_df.collect()]

plt.figure(figsize=(8, 5))
plt.plot(k_values, training_costs, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Costo de entrenamiento (Suma de distancias cuadradas)")
plt.title("Método del codo para KMeans")
plt.xticks(k_values)
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

display(
    df_kmeans_best
    .groupBy("prediction", "age_group")
    .agg(F.count("*").alias("num_jugadores"))
    .orderBy("prediction", "age_group")
)

# COMMAND ----------

# MAGIC %md
# MAGIC No parece que KMeans esté clusterizando simplemente por edad, porque los tres clusters tienen mezcla de grupos de edad. Esto sugiere que el haber escalado por grupos ha tenido éxito.

# COMMAND ----------

# Columnas originales usadas para crear las variables escaladas
original_feature_cols = [
    c.replace("_age_scaled", "")
    for c in df.columns
    if c.endswith("_age_scaled")
]

cluster_profile = (
    df_kmeans_best
    .groupBy("prediction")
    .agg(
        F.count("*").alias("num_jugadores"),
        *[
            F.round(F.avg(c), 2).alias(f"avg_{c}")
            for c in original_feature_cols
            if c in df_kmeans_best.columns
        ]
    )
    .orderBy("prediction")
)

display(cluster_profile)

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

# MAGIC %md
# MAGIC ## Resultados iniciales KMeans
# MAGIC
# MAGIC Como primera aproximación se ha entrenado un modelo KMeans utilizando las variables fisiológicas escaladas por grupo de edad, de forma que los jugadores se comparan en función de su perfil relativo respecto a jugadores de una categoría similar. Al probar distintos valores de `k`, el mejor valor de silhouette se obtiene con `k = 3` (`silhouette ≈ 0.25`), seguido de `k = 2` (`silhouette ≈ 0.23`). A partir de `k = 4`, la calidad de separación disminuye de forma clara, lo que sugiere que aumentar el número de clusters fragmenta los grupos sin aportar una estructura más clara. Aunque el valor de silhouette no es muy alto, esto es esperable en datos fisiológicos multivariantes. La distribución por grupos de edad muestra que los clusters no están formados únicamente por categorías de edad, lo que indica que el escalado por edad ha ayudado a comparar jugadores de diferentes edades de forma más justa. De forma preliminar, el modelo con `k = 3` parece una buena baseline para continuar el análisis e interpretar los perfiles fisiológicos resultantes.
# MAGIC
# MAGIC ## Interpretación preliminar de los clusters fisiológicos
# MAGIC
# MAGIC A partir del modelo KMeans con `k = 3`, se observan tres perfiles fisiológicos diferenciados. El **cluster 0** es el grupo más numeroso y presenta valores intermedios en la mayoría de variables antropométricas y de composición corporal, con una estatura media aproximada de 175 cm, peso medio de 68 kg, porcentaje graso moderado y valores medios de masa muscular, por lo que puede interpretarse como un **perfil físico equilibrado o intermedio**. El **cluster 1** agrupa a un número menor de jugadores y se caracteriza por menor estatura y peso medio, pero mayores valores de pliegues, suma de pliegues, porcentaje graso y endomorfia, junto con menor masa muscular absoluta; por tanto, representa un **perfil más ligero y compacto, con mayor componente adiposo relativo**. El **cluster 2** muestra los valores más altos de estatura, peso, superficie corporal, masa estimada total, masa muscular y perímetros corporales, con un porcentaje graso no especialmente elevado, lo que sugiere un **perfil grande, pesado y muscular**. Esta interpretación debe considerarse preliminar, ya que la tabla muestra medias absolutas, mientras que el clustering se ha realizado sobre variables escaladas por grupo de edad; aun así, permite dar una primera lectura fisiológica de los grupos obtenidos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reducción de dimensionalidad

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA

# COMMAND ----------

# DBTITLE 1,Cell 18
import gc

# Liberar modelos cacheados para evitar overflow de la caché ML
for var_name in ["models_kmeans", "predictions_kmeans", "models_pca_kmeans", "predictions_pca_kmeans", "pca_models_explain", "pca_model", "pca_tmp_model", "model_best_pca_kmeans", "df_kmeans_best", "df_best_pca_kmeans"]:
    if var_name in dir():
        try:
            exec(f"del {var_name}")
        except:
            pass
gc.collect()

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

target_variance = 0.95

recommended_components = int(np.argmax(cum_var >= target_variance) + 1)

print(f"\nComponentes recomendados para {target_variance*100:.0f}% de varianza: {recommended_components}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explicabilidad PCA

# COMMAND ----------

attrs = df.schema[FEATURES_COL].metadata["ml_attr"]["attrs"]

feature_names = []

for attr_type in attrs:
    for attr in attrs[attr_type]:
        feature_names.append(attr["name"])

print(f"Número de variables: {len(feature_names)}")
print(feature_names[:10])

# COMMAND ----------

# DBTITLE 1,Cell 21

# Entrenar UN solo modelo PCA con todos los componentes necesarios.
# Los autovectores de PCA son iguales independientemente de cuantos se pidan,
# por lo que PC1 de PCA(1) == PC1 de PCA(19). Esto evita acumular modelos en cache.
print(f"Entrenando PCA con {recommended_components} componentes...")

pca_full = PCA(
    k=recommended_components,
    inputCol=FEATURES_COL,
    outputCol="features_pca_full"
)

pca_full_model = pca_full.fit(df)

# Matriz de coeficientes: filas = variables originales, columnas = componentes principales
pc_matrix_full = pca_full_model.pc.toArray()
del pca_full_model  # Liberar cache ML inmediatamente tras extraer la matriz

loadings_all_pd = pd.DataFrame({
    "variable": feature_names
})

for n_comp in range(1, recommended_components + 1):
    # Matriz de coeficientes
    # Filas = variables originales
    # Columnas = componentes principales
    pc_matrix = pc_matrix_full[:, :n_comp]

    for pc_idx in range(n_comp):
        col_name = f"PCA_{n_comp}_PC{pc_idx + 1}"
        loadings_all_pd[col_name] = pc_matrix[:, pc_idx]

display(loadings_all_pd)

# COMMAND ----------

# DBTITLE 1,Cell 22
spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.anthropometry_loadingsPCA")

# COMMAND ----------

# Componentes PCA a probar
PCA_COMPONENTS_VALUES = [19, 25, 34]

PREDICTION_COL = "prediction"
SEED=42

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

# DBTITLE 1,Cell 24
ID_COL = "id_unico_deportista"

# Crear un DataFrame con una columna por cada combinación (n_components, k) de clustering
# Cada columna tendrá el cluster asignado para ese jugador en esa combinación

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

# Empezar con el DataFrame base de características
df_base = df.select(ID_COL, *caracteristicas_cols)

# Añadir una columna por cada combinación (n_components, k) con el cluster asignado
from functools import reduce

dfs_with_preds = [df_base]

for n_components, k in combinaciones:
    pred_df = (
        predictions_pca_kmeans[(n_components, k)]
        .select(ID_COL, F.col("prediction").alias(f"cluster_PCA{n_components}_K{k}"))
    )
    dfs_with_preds.append(pred_df)

# Unir todas las columnas de cluster por ID
from functools import reduce

df_final = reduce(
    lambda left, right: left.join(right, on=ID_COL, how="left"),
    dfs_with_preds
)

display(df_final)

# COMMAND ----------

df_resultados = df_final

# COMMAND ----------

display(df_resultados)

# COMMAND ----------

BEST_PCA_COMPONENTS = 25
BEST_K = 4

df_best_pca_kmeans = predictions_pca_kmeans[(BEST_PCA_COMPONENTS, BEST_K)]
model_best_pca_kmeans = models_pca_kmeans[(BEST_PCA_COMPONENTS, BEST_K)]

# COMMAND ----------

display(
    df_best_pca_kmeans
    .groupBy("prediction")
    .agg(F.count("*").alias("num_jugadores"))
    .orderBy("prediction")
)

# COMMAND ----------

# DBTITLE 1,Cell 20

# Columnas a excluir del perfil
exclude_cols = [
    "id_unico_deportista",
    "prediction",
    "test_date",
    "season",
    "age_group",
    "features_age_scaled",
    "features_pca"
]

# Variables numéricas originales
# Excluimos las columnas escaladas porque luego, si quieres, hacemos otra tabla aparte con ellas
numeric_original_cols = [
    field.name
    for field in df_best_pca_kmeans.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
    and not field.name.endswith("_age_scaled")
]

print(f"Variables numéricas originales usadas en el perfil: {len(numeric_original_cols)}")
print(numeric_original_cols)

cluster_profile_all = (
    df_best_pca_kmeans
    .groupBy("prediction")
    .agg(
        F.count("*").alias("num_jugadores"),
        *[
            F.round(F.avg(c), 2).alias(f"avg_{c}")
            for c in numeric_original_cols
        ]
    )
    .orderBy("prediction")
)

display(cluster_profile_all)

# COMMAND ----------

# DBTITLE 1,Cell 29
# Crear plot_pd con PCA 2D para visualización del mejor modelo
to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")

pca_2d = PCA(k=2, inputCol=FEATURES_COL, outputCol="pca_2d_features")
pca_2d_model = pca_2d.fit(df_best_pca_kmeans)
df_pca_2d = pca_2d_model.transform(df_best_pca_kmeans)

plot_pd = (
    df_pca_2d
    .withColumn("pca_array", to_array(F.col("pca_2d_features")))
    .withColumn("PC1", F.col("pca_array")[0])
    .withColumn("PC2", F.col("pca_array")[1])
    .select(ID_COL, "PC1", "PC2", PREDICTION_COL)
    .toPandas()
)

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=plot_pd,
    x="PC1",
    y="PC2",
    hue=PREDICTION_COL,
    palette="tab10",
    alpha=0.8,
    s=70
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Clusters PCA({BEST_PCA_COMPONENTS}) + KMeans(k={BEST_K})")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Cell 31
# Heatmap de medias z por cluster para las variables más importantes

import seaborn as sns

# Calcular media por cluster
cluster_means = (
    df_best_pca_kmeans
    .groupBy("prediction")
    .agg(*[F.avg(c).alias(c) for c in numeric_original_cols])
    .orderBy("prediction")
    .toPandas()
)

# Estandarizar (z-score) por columna
cluster_z_pd = cluster_means.copy()
for c in numeric_original_cols:
    mean_val = cluster_z_pd[c].mean()
    std_val = cluster_z_pd[c].std()
    cluster_z_pd[f"{c}_z"] = (
        (cluster_z_pd[c] - mean_val) / std_val if std_val > 0 else 0
    )

# Importancia: varianza del z-score entre clusters
importance = []
for c in numeric_original_cols:
    col_z = f"{c}_z"
    var_z = cluster_z_pd[col_z].var()
    importance.append({"variable": c, "importance": var_z})

importance_pd = (
    pd.DataFrame(importance)
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# Top 20 variables más diferenciadoras
top_n = 20
top_vars = importance_pd.head(top_n)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = cluster_z_pd.set_index("prediction")[top_z_cols]
heatmap_data.columns = top_vars

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
plt.title(
    f"Anthropometry - Perfil medio estandarizado por cluster | "
    f"PCA({BEST_PCA_COMPONENTS}) + KMeans(k={BEST_K})"
)
plt.tight_layout()
plt.show()

# COMMAND ----------

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

umap_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=SEED
)

X_umap_2d = umap_2d.fit_transform(X)

umap_pd["UMAP1"] = X_umap_2d[:, 0]
umap_pd["UMAP2"] = X_umap_2d[:, 1]

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=umap_pd,
    x="UMAP1",
    y="UMAP2",
    hue="age_group",
    alpha=0.8,
    s=70
)

plt.title("UMAP 2D coloreado por grupo de edad")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

BEST_PCA_COMPONENTS = 19
BEST_K = 4

df_best_pca_kmeans = predictions_pca_kmeans[(BEST_PCA_COMPONENTS, BEST_K)]

clusters_pd = (
    df_best_pca_kmeans
    .select(ID_COL, "prediction")
    .toPandas()
)

umap_plot_pd = umap_pd.merge(
    clusters_pd,
    on=ID_COL,
    how="left"
)

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=umap_plot_pd,
    x="UMAP1",
    y="UMAP2",
    hue="prediction",
    palette="tab10",
    alpha=0.8,
    s=70
)

plt.title("UMAP 2D coloreado por clusters PCA(5) + KMeans(k=3)")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC La visualización UMAP muestra que los clusters obtenidos con PCA(19) + KMeans(k=4) tienen una separación bastante clara. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMAP +KMeans para probar clustering
# MAGIC

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

# DBTITLE 1,Cell 38
ID_COL = "id_unico_deportista"

# Crear dataframe auxiliar con una fila por jugador
df_umap_labels_pd = pd.DataFrame({
    ID_COL: umap_pd[ID_COL].values
})

# Añadir una columna por cada prueba UMAP + KMeans
for (n_components, k), labels in umap_kmeans_labels.items():
    
    col_name = f"cluster_umap_{n_components}_k{k}"
    
    df_umap_labels_pd[col_name] = labels.astype(int)

display(df_umap_labels_pd.head())

# COMMAND ----------

df_umap_labels_spark = spark.createDataFrame(df_umap_labels_pd)

df_resultados = df_resultados.join(
    df_umap_labels_spark,
    on=ID_COL,
    how="left"
)

display(df_resultados)

# COMMAND ----------

BEST_UMAP_COMPONENTS = 5
BEST_K_UMAP = 4

labels_best = umap_kmeans_labels[(BEST_UMAP_COMPONENTS, BEST_K_UMAP)]

# Para visualizar siempre usamos el UMAP 2D creado antes
umap_pd["umap_kmeans_cluster"] = labels_best

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

BEST_UMAP_COMPONENTS = 2
BEST_K_UMAP = 3

labels_best = umap_kmeans_labels[(BEST_UMAP_COMPONENTS, BEST_K_UMAP)]

# Para visualizar siempre usamos el UMAP 2D creado antes
umap_pd["umap_kmeans_cluster"] = labels_best

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

BEST_UMAP_COMPONENTS = 2
BEST_K_UMAP = 6

labels_best = umap_kmeans_labels[(BEST_UMAP_COMPONENTS, BEST_K_UMAP)]

# Para visualizar siempre usamos el UMAP 2D creado antes
umap_pd["umap_kmeans_cluster"] = labels_best

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

# MAGIC %md
# MAGIC ### HDBSCAN

# COMMAND ----------

# UMAP(5) + HDBSCAN

umap_5 = umap.UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=SEED
)

X_umap_5 = umap_5.fit_transform(X)

hdb = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom"
)

labels_hdbscan = hdb.fit_predict(X_umap_5)

# -1 significa ruido / no asignado
unique_labels, counts = np.unique(labels_hdbscan, return_counts=True)

hdbscan_summary = pd.DataFrame({
    "cluster": unique_labels,
    "num_jugadores": counts
}).sort_values("cluster")

display(hdbscan_summary)

# Silhouette solo si hay al menos 2 clusters reales, excluyendo ruido
mask_clustered = labels_hdbscan != -1
real_clusters = set(labels_hdbscan[mask_clustered])

if len(real_clusters) >= 2:
    silhouette_hdbscan = silhouette_score(
        X_umap_5[mask_clustered],
        labels_hdbscan[mask_clustered],
        metric="euclidean"
    )
    print(f"Silhouette HDBSCAN sin ruido: {silhouette_hdbscan:.4f}")
else:
    print("No hay suficientes clusters reales para calcular silhouette.")

# COMMAND ----------

# Si ya tienes UMAP1/UMAP2 en umap_pd, reutilízalo
umap_pd["hdbscan_cluster"] = labels_hdbscan

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=umap_pd,
    x="UMAP1",
    y="UMAP2",
    hue="hdbscan_cluster",
    palette="tab10",
    alpha=0.85,
    s=70
)

plt.title("HDBSCAN sobre UMAP(5), visualizado en UMAP 2D")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

from sklearn.decomposition import PCA as SklearnPCA

# COMPARATIVA REDUCCIÓN DIMENSIONAL + HDBSCAN: PCA / UMAP + HDBSCAN
# Reducciones PCA a probar
PCA_COMPONENTS_VALUES = [10, 19, 25]

# Reducciones UMAP a probar
UMAP_COMPONENTS_VALUES = [2, 5]
UMAP_N_NEIGHBORS_VALUES = [10, 30]
UMAP_MIN_DIST_VALUES = [0.0, 0.1]

# Configuraciones HDBSCAN
MIN_CLUSTER_SIZE_VALUES = [5, 10, 20]
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
        (results_hdbscan_all_pd["pct_noise"] < 15)
    ]
    .sort_values(
        ["silhouette_sin_ruido", "pct_noise"],
        ascending=[False, True]
    )
)

display(filtered_hdbscan_results)

# COMMAND ----------

# MAGIC %md
# MAGIC Representamos para cada k(numero de clusters) el mejor resultado según silhoutte sobre las dos primeras componentes de la representación reducida utilizada en cada caso.

# COMMAND ----------

K_MIN = 2
K_MAX = 8
MAX_NOISE_PCT = 10

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

# COMMAND ----------

display(df_resultados)

# COMMAND ----------

# DBTITLE 1,Cell 53
df_resultados.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.resultsAnthropometry")

# COMMAND ----------

# MAGIC %md
# MAGIC
