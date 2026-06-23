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


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

ID_COL = "id_unico_deportista"

TABLE_COMPETITIVE        = "dev_gold.pitchvision.competitive_profile_agregado_por_jugador"
TABLE_GPS                = "dev_gold.pitchvision.gps_data_agregado_por_jugador"
TABLE_CMJ                = "dev_gold.pitchvision.cmj_agregado_por_jugador"
TABLE_ISOPRONE           = "dev_gold.pitchvision.isoprone_agregado_por_jugador"
TABLE_NORDIC             = "dev_gold.pitchvision.nordic_Agregado_por_jugador"
TABLE_SLJ                = "dev_gold.pitchvision.slj_agregado_por_jugador"
TABLE_ANTHROPOMETRY      = "dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering"

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

# COMMAND ----------

from functools import reduce

dfs_to_join = [
    df_competitive,
    df_gps,
    df_cmj,
    df_isoprone,
    df_nordic,
    df_slj,
    df_anthropometry
]

df_all_player_agg = reduce(
    lambda left, right: left.join(right, on=ID_COL, how="inner"),
    dfs_to_join
)

print("Nº jugadores finales:", df_all_player_agg.count())
print("Nº columnas:", len(df_all_player_agg.columns))

display(df_all_player_agg.limit(5))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

ID_COL = "id_unico_deportista"

# Columnas que NO deben entrar al clustering
cols_excluir_cluster = [
    ID_COL,
    "id_unico_deportista",

    # Fechas / fuentes / control general
    "temporada_fuente",
    "first_test_date",
    "last_test_date",

    # GPS
    "n_sesiones_validas_gps",
    "n_temporadas_gps",

    # CMJ
    "temporadas_fuente_cmj",

    # Isoprone
    "n_tests_isoprone_isoprone",
    "n_seasons_isoprone_isoprone",
    "first_isoprone_test_date_isoprone",
    "last_isoprone_test_date_isoprone",
    "first_isoprone_test_date",
    "last_isoprone_test_date",
    "pct_torque_asym_pct_missing_isoprone",
    "pct_max_force_asym_pct_missing_isoprone",
    "pct_avg_force_asym_pct_missing_isoprone",
    "pct_impulse_asym_pct_missing_isoprone",

    # Nordic
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

    # SLJ
    "n_tests_agregados_slj",
    "n_temporadas_agregadas_slj",
    "temporada_fuente_slj",
    "first_test_date_slj",
    "last_test_date_slj",
    "n_trials_mean_slj",
    "weight_kg_mean_slj",

    # Anthropometry - excluir edad directa / metabólicas originales
    "harris_benedict_BMR_anthropometry",
    "estimated_GET_kcal_anthropometry",
    "basal_metabolism_kcal_anthropometry",
    "age_years_anthropometry"
]

# =========================
# 1. VARIABLES NUMÉRICAS CANDIDATAS
# =========================

numeric_cols = [
    field.name
    for field in df_all_player_agg.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in cols_excluir_cluster
    and not field.name.startswith("pct_")
    and not field.name.endswith("_missing")
    and not field.name.startswith("n_tests")
    and not field.name.startswith("n_temporadas")
]

# =========================
# 2. QUEDARSE SOLO CON ANTHROPOMETRY AGE-SCALED
# =========================

features_cluster = []

for c in numeric_cols:
    es_anthro = c.endswith("_anthropometry")
    es_anthro_age_scaled = "_age_scaled_" in c and c.endswith("_anthropometry")
    
    if es_anthro:
        # Si es anthropometry, solo entra si es age_scaled
        if es_anthro_age_scaled:
            features_cluster.append(c)
    else:
        # Si no es anthropometry, entra normal
        features_cluster.append(c)

print(f"Nº variables iniciales para clustering: {len(features_cluster)}")

# =========================
# 3. REVISAR QUÉ SE HA INCLUIDO Y EXCLUIDO
# =========================

features_anthro_age_scaled = [
    c for c in features_cluster
    if c.endswith("_anthropometry")
]

features_no_anthro = [
    c for c in features_cluster
    if not c.endswith("_anthropometry")
]

anthro_original_excluidas = [
    c for c in numeric_cols
    if c.endswith("_anthropometry")
    and "_age_scaled_" not in c
]

print("\nVariables anthropometry age_scaled incluidas:", len(features_anthro_age_scaled))
for c in features_anthro_age_scaled:
    print(c)

print("\nVariables anthropometry originales excluidas:", len(anthro_original_excluidas))
for c in anthro_original_excluidas:
    print(c)

print("\nVariables del resto de tablas incluidas:", len(features_no_anthro))
for c in features_no_anthro:
    print(c)

# COMMAND ----------

# MAGIC %md
# MAGIC No escalamos las features de la tabal anthropometrica porque ya están escaladas por grupo de edad.

# COMMAND ----------

std_row = df_all_player_agg.select([
    F.stddev(c).alias(c)
    for c in features_cluster
]).collect()[0].asDict()

cols_constantes = [
    c for c, std in std_row.items()
    if std is None or std == 0
]

features_cluster = [
    c for c in features_cluster
    if c not in cols_constantes
]

print(f"Variables constantes eliminadas: {len(cols_constantes)}")
for c in cols_constantes:
    print(c)

print(f"Nº variables finales para clustering: {len(features_cluster)}")

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import VectorAssembler

ID_COL = "id_unico_deportista"

# 1. Filtrar filas completas
df_cluster_input = df_all_player_agg.dropna(subset=features_cluster)

print("Jugadores antes de quitar nulos:", df_all_player_agg.count())
print("Jugadores después de quitar nulos:", df_cluster_input.count())
print("Nº features:", len(features_cluster))

# 2. Pasar a pandas solo con columnas simples
pdf_cluster = (
    df_cluster_input
    .select(ID_COL, *features_cluster)
    .toPandas()
)

# 3. Matriz original
X_raw = pdf_cluster[features_cluster].astype(float).values

# 4. Escalado final conjunto
scaler_sklearn = StandardScaler()
X_scaled = scaler_sklearn.fit_transform(X_raw)

# 5. Crear dataframe pandas con columnas escalares escaladas
scaled_feature_cols = [f"{c}__scaled_final" for c in features_cluster]

pdf_scaled = pd.DataFrame(
    X_scaled,
    columns=scaled_feature_cols
)

pdf_scaled[ID_COL] = pdf_cluster[ID_COL].values

# Reordenar columnas
pdf_scaled = pdf_scaled[[ID_COL] + scaled_feature_cols]

# 6. Convertir a Spark, sin columnas vectoriales todavía
df_scaled_cols = spark.createDataFrame(pdf_scaled)

# 7. Crear vector features_scaled en Spark
assembler_scaled = VectorAssembler(
    inputCols=scaled_feature_cols,
    outputCol="features_scaled"
)

df_scaled_vector = assembler_scaled.transform(df_scaled_cols)

# 8. Unir con variables originales para interpretación posterior
df_original_features = df_cluster_input.select(ID_COL, *features_cluster)

df_cluster_ready = (
    df_scaled_vector
    .select(ID_COL, "features_scaled", *scaled_feature_cols)
    .join(df_original_features, on=ID_COL, how="inner")
    .cache()
)

df_cluster_ready.count()

display(df_cluster_ready.limit(10))

# COMMAND ----------

variance_pd = pd.DataFrame({
    "variable": features_cluster,
    "mean_scaled": X_scaled.mean(axis=0),
    "std_scaled": X_scaled.std(axis=0),
    "variance_scaled": X_scaled.var(axis=0)
})

display(
    variance_pd
    .sort_values("variance_scaled", ascending=False)
    .head(30)
)

# COMMAND ----------

ID_COL = "id_unico_deportista"

df_kmeans_input = (
    df_cluster_ready
    .select(ID_COL, "features_scaled")
    .cache()
)

n_jugadores = df_kmeans_input.count()
print(f"Nº jugadores usados para clustering: {n_jugadores}")

# Rango de k a probar
k_values = range(2, 16)

elbow_results = []
models_elbow = {}

for k in k_values:
    print(f"Entrenando KMeans con k={k}...")
    
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
    
    # WSSSE: suma de distancias cuadradas dentro de los clusters
    wssse = model.summary.trainingCost
    
    elbow_results.append((int(k), float(wssse)))
    models_elbow[k] = model
    
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
# MAGIC ## KMeans + PCA

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ID_COL = "id_unico_deportista"

# Si ya tienes X_scaled del escalado anterior, puedes usarlo directamente.
# Si no, lo reconstruimos desde df_cluster_ready:
pdf_vectors = df_cluster_ready.select(ID_COL, "features_scaled").toPandas()

X_scaled_pca = np.vstack(
    pdf_vectors["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_vectors[ID_COL].values

n_jugadores = X_scaled_pca.shape[0]
n_features = X_scaled_pca.shape[1]

max_components = min(n_features, n_jugadores - 1, 100)

print(f"Nº jugadores usados para PCA: {n_jugadores}")
print(f"Nº variables finales en clustering: {n_features}")
print(f"Nº componentes PCA calculadas: {max_components}")

pca = PCA(
    n_components=max_components,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled_pca)

explained_variance = pca.explained_variance_ratio_

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
plt.title("PCA - Varianza explicada acumulada | Tabla conjunta final")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# COMPONENTES NECESARIAS POR UMBRAL
# =========================

for threshold in [0.80, 0.85, 0.90, 0.95]:
    n_comp = pca_variance_pd.loc[
        pca_variance_pd["cumulative_explained_variance"] >= threshold,
        "component"
    ].min()
    
    print(f"Componentes necesarias para explicar {int(threshold * 100)}%: {n_comp}")

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ID_COL = "id_unico_deportista"
N_COMPONENTS_LOADINGS = 42

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

# Entrenar PCA una sola vez hasta 42 componentes
pca_loadings_model = PCA(
    n_components=max_components_possible,
    random_state=42
)

X_pca_loadings = pca_loadings_model.fit_transform(X_scaled_loadings)

# En sklearn:
# components_ tiene shape = (n_components, n_features)
# Cada fila es una componente principal
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

# Añadir valores absolutos para facilitar interpretación
for pc in pc_cols:
    loadings_all_pd[f"abs_{pc}"] = loadings_all_pd[pc].abs()

display(loadings_all_pd)

# COMMAND ----------

spark.createDataFrame(loadings_all_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.agregadopruebasfisicas_anthropometry_loadingsPCA")

# COMMAND ----------

# =========================
# PCA + KMEANS CON SKLEARN
# TABLA CONJUNTA FINAL
# =========================

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
ID_COL = "id_unico_deportista"

# Si ya tienes X_scaled_pca y player_ids de la celda PCA anterior, se usan directamente.
# Si no, reconstruirlos desde df_cluster_ready:
pdf_vectors = df_cluster_ready.select(ID_COL, "features_scaled").toPandas()

X_scaled_pca = np.vstack(
    pdf_vectors["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_vectors[ID_COL].values

print("Shape X_scaled_pca:", X_scaled_pca.shape)

# Componentes PCA a probar
# Ajusta según lo que viste en la varianza explicada acumulada
n_components_values = [25, 31, 42, 50]
n_components_values = [
    n for n in n_components_values
    if n <= min(X_scaled_pca.shape[0] - 1, X_scaled_pca.shape[1])
]

# Valores de k a probar
k_values = [2, 3, 4, 5, 6, 7, 8]

results_pca_kmeans = []
pca_models = {}
pca_embeddings = {}
pca_kmeans_models = {}
pca_kmeans_labels = {}

for n_components in n_components_values:
    
    print("=" * 80)
    print(f"Calculando PCA({n_components})")
    
    pca_model = PCA(
        n_components=n_components,
        random_state=SEED
    )
    
    X_pca = pca_model.fit_transform(X_scaled_pca)
    explained_variance = float(pca_model.explained_variance_ratio_.sum())
    
    pca_models[n_components] = pca_model
    pca_embeddings[n_components] = X_pca
    
    for k in k_values:
        
        print(f"Entrenando KMeans | PCA={n_components} | k={k}")
        
        km = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=20,
            max_iter=300
        )
        
        labels = km.fit_predict(X_pca)
        
        silhouette = silhouette_score(
            X_pca,
            labels,
            metric="euclidean"
        )
        
        inertia = km.inertia_
        
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))
        
        results_pca_kmeans.append({
            "pca_components": n_components,
            "explained_variance": explained_variance,
            "k": k,
            "silhouette": float(silhouette),
            "inertia": float(inertia),
            "cluster_sizes": str(cluster_sizes)
        })
        
        pca_kmeans_models[(n_components, k)] = km
        pca_kmeans_labels[(n_components, k)] = labels
        
        print(
            f"PCA={n_components} | k={k} | "
            f"var_exp={explained_variance:.3f} | "
            f"silhouette={silhouette:.4f} | "
            f"inertia={inertia:.2f} | "
            f"tamaños={cluster_sizes}"
        )

pca_kmeans_results_pd = (
    pd.DataFrame(results_pca_kmeans)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

display(pca_kmeans_results_pd)

# COMMAND ----------

# =========================
# MEJOR PCA PARA CADA k
# =========================

best_pca_by_k_pd = (
    pca_kmeans_results_pd
    .sort_values("silhouette", ascending=False)
    .groupby("k", as_index=False)
    .head(1)
    .sort_values("k")
    .reset_index(drop=True)
)

display(best_pca_by_k_pd)

# COMMAND ----------

# =========================
# GRÁFICO MEJOR SILHOUETTE POR k
# =========================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    best_pca_by_k_pd["k"],
    best_pca_by_k_pd["silhouette"],
    marker="o"
)

for _, row in best_pca_by_k_pd.iterrows():
    label = f"PCA {int(row['pca_components'])}\n{row['explained_variance']:.1%}"
    plt.text(
        row["k"],
        row["silhouette"],
        label,
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.xlabel("Número de clusters k")
plt.ylabel("Mejor silhouette")
plt.title("PCA + KMeans: mejor silhouette por k | Tabla conjunta final")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# PLOT PCA(25) + KMEANS(k=4)
# VISUALIZADO EN PC1 Y PC2
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ID_COL = "id_unico_deportista"

N_COMPONENTS = 25
K = 4

# Recuperar embedding PCA(25) y labels KMeans(k=4)
X_pca_plot = pca_embeddings[N_COMPONENTS]
labels_plot = pca_kmeans_labels[(N_COMPONENTS, K)]

# Recuperar métricas de esa configuración
config_row = (
    pca_kmeans_results_pd[
        (pca_kmeans_results_pd["pca_components"] == N_COMPONENTS) &
        (pca_kmeans_results_pd["k"] == K)
    ]
    .iloc[0]
)

silhouette = float(config_row["silhouette"])
explained_variance = float(config_row["explained_variance"])
cluster_sizes = config_row["cluster_sizes"]

print(f"PCA({N_COMPONENTS}) + KMeans(k={K})")
print(f"Silhouette: {silhouette:.4f}")
print(f"Varianza explicada: {explained_variance:.2%}")
print(f"Tamaños clusters: {cluster_sizes}")

# DataFrame para plot
plot_pd = pd.DataFrame({
    ID_COL: player_ids,
    "PC1": X_pca_plot[:, 0],
    "PC2": X_pca_plot[:, 1],
    "cluster": labels_plot
})

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
plt.title(
    f"PCA({N_COMPONENTS}) + KMeans(k={K})\n"
    f"Silhouette={silhouette:.3f} | Varianza explicada={explained_variance:.1%}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 0.085 es un silhouette muy bajo. Indica que la separación entre clusters es débil: los grupos existen porque KMeans fuerza k=4, pero los jugadores no están formando bloques muy compactos y claramente separados. Puede deberse a que la tabla conjunta mezcla dominios muy distintos: GPS, CMJ, isoprone, nordic, SLJ, competitive y anthropometry. Puede que no exista una estructura natural clara en 4 grupos globales.

# COMMAND ----------

# =========================
# CREAR df_resultados
# TABLA CONJUNTA FINAL
# Una fila por jugador + variables originales + clusters PCA/KMeans
# =========================

import pandas as pd

ID_COL = "id_unico_deportista"

# =========================
# 1. DataFrame base con variables originales
# =========================

df_resultados = df_cluster_ready.select(
    ID_COL,
    *features_cluster
)

# =========================
# 2. Crear DataFrame pandas con labels de todas las combinaciones PCA/K
# =========================

df_labels_pd = pd.DataFrame({
    ID_COL: player_ids
})

# Combinaciones entrenadas en sklearn
combinaciones = list(pca_kmeans_labels.keys())

for n_components, k in combinaciones:
    labels = pca_kmeans_labels[(n_components, k)]
    col_name = f"cluster_PCA{n_components}_K{k}"
    df_labels_pd[col_name] = labels.astype(int)

# =========================
# 3. Pasar labels a Spark
# =========================

df_labels_spark = spark.createDataFrame(df_labels_pd)

# =========================
# 4. Unir variables originales + clusters
# =========================

df_resultados = (
    df_resultados
    .join(df_labels_spark, on=ID_COL, how="left")
)

display(df_resultados)

# COMMAND ----------

# =========================
# PREPARAR DATOS PARA UMAP
# =========================

import numpy as np
import pandas as pd

ID_COL = "id_unico_deportista"

# Reconstruimos matriz desde df_cluster_ready
pdf_vectors = df_cluster_ready.select(ID_COL, "features_scaled").toPandas()

X_scaled_umap = np.vstack(
    pdf_vectors["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_vectors[ID_COL].values

print("Shape X_scaled_umap:", X_scaled_umap.shape)
print("Nº jugadores:", len(player_ids))

# COMMAND ----------

import umap

SEED = 42

# Grid sencillo para no tardar demasiado
umap_components_values = [2, 5]
n_neighbors_values = [5, 15]
min_dist_values = [0.1]

k_values = [2, 3, 4, 5, 6, 7]

results_umap_kmeans = []
umap_embeddings = {}
umap_kmeans_models = {}
umap_kmeans_labels = {}

for n_components in umap_components_values:
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
            
            X_umap = reducer.fit_transform(X_scaled_umap)
            
            umap_embeddings[(n_components, n_neighbors, min_dist)] = X_umap
            
            for k in k_values:
                
                print(f"Entrenando KMeans sobre UMAP | k={k}")
                
                km = KMeans(
                    n_clusters=k,
                    random_state=SEED,
                    n_init=20,
                    max_iter=300
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
                    "umap_components": n_components,
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
                    f"silhouette={silhouette:.4f} | "
                    f"inertia={inertia:.2f} | "
                    f"tamaños={cluster_sizes}"
                )

umap_kmeans_results_pd = (
    pd.DataFrame(results_umap_kmeans)
    .sort_values("silhouette", ascending=False)
    .reset_index(drop=True)
)

display(umap_kmeans_results_pd)

# COMMAND ----------

best_umap_by_k_pd = (
    umap_kmeans_results_pd
    .sort_values("silhouette", ascending=False)
    .groupby("k", as_index=False)
    .head(1)
    .sort_values("k")
    .reset_index(drop=True)
)

display(best_umap_by_k_pd)

# COMMAND ----------

plt.figure(figsize=(10, 5))

plt.plot(
    best_umap_by_k_pd["k"],
    best_umap_by_k_pd["silhouette"],
    marker="o"
)

for _, row in best_umap_by_k_pd.iterrows():
    label = (
        f"UMAP {int(row['umap_components'])}\n"
        f"nn={int(row['n_neighbors'])}, md={row['min_dist']}"
    )
    plt.text(
        row["k"],
        row["silhouette"],
        label,
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.xlabel("Número de clusters k")
plt.ylabel("Mejor silhouette")
plt.title("UMAP + KMeans: mejor silhouette por k | Tabla conjunta final")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# PLOT UMAP(10) + KMEANS(k=4)
# VISUALIZADO EN UMAP1 Y UMAP2
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ID_COL = "id_unico_deportista"

UMAP_COMPONENTS = 10
N_NEIGHBORS = 5
MIN_DIST = 0.0
K = 4

# Recuperar embedding y labels
X_umap_k4 = umap_embeddings[(UMAP_COMPONENTS, N_NEIGHBORS, MIN_DIST)]
labels_umap_k4 = umap_kmeans_labels[(UMAP_COMPONENTS, N_NEIGHBORS, MIN_DIST, K)]

# Recuperar métricas de la configuración
config_row = (
    umap_kmeans_results_pd[
        (umap_kmeans_results_pd["umap_components"] == UMAP_COMPONENTS) &
        (umap_kmeans_results_pd["n_neighbors"] == N_NEIGHBORS) &
        (umap_kmeans_results_pd["min_dist"] == MIN_DIST) &
        (umap_kmeans_results_pd["k"] == K)
    ]
    .iloc[0]
)

silhouette = float(config_row["silhouette"])
cluster_sizes = config_row["cluster_sizes"]

print(f"UMAP({UMAP_COMPONENTS}) + KMeans(k={K})")
print(f"n_neighbors={N_NEIGHBORS} | min_dist={MIN_DIST}")
print(f"Silhouette: {silhouette:.4f}")
print(f"Tamaños clusters: {cluster_sizes}")

# DataFrame para plot
plot_umap_pd = pd.DataFrame({
    ID_COL: player_ids,
    "UMAP1": X_umap_k4[:, 0],
    "UMAP2": X_umap_k4[:, 1],
    "cluster": labels_umap_k4.astype(int)
})

# Plot
plt.figure(figsize=(8, 6))

for cluster_id in sorted(plot_umap_pd["cluster"].unique()):
    subset = plot_umap_pd[plot_umap_pd["cluster"] == cluster_id]
    plt.scatter(
        subset["UMAP1"],
        subset["UMAP2"],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title(
    f"UMAP({UMAP_COMPONENTS}) + KMeans(k={K})\n"
    f"n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST} | silhouette={silhouette:.3f}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretación de los clústeres

# COMMAND ----------

# =========================
# DATAFRAME PARA INTERPRETACIÓN DE CLUSTERS
# =========================

import pandas as pd
import numpy as np

CLUSTER_COL = f"cluster_UMAP{UMAP_COMPONENTS}_nn{N_NEIGHBORS}_md{str(MIN_DIST).replace('.', '')}_K{K}"

# Pasamos variables originales a pandas
interp_pd = (
    df_cluster_ready
    .select(ID_COL, *features_cluster)
    .toPandas()
)

# Añadir labels del clustering
labels_pd = pd.DataFrame({
    ID_COL: player_ids,
    CLUSTER_COL: labels_umap_k4.astype(int)
})

interp_pd = interp_pd.merge(labels_pd, on=ID_COL, how="inner")

print("Shape dataframe interpretación:", interp_pd.shape)
display(interp_pd.head())

# COMMAND ----------

# =========================
# Z-SCORES Y MEDIAS ESTANDARIZADAS POR CLUSTER
# =========================

# Copia de trabajo
z_pd = interp_pd[[ID_COL, CLUSTER_COL] + features_cluster].copy()

# Convertir features a numéricas
for c in features_cluster:
    z_pd[c] = pd.to_numeric(z_pd[c], errors="coerce")

# Calcular z-score global por variable
z_feature_cols = []

for c in features_cluster:
    mean_val = z_pd[c].mean()
    std_val = z_pd[c].std()
    
    z_col = f"{c}_z"
    
    if pd.notna(std_val) and std_val != 0:
        z_pd[z_col] = (z_pd[c] - mean_val) / std_val
    else:
        z_pd[z_col] = 0.0
    
    z_feature_cols.append(z_col)

# Media z-score por cluster
cluster_z_means_pd = (
    z_pd
    .groupby(CLUSTER_COL)[z_feature_cols]
    .mean()
    .reset_index()
)

# Añadir tamaño de cluster
cluster_sizes_pd = (
    z_pd
    .groupby(CLUSTER_COL)
    .size()
    .reset_index(name="n_jugadores")
)

cluster_z_means_pd = cluster_sizes_pd.merge(
    cluster_z_means_pd,
    on=CLUSTER_COL,
    how="left"
)

display(cluster_z_means_pd)

# COMMAND ----------

# =========================
# VARIABLES MÁS DIFERENCIADORAS ENTRE CLUSTERS
# =========================

importance_rows = []

for z_col in z_feature_cols:
    values = cluster_z_means_pd[z_col]
    var_name = z_col.replace("_z", "")
    
    importance_rows.append({
        "variable": var_name,
        "min_cluster_mean_z": values.min(),
        "max_cluster_mean_z": values.max(),
        "range_between_clusters": values.max() - values.min(),
        "mean_abs_z": values.abs().mean()
    })

importance_umap_pd = (
    pd.DataFrame(importance_rows)
    .sort_values("range_between_clusters", ascending=False)
    .reset_index(drop=True)
)

display(importance_umap_pd.head(40))

# COMMAND ----------

# =========================
# HEATMAP Z-SCORES POR CLUSTER
# UMAP(10) + KMEANS(k=4)
# =========================

import matplotlib.pyplot as plt
import numpy as np

TOP_N = 35

top_vars = importance_umap_pd.head(TOP_N)["variable"].tolist()
top_z_cols = [f"{v}_z" for v in top_vars]

heatmap_data = (
    cluster_z_means_pd
    .set_index(CLUSTER_COL)[top_z_cols]
)

plt.figure(figsize=(18, 7))

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
    f"Perfil medio estandarizado por cluster | "
    f"UMAP({UMAP_COMPONENTS}) + KMeans(k={K})"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================
# AÑADIR A df_resultados LAS LABELS DEL MEJOR UMAP PARA CADA k
# =========================

import pandas as pd

ID_COL = "id_unico_deportista"

# Partimos de df_resultados existente
# Si no existe, lo creamos con las variables originales
if "df_resultados" not in globals():
    df_resultados = df_cluster_ready.select(ID_COL, *features_cluster)

# DataFrame pandas base con ids
df_umap_best_labels_pd = pd.DataFrame({
    ID_COL: player_ids
})

# Recorrer mejores configuraciones por k
for _, row in best_umap_by_k_pd.iterrows():
    
    umap_components = int(row["umap_components"])
    n_neighbors = int(row["n_neighbors"])
    min_dist = float(row["min_dist"])
    k = int(row["k"])
    
    labels = umap_kmeans_labels[
        (umap_components, n_neighbors, min_dist, k)
    ]
    
    # Nombre de columna claro
    min_dist_str = str(min_dist).replace(".", "")
    
    col_name = (
        f"cluster_UMAP{umap_components}"
        f"_nn{n_neighbors}"
        f"_md{min_dist_str}"
        f"_K{k}"
    )
    
    df_umap_best_labels_pd[col_name] = labels.astype(int)

# Pasar labels a Spark
df_umap_best_labels_spark = spark.createDataFrame(df_umap_best_labels_pd)

# Añadir a df_resultados
df_resultados = (
    df_resultados
    .join(df_umap_best_labels_spark, on=ID_COL, how="left")
)

display(df_resultados)

# COMMAND ----------

# MAGIC %md
# MAGIC ## HDBSCAN

# COMMAND ----------

ID_COL = "id_unico_deportista"
SEED = 42

# Reconstruir matriz escalada desde df_cluster_ready
pdf_vectors = df_cluster_ready.select(ID_COL, "features_scaled").toPandas()

X_scaled_hdbscan = np.vstack(
    pdf_vectors["features_scaled"].apply(lambda x: x.toArray())
)

player_ids = pdf_vectors[ID_COL].values

print("Shape matriz HDBSCAN:", X_scaled_hdbscan.shape)
print("Nº jugadores:", len(player_ids))

# COMMAND ----------

import hdbscan
pca_components_values = [25, 31, 42, 50]

umap_configs = [
    {"n_components": 2, "n_neighbors": 5, "min_dist": 0.0},
    {"n_components": 5, "n_neighbors": 5, "min_dist": 0.0},
    {"n_components": 10, "n_neighbors": 5, "min_dist": 0.0},
]

min_cluster_size_values = [5, 8, 10, 15]
min_samples_values = [1, 3, 5]

results_hdbscan = []
embeddings_hdbscan = {}
labels_hdbscan = {}
models_hdbscan = {}


def evaluar_hdbscan(X_reduced, labels):
    """
    Evalúa una solución HDBSCAN.
    El cluster -1 representa ruido.
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


# =========================
# PCA + HDBSCAN
# =========================

for n_components in pca_components_values:
    
    print("=" * 80)
    print(f"Calculando PCA({n_components})")
    
    pca_model = PCA(
        n_components=n_components,
        random_state=SEED
    )
    
    X_pca = pca_model.fit_transform(X_scaled_hdbscan)
    explained_variance = float(pca_model.explained_variance_ratio_.sum())
    
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
            
            labels_hdbscan[("pca", n_components, min_cluster_size, min_samples)] = labels
            models_hdbscan[("pca", n_components, min_cluster_size, min_samples)] = clusterer
            
            print(
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette} | tamaños={cluster_sizes}"
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
    
    X_umap = reducer.fit_transform(X_scaled_hdbscan)
    
    embeddings_hdbscan[("umap", n_components, n_neighbors, min_dist)] = X_umap
    
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
                f"clusters={n_clusters} | ruido={pct_noise:.1%} | "
                f"silhouette={silhouette} | tamaños={cluster_sizes}"
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

max_noise = 0.10

hdbscan_valid_pd = hdbscan_results_pd[
    (hdbscan_results_pd["n_clusters"] >= 2) &
    (hdbscan_results_pd["pct_noise"] <= max_noise) &
    (hdbscan_results_pd["silhouette_no_noise"].notna())
].copy()

hdbscan_valid_pd = (
    hdbscan_valid_pd
    .sort_values(["silhouette_no_noise", "pct_noise"], ascending=[False, True])
    .reset_index(drop=True)
)

display(hdbscan_valid_pd)

# COMMAND ----------

# =========================
# MEJOR HDBSCAN POR Nº DE CLUSTERS
# =========================

best_hdbscan_by_k_pd = (
    hdbscan_valid_pd
    .sort_values("silhouette_no_noise", ascending=False)
    .groupby("n_clusters", as_index=False)
    .head(1)
    .sort_values("n_clusters")
    .reset_index(drop=True)
)

display(best_hdbscan_by_k_pd)

# COMMAND ----------

METHOD = "umap"
UMAP_COMPONENTS = 10
N_NEIGHBORS = 5
MIN_DIST = 0.0
MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 3

# Recuperar embedding y labels
X_hdbscan_plot = embeddings_hdbscan[
    (METHOD, UMAP_COMPONENTS, N_NEIGHBORS, MIN_DIST)
]

labels_hdbscan_k4 = labels_hdbscan[
    (METHOD, UMAP_COMPONENTS, N_NEIGHBORS, MIN_DIST, MIN_CLUSTER_SIZE, MIN_SAMPLES)
]

# Recuperar métricas
config_hdbscan_row = (
    hdbscan_results_pd[
        (hdbscan_results_pd["method"] == "UMAP") &
        (hdbscan_results_pd["n_components"] == UMAP_COMPONENTS) &
        (hdbscan_results_pd["n_neighbors"] == N_NEIGHBORS) &
        (hdbscan_results_pd["min_dist"] == MIN_DIST) &
        (hdbscan_results_pd["min_cluster_size"] == MIN_CLUSTER_SIZE) &
        (hdbscan_results_pd["min_samples"] == MIN_SAMPLES)
    ]
    .iloc[0]
)

n_clusters = int(config_hdbscan_row["n_clusters"])
pct_noise = float(config_hdbscan_row["pct_noise"])
silhouette = float(config_hdbscan_row["silhouette_no_noise"])
cluster_sizes = config_hdbscan_row["cluster_sizes"]

print("Configuración HDBSCAN seleccionada:")
print(config_hdbscan_row)
print(f"\nClusters: {n_clusters}")
print(f"Ruido: {pct_noise:.1%}")
print(f"Silhouette sin ruido: {silhouette:.4f}")
print(f"Tamaños: {cluster_sizes}")

# DataFrame para plot
plot_hdbscan_pd = pd.DataFrame({
    ID_COL: player_ids,
    "UMAP1": X_hdbscan_plot[:, 0],
    "UMAP2": X_hdbscan_plot[:, 1],
    "cluster": labels_hdbscan_k4.astype(int)
})

# Plot
plt.figure(figsize=(8, 6))

for cluster_id in sorted(plot_hdbscan_pd["cluster"].unique()):
    subset = plot_hdbscan_pd[plot_hdbscan_pd["cluster"] == cluster_id]
    
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
    f"HDBSCAN sobre UMAP({UMAP_COMPONENTS}) | {n_clusters} clusters\n"
    f"mcs={MIN_CLUSTER_SIZE}, ms={MIN_SAMPLES} | "
    f"silhouette={silhouette:.3f}, ruido={pct_noise:.1%}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

CLUSTER_HDBSCAN_COL = (
    f"cluster_HDBSCAN_UMAP{UMAP_COMPONENTS}"
    f"_nn{N_NEIGHBORS}"
    f"_md{str(MIN_DIST).replace('.', '')}"
    f"_mcs{MIN_CLUSTER_SIZE}"
    f"_ms{MIN_SAMPLES}"
)

df_hdbscan_labels_pd = pd.DataFrame({
    ID_COL: player_ids,
    CLUSTER_HDBSCAN_COL: labels_hdbscan_k4.astype(int)
})

df_hdbscan_labels_spark = spark.createDataFrame(df_hdbscan_labels_pd)

# Si df_resultados ya existe, añadimos la columna
df_resultados = (
    df_resultados
    .join(df_hdbscan_labels_spark, on=ID_COL, how="left")
)

display(
    df_resultados.select(
        ID_COL,
        CLUSTER_HDBSCAN_COL
    )
)

# COMMAND ----------

display(df_resultados)

# COMMAND ----------

df_resultados.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev_gold.pitchvision.clustersresults_agregadopruebasfisicas_anthropometry")
