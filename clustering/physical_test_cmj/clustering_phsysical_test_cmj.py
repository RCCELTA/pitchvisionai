# Databricks notebook source
# DBTITLE 1,Imports y configuración
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Tablas
TABLE_NAME  = "dev_silver.pitchvision.v_physical_test_cmj"
TABLE_AGES  = "dev_silver.pitchvision.v_anthropometry"

# Temporada
LAST_SEASON = "25-26" 

# Columns
ID_COL      = "id_unico_deportista"
DATE_COL    = "test_date"

# Constantes
CORR_THRESHOLD = 0.95
IQR_FACTOR     = 3.0
SEED           = 42

# COMMAND ----------

# MAGIC %md
# MAGIC Approach con solo datos de última temporada

# COMMAND ----------

# DBTITLE 1,Carga y filtro temporada
df_raw = spark.table(TABLE_NAME)

df = df_raw.filter(F.col("season") == LAST_SEASON)

print(f"Filas temporada {LAST_SEASON}: {df.count():,}")
print(f"Jugadores únicos:              {df.select(ID_COL).distinct().count():,}")

# COMMAND ----------

# DBTITLE 1,Columnas numericas para sacar media
EXCLUDE_ALWAYS = [ID_COL, DATE_COL, "season"]

numeric_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]

print(f"Columnas numéricas para análisis: {len(numeric_cols)}")
print(numeric_cols)

# COMMAND ----------

# DBTITLE 1,Medias por jugador
agg_exprs = [F.avg(c).alias(c) for c in numeric_cols]

df_player = (
    df
    .groupBy(ID_COL)
    .agg(*agg_exprs)
)

print(f"Jugadores tras agregación: {df_player.count():,}")
display(df_player.limit(5))

# COMMAND ----------

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in numeric_cols
]

nulls_pd = df_player.agg(*null_exprs).toPandas().T
nulls_pd.columns = ["nulos"]
nulls_pd["pct"] = (nulls_pd["nulos"] / df_player.count() * 100).round(2)
nulls_pd = nulls_pd.sort_values("pct", ascending=False)

display(nulls_pd)

# Columnas con > 50 % de nulos → las descartamos directamente
cols_high_null = nulls_pd[nulls_pd["pct"] > 50].index.tolist()
print(f"\nColumnas descartadas por >50% nulos: {cols_high_null}")

numeric_cols = [c for c in numeric_cols if c not in cols_high_null]
df_player = df_player.drop(*cols_high_null)

# COMMAND ----------

# DBTITLE 1,Winsorizacion

bound_exprs = []
for c in numeric_cols:
    bound_exprs += [
        F.expr(f"percentile_approx(`{c}`, 0.25)").alias(f"{c}__q1"),
        F.expr(f"percentile_approx(`{c}`, 0.75)").alias(f"{c}__q3"),
    ]

bounds = df_player.agg(*bound_exprs)
df_w = df_player.crossJoin(bounds)

select_exprs = []
for col_name in df_player.columns:
    if col_name in numeric_cols:
        iqr  = F.col(f"{col_name}__q3") - F.col(f"{col_name}__q1")
        low  = F.col(f"{col_name}__q1") - F.lit(IQR_FACTOR) * iqr
        high = F.col(f"{col_name}__q3") + F.lit(IQR_FACTOR) * iqr
        select_exprs.append(
            F.when(F.col(col_name) < low,  low)
             .when(F.col(col_name) > high, high)
             .otherwise(F.col(col_name))
             .alias(col_name)
        )
    else:
        select_exprs.append(F.col(col_name))

df_winsor = df_w.select(*select_exprs)
print("Winsorización aplicada.")

# COMMAND ----------

from pyspark.ml.stat import Correlation as SparkCorr

assembler_corr = VectorAssembler(
    inputCols=numeric_cols, outputCol="features_corr", handleInvalid="skip"
)
df_vec_corr = assembler_corr.transform(df_winsor.select(*numeric_cols))

corr_matrix = SparkCorr.corr(df_vec_corr, "features_corr", method="pearson").head()[0]
corr_pd = pd.DataFrame(corr_matrix.toArray(), index=numeric_cols, columns=numeric_cols)

display(corr_pd)

high_corr_pairs = []

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        corr_value = corr_pd.iloc[i, j]
        
        if abs(corr_value) >= CORR_THRESHOLD:
            high_corr_pairs.append({
                "variable_1": numeric_cols[i],
                "variable_2": numeric_cols[j],
                "correlacion": float(corr_value),
                "correlacion_abs": float(abs(corr_value))
            })

high_corr_pd = pd.DataFrame(high_corr_pairs)

if not high_corr_pd.empty:
    high_corr_pd = high_corr_pd.sort_values("correlacion_abs", ascending=False)
    display(high_corr_pd)
else:
    print(f"No hay pares de variables con correlaci\u00f3n >= {CORR_THRESHOLD}")

# COMMAND ----------

# Detección automática de pares altamente correlados
cols_to_drop_auto = set()

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        if abs(corr_pd.iloc[i, j]) >= CORR_THRESHOLD:
            col_i = numeric_cols[i]
            col_j = numeric_cols[j]

            # si una es _avg y otra _best -> eliminar  _best

            if col_i.endswith("_avg") and col_j.endswith("_best"):
                cols_to_drop_auto.add(col_j)

            elif col_i.endswith("_best") and col_j.endswith("_avg"):
                cols_to_drop_auto.add(col_i)


print(f"Columnas eliminadas por correlación >= {CORR_THRESHOLD}:")
print(sorted(cols_to_drop_auto))

# ────────────────────────────────────────────────────────────────────────────

MANUAL_DROP = [
    "concentric_mean_power_bm_w_kg_avg",
    "concentric_mean_power_bm_w_kg_best",
    "peak_power_bm_w_kg_avg",
]

cols_to_drop_final = list(cols_to_drop_auto | set(MANUAL_DROP))
feature_cols = [c for c in numeric_cols if c not in cols_to_drop_final]

df_final = df_winsor.drop(*cols_to_drop_final)
print(f"Features para clustering: {len(feature_cols)}")
print(feature_cols)

# COMMAND ----------

plt.figure(figsize=(max(10, len(feature_cols)), max(8, len(feature_cols) - 2)))
sns.heatmap(
    corr_pd.loc[feature_cols, feature_cols],
    annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, annot_kws={"size": 7}
)
plt.title("Correlación entre features finales")
plt.tight_layout()
plt.show()

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip"
)
df_vec = assembler.transform(df_final.select([ID_COL] + feature_cols))

scaler = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled",
    withMean=True, withStd=True
)
df_scaled = scaler.fit(df_vec).transform(df_vec)

print(f"Filas listas para clustering: {df_scaled.count():,}")

# COMMAND ----------


MAX_PCA = min(len(feature_cols), 10)

pca_model_full = PCA(
    k=MAX_PCA, inputCol="features_scaled", outputCol="pca_features"
).fit(df_scaled)

explained = pca_model_full.explainedVariance.toArray()
cumulative = np.cumsum(explained)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, MAX_PCA + 1), explained * 100, alpha=0.6, label="Varianza por PC")
ax.plot(range(1, MAX_PCA + 1), cumulative * 100, "o-", color="red", label="Acumulada")
ax.axhline(80, linestyle="--", color="gray", linewidth=0.8, label="80%")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Varianza explicada (%)")
ax.set_title("Varianza explicada por PCA")
ax.legend()
plt.tight_layout()
plt.show()

# Elige automáticamente el nº de PCs que acumulan ≥ 80% de varianza
PCA_K = int(np.argmax(cumulative >= 0.80)) + 1
print(f"\nPCs seleccionadas para ≥80% varianza: {PCA_K}")

# COMMAND ----------

pca_final = PCA(
    k=PCA_K, inputCol="features_scaled", outputCol="pca_features"
).fit(df_scaled)

df_pca = pca_final.transform(df_scaled)

to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")
df_pca = df_pca.withColumn("pca_array", to_array(F.col("pca_features")))

for i in range(PCA_K):
    df_pca = df_pca.withColumn(f"PC{i+1}", F.col("pca_array")[i])

df_pca = df_pca.drop("pca_array")

# COMMAND ----------

K_VALUES = list(range(2, 9))

evaluator_kmeans = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

kmeans_results = []

for k in K_VALUES:
    km = KMeans(k=k, seed=SEED, featuresCol="features_scaled", predictionCol="prediction")
    km_model = km.fit(df_scaled)
    preds = km_model.transform(df_pca)
    sil  = evaluator_kmeans.evaluate(preds)
    cost = km_model.summary.trainingCost
    kmeans_results.append({"k": k, "silhouette": sil, "inertia": cost})
    print(f"k={k} | silhouette={sil:.4f} | inertia={cost:,.1f}")

results_pd = pd.DataFrame(kmeans_results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(results_pd["k"], results_pd["inertia"], "o-")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Curva del codo"); ax1.grid(True)

ax2.plot(results_pd["k"], results_pd["silhouette"], "o-", color="orange")
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette score")
ax2.set_title("Silhouette por k"); ax2.grid(True)

plt.tight_layout()
plt.show()

# COMMAND ----------

K_FINAL = 4

km_final = KMeans(k=K_FINAL, seed=SEED, featuresCol="features_scaled", predictionCol="cluster")
km_model_final = km_final.fit(df_scaled)
df_clustered = km_model_final.transform(df_pca)

print("Tamaño de cada cluster:")
display(df_clustered.groupBy("cluster").count().orderBy("cluster"))

# COMMAND ----------

from sklearn.cluster import HDBSCAN

pc_cols = [f"PC{i+1}" for i in range(PCA_K)]

df_hdb_pd = df_pca.select([ID_COL] + pc_cols).toPandas()

hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
df_hdb_pd["cluster_hdbscan"] = hdb.fit_predict(df_hdb_pd[pc_cols].values)

print("Distribución de clusters HDBSCAN (-1 = ruido):")
print(df_hdb_pd["cluster_hdbscan"].value_counts().sort_index())

# COMMAND ----------

df_plot = (
    df_clustered
    .select([ID_COL, "cluster"] + pc_cols)
    .toPandas()
    .merge(df_hdb_pd[[ID_COL, "cluster_hdbscan"]], on=ID_COL, how="left")
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, col, title in zip(
    axes,
    ["cluster", "cluster_hdbscan"],
    [f"KMeans k={K_FINAL}", "HDBSCAN"]
):
    scatter = ax.scatter(
        df_plot["PC1"], df_plot["PC2"],
        c=df_plot[col], cmap="tab10", alpha=0.7, s=40
    )
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title); ax.grid(True)
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# COMMAND ----------

profile_pd = (
    df_clustered
    .groupBy("cluster")
    .agg(
        F.count("*").alias("n_jugadores"),
        *[F.round(F.avg(c), 3).alias(c) for c in feature_cols]
    )
    .orderBy("cluster")
    .toPandas()
)

display(profile_pd)

# COMMAND ----------


