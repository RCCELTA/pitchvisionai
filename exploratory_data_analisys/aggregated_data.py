# Databricks notebook source
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

# COMMAND ----------

TABLE_ANTHROPOMETRY      = "dev_silver.pitchvision.v_anthropometry"
TABLE_COMPETITIVE        = "dev_silver.pitchvision.v_competitive_profile"
TABLE_GPS                = "dev_silver.pitchvision.v_gps_data"
TABLE_CMJ                = "dev_silver.pitchvision.v_physical_test_cmj"
TABLE_ISOPRONE           = "dev_silver.pitchvision.v_physical_test_isoprone"
TABLE_NORDIC             = "dev_silver.pitchvision.v_physical_test_nordic"
TABLE_SLJ                = "dev_silver.pitchvision.v_physical_test_slj"


TABLE_AGREGATED_PHYSICAL_TESTS = "dev_gold.pitchvision.agregado_physical_tests"
TABLE_ANTHROPOMETRY_CLEANED = "dev_gold.pitchvision.anthropometrylimpio_escalado_listo_para_clustering"

df_anthropometry = spark.table(TABLE_ANTHROPOMETRY)
df_competitive   = spark.table(TABLE_COMPETITIVE)
df_gps           = spark.table(TABLE_GPS)
df_cmj           = spark.table(TABLE_CMJ)
df_isoprone      = spark.table(TABLE_ISOPRONE)
df_nordic        = spark.table(TABLE_NORDIC)
df_slj           = spark.table(TABLE_SLJ)


df_agregated_physical_tests = spark.table(TABLE_AGREGATED_PHYSICAL_TESTS)
df_antropometry_cleaned = spark.table(TABLE_ANTHROPOMETRY_CLEANED)


ID_COL      = "id_unico_deportista"
DATE_COL    = "test_date"
SEASON_COL  = "season"
AGE_COL     = "age_years"
EXCLUDE_ALWAYS = [ID_COL, DATE_COL, SEASON_COL]

PHYSICAL_TABLES = {
    "cmj":         df_cmj,
    "isoprone":    df_isoprone,
    "nordic":      df_nordic,
    "slj":         df_slj,
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Tabla antropometrica

# COMMAND ----------

print(f"\n{'=' * 60}")
print(f"TABLA: anthropometry")
print('=' * 60)

numeric_cols_antro = [
    f.name for f in df_anthropometry.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]
print(f"  Columnas numéricas: {len(numeric_cols_antro)}")
print(f"  {numeric_cols_antro}")

print(f"\n Jugadores únicos totales:")
print(f"  {df_anthropometry.select(ID_COL).distinct().count():,}")
print(f"\n  Jugadores únicos por temporada:")
season_counts_antro = (
    df_anthropometry
    .groupBy(SEASON_COL)
    .agg(F.countDistinct(ID_COL).alias("jugadores_unicos"))
    .orderBy(SEASON_COL)
)
for row in season_counts_antro.collect():
    print(f"    {row[SEASON_COL]}: {row['jugadores_unicos']:,} jugadores")

print(f"\n  Total filas: {df_anthropometry.count():,}")
display(df_anthropometry.limit(5))

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from itertools import combinations

# ── JUGADORES POR TEMPORADA ───────────────────────────────────────────────────
seasons = [r[SEASON_COL] for r in df_anthropometry.select(SEASON_COL).distinct().orderBy(SEASON_COL).collect()]

season_sets = {}
for s in seasons:
    ids = df_anthropometry.filter(F.col(SEASON_COL) == s).select(ID_COL).distinct()
    season_sets[s] = set(r[ID_COL] for r in ids.collect())

# ── CONTEOS DE SOLAPAMIENTO ───────────────────────────────────────────────────
s1, s2, s3 = seasons
A, B, C = season_sets[s1], season_sets[s2], season_sets[s3]

only_A     = A - B - C
only_B     = B - A - C
only_C     = C - A - B
AB_only    = (A & B) - C
AC_only    = (A & C) - B
BC_only    = (B & C) - A
ABC        = A & B & C

print(f"  Solo {s1}:            {len(only_A):>3} jugadores")
print(f"  Solo {s2}:            {len(only_B):>3} jugadores")
print(f"  Solo {s3}:            {len(only_C):>3} jugadores")
print(f"  {s1} ∩ {s2} solo:     {len(AB_only):>3} jugadores")
print(f"  {s1} ∩ {s3} solo:     {len(AC_only):>3} jugadores")
print(f"  {s2} ∩ {s3} solo:     {len(BC_only):>3} jugadores")
print(f"  {s1} ∩ {s2} ∩ {s3}:  {len(ABC):>3} jugadores")
print(f"  TOTAL únicos:         {len(A | B | C):>3} jugadores")

# ── DIAGRAMA DE VENN ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
venn3(
    subsets=(len(only_A), len(only_B), len(AB_only),
             len(only_C), len(AC_only), len(BC_only), len(ABC)),
    set_labels=(s1, s2, s3),
    ax=ax
)
ax.set_title("Distribución de jugadores únicos por temporada — Antropometría", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ── NULOS ANTROPOMETRÍA ───────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"NULOS — TABLA: anthropometry")
print('=' * 60)

numeric_cols_antro = [
    f.name for f in df_anthropometry.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]

total_rows_antro = df_anthropometry.count()
seasons_antro    = [r[SEASON_COL] for r in df_anthropometry.select(SEASON_COL).distinct().orderBy(SEASON_COL).collect()]

# ── NULOS GLOBALES ────────────────────────────────────────────────────────────
null_exprs_antro = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols_antro]

nulls_antro = df_anthropometry.agg(*null_exprs_antro).toPandas().T.reset_index()
nulls_antro.columns = ["columna", "nulos_total"]
nulls_antro = nulls_antro.set_index("columna")
nulls_antro["pct_total"] = (nulls_antro["nulos_total"] / total_rows_antro * 100).round(2)

# ── NULOS POR SEASON ──────────────────────────────────────────────────────────
for season in seasons_antro:
    df_s    = df_anthropometry.filter(F.col(SEASON_COL) == season)
    total_s = df_s.count()
    null_s  = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols_antro]
    nulls_s = df_s.agg(*null_s).toPandas().T.reset_index()
    nulls_s.columns = ["columna", f"nulos_{season}"]
    nulls_s[f"pct_{season}"] = (nulls_s[f"nulos_{season}"] / total_s * 100).round(2)
    nulls_s = nulls_s.set_index("columna")
    nulls_antro = nulls_antro.join(nulls_s, how="left")

nulls_antro = nulls_antro.sort_values("pct_total", ascending=False)

# solo filas con algún nulo
nulls_antro_display = nulls_antro[nulls_antro["nulos_total"] > 0]

if nulls_antro_display.empty:
    print("  Sin nulos en ninguna columna ni temporada")
else:
    display(nulls_antro_display.reset_index())

# ── DESCARTE >50% NULOS GLOBALES ──────────────────────────────────────────────
cols_high_null_antro   = nulls_antro[nulls_antro["pct_total"] > 50].index.tolist()
numeric_cols_antro_clean = [c for c in numeric_cols_antro if c not in cols_high_null_antro]
df_anthropometry_clean   = df_anthropometry.drop(*cols_high_null_antro)

print(f"\n  Total filas:                       {total_rows_antro:,}")
print(f"  Columnas originales:               {len(numeric_cols_antro)}")
print(f"  Columnas descartadas (>50% nulos): {len(cols_high_null_antro)} {cols_high_null_antro}")
print(f"  Columnas restantes:                {len(numeric_cols_antro_clean)}")

# COMMAND ----------

AGE_COL = "age_years"

EXCLUDE_COLS = [
    ID_COL,
    AGE_COL,
    "season",
    "is_restricted"
]

# Aseguramos que test_date sea tipo date
df_imputed = df_anthropometry.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))

# Columnas numéricas a imputar, excepto edad e identificadores
numeric_cols_to_impute = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in EXCLUDE_COLS
]

print(f"Columnas numéricas a imputar con mediciones del mismo jugador: {len(numeric_cols_to_impute)}")
print(numeric_cols_to_impute)

# VENTANAS TEMPORALES POR JUGADOR
w_forward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(Window.unboundedPreceding, 0)
)

w_backward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(0, Window.unboundedFollowing)
)

# 1. IMPUTACIÓN ESPECIAL PARA age_years
#    Ajusta edad según diferencia entre test_date y la fecha
#    donde sí existe age_years.
prev_age_struct = F.last(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_forward)

next_age_struct = F.first(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_backward)

df_imputed = (
    df_imputed
    .withColumn("_prev_age_ref", prev_age_struct)
    .withColumn("_next_age_ref", next_age_struct)
)

# Si usamos una medición anterior:
age_from_prev = (
    F.col("_prev_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_prev_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

# Si usamos una medición posterior:
age_from_next = (
    F.col("_next_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_next_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

df_imputed = (
    df_imputed
    .withColumn(
        AGE_COL,
        F.coalesce(
            F.col(AGE_COL),
            age_from_prev,
            age_from_next
        ).cast("int")
    )
    .drop("_prev_age_ref", "_next_age_ref")
)

print("Imputación de age_years terminada.")

# 2. IMPUTACIÓN DEL RESTO DE VARIABLES NUMÉRICAS
for c in numeric_cols_to_impute:
    last_value = F.last(F.col(c), ignorenulls=True).over(w_forward)
    next_value = F.first(F.col(c), ignorenulls=True).over(w_backward)

    df_imputed = df_imputed.withColumn(
        c,
        F.coalesce(
            F.col(c),
            last_value,
            next_value
        )
    )

print("Imputación del resto de variables numéricas terminada.")

# 3. CACHEAR DF
df_imputed = df_imputed.cache()

print(f"Filas finales: {df_imputed.count():,}")

display(df_imputed.limit(20))

# COMMAND ----------

# ── NULOS TRAS IMPUTACIÓN ─────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"NULOS TRAS IMPUTACIÓN — TABLA: anthropometry")
print('=' * 60)

all_numeric = [
    f.name for f in df_imputed.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]

total_rows_imp = df_imputed.count()
seasons_imp    = [r[SEASON_COL] for r in df_imputed.select(SEASON_COL).distinct().orderBy(SEASON_COL).collect()]

# ── NULOS GLOBALES ────────────────────────────────────────────────────────────
null_exprs_imp = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in all_numeric]

nulls_imp = df_imputed.agg(*null_exprs_imp).toPandas().T.reset_index()
nulls_imp.columns = ["columna", "nulos_total"]
nulls_imp = nulls_imp.set_index("columna")
nulls_imp["pct_total"] = (nulls_imp["nulos_total"] / total_rows_imp * 100).round(2)

# ── NULOS POR SEASON ──────────────────────────────────────────────────────────
for season in seasons_imp:
    df_s    = df_imputed.filter(F.col(SEASON_COL) == season)
    total_s = df_s.count()
    null_s  = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in all_numeric]
    nulls_s = df_s.agg(*null_s).toPandas().T.reset_index()
    nulls_s.columns = ["columna", f"nulos_{season}"]
    nulls_s[f"pct_{season}"] = (nulls_s[f"nulos_{season}"] / total_s * 100).round(2)
    nulls_s = nulls_s.set_index("columna")
    nulls_imp = nulls_imp.join(nulls_s, how="left")

nulls_imp = nulls_imp.sort_values("pct_total", ascending=False)

nulls_imp_display = nulls_imp[nulls_imp["nulos_total"] > 0]

if nulls_imp_display.empty:
    print("  Sin nulos tras imputación")
else:
    print(f"  Columnas con nulos restantes: {len(nulls_imp_display)}")
    display(nulls_imp_display.reset_index())

print(f"\n  Total filas:               {total_rows_imp:,}")
print(f"  Columnas analizadas:       {len(all_numeric)}")
print(f"  Columnas sin nulos:        {len(nulls_imp[nulls_imp['nulos_total'] == 0])}")
print(f"  Columnas con nulos restantes: {len(nulls_imp_display)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Agregado medias jugador x temporada tablas tests fisicos

# COMMAND ----------

player_season_dfs = {}

for name, df in PHYSICAL_TABLES.items():
    print(f"\n{'=' * 60}")
    print(f"TABLA: {name}")
    print('=' * 60)

    numeric_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in EXCLUDE_ALWAYS
    ]
    print(f"  Columnas numéricas: {len(numeric_cols)}")
    print(f"  {numeric_cols}")

    agg_exprs = [F.avg(c).alias(c) for c in numeric_cols]

    df_agg = (
        df
        .groupBy(ID_COL, SEASON_COL)
        .agg(*agg_exprs)
        .orderBy(ID_COL, SEASON_COL)
    )

    player_season_dfs[name] = df_agg

    print(f"\n  Jugadores únicos por temporada:")
    season_counts = (
        df_agg
        .groupBy(SEASON_COL)
        .agg(F.countDistinct(ID_COL).alias("jugadores_unicos"))
        .orderBy(SEASON_COL)
    )
    for row in season_counts.collect():
        print(f"    {row[SEASON_COL]}: {row['jugadores_unicos']:,} jugadores")


    print(f"\n  Filas tras agregación (jugador x temporada): {df_agg.count():,}")
    display(df_agg.limit(5))

print(f"\n{'=' * 60}")
print("DFs por jugador y temporada disponibles en player_season_dfs:")
for k in player_season_dfs:
    print(f"   player_season_dfs['{k}']")

# COMMAND ----------

# ── NULOS POR TABLA (jugador x temporada) ────────────────────────────────────
player_season_dfs_clean = {}

for name, df_agg in player_season_dfs.items():
    print(f"\n{'=' * 60}")
    print(f"NULOS — TABLA: {name}")
    print('=' * 60)

    numeric_cols = [
        f.name for f in df_agg.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in EXCLUDE_ALWAYS
    ]

    total_rows = df_agg.count()
    seasons    = [r[SEASON_COL] for r in df_agg.select(SEASON_COL).distinct().orderBy(SEASON_COL).collect()]

    # ── NULOS GLOBALES ────────────────────────────────────────────────────
    null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols]

    nulls_pd = df_agg.agg(*null_exprs).toPandas().T.reset_index()
    nulls_pd.columns = ["columna", "nulos_total"]
    nulls_pd = nulls_pd.set_index("columna")
    nulls_pd["pct_total"] = (nulls_pd["nulos_total"] / total_rows * 100).round(2)

    # ── NULOS POR SEASON ──────────────────────────────────────────────────
    for season in seasons:
        df_s      = df_agg.filter(F.col(SEASON_COL) == season)
        total_s   = df_s.count()
        null_s    = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols]
        nulls_s   = df_s.agg(*null_s).toPandas().T.reset_index()
        nulls_s.columns = ["columna", f"nulos_{season}"]
        nulls_s[f"pct_{season}"] = (nulls_s[f"nulos_{season}"] / total_s * 100).round(2)
        nulls_s   = nulls_s.set_index("columna")
        nulls_pd  = nulls_pd.join(nulls_s, how="left")

    nulls_pd = nulls_pd.sort_values("pct_total", ascending=False)

    # solo filas con algún nulo
    nulls_pd_display = nulls_pd[nulls_pd["nulos_total"] > 0]

    if nulls_pd_display.empty:
        print("  Sin nulos en ninguna columna ni temporada")
    else:
        display(nulls_pd_display.reset_index())

    # ── DESCARTE >50% NULOS GLOBALES ──────────────────────────────────────
    cols_high_null   = nulls_pd[nulls_pd["pct_total"] > 50].index.tolist()
    numeric_cols_clean = [c for c in numeric_cols if c not in cols_high_null]
    df_clean         = df_agg.drop(*cols_high_null)

    print(f"\n  Total filas:                       {total_rows:,}")
    print(f"  Columnas originales:               {len(numeric_cols)}")
    print(f"  Columnas descartadas (>50% nulos): {len(cols_high_null)} {cols_high_null}")
    print(f"  Columnas restantes:                {len(numeric_cols_clean)}")

    player_season_dfs_clean[name] = {
        "df":           df_clean,
        "numeric_cols": numeric_cols_clean,
    }

print(f"\n{'=' * 60}")
print("DFs limpios disponibles en player_season_dfs_clean:")
for k in player_season_dfs_clean:
    print(f"   player_season_dfs_clean['{k}']['df']")
    print(f"   player_season_dfs_clean['{k}']['numeric_cols']")

# COMMAND ----------

#%pip install matplotlib_venn

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# ── DIAGRAMA DE VENN POR TABLA ────────────────────────────────────────────────
seasons = ["23-24", "24-25", "25-26"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, df_agg) in enumerate(player_season_dfs.items()):

    # sets de jugadores por temporada
    season_sets = {}
    for s in seasons:
        ids = df_agg.filter(F.col(SEASON_COL) == s).select(ID_COL).distinct()
        season_sets[s] = set(r[ID_COL] for r in ids.collect())

    A, B, C = season_sets[seasons[0]], season_sets[seasons[1]], season_sets[seasons[2]]

    only_A  = A - B - C
    only_B  = B - A - C
    only_C  = C - A - B
    AB_only = (A & B) - C
    AC_only = (A & C) - B
    BC_only = (B & C) - A
    ABC     = A & B & C

    ax = axes[idx]
    venn3(
        subsets=(
            len(only_A), len(only_B), len(AB_only),
            len(only_C), len(AC_only), len(BC_only),
            len(ABC)
        ),
        set_labels=(seasons[0], seasons[1], seasons[2]),
        ax=ax
    )
    ax.set_title(f"Tabla: {name}  |  {len(A | B | C)} jugadores únicos", fontsize=11)

    # resumen texto
    summary = (
        f"Solo {seasons[0]}: {len(only_A)}  |  "
        f"Solo {seasons[1]}: {len(only_B)}  |  "
        f"Solo {seasons[2]}: {len(only_C)}\n"
        f"{seasons[0]}∩{seasons[1]}: {len(AB_only)}  |  "
        f"{seasons[0]}∩{seasons[2]}: {len(AC_only)}  |  "
        f"{seasons[1]}∩{seasons[2]}: {len(BC_only)}  |  "
        f"∩3: {len(ABC)}"
    )
    ax.set_xlabel(summary, fontsize=8)

plt.suptitle("Distribución de jugadores únicos por temporada", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

#%pip install upsetplot
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_memberships
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# ── MEMBERSHIPS ───────────────────────────────────────────────────────────────
table_sets = {}
for name, df_agg in player_season_dfs.items():
    ids = df_agg.select(ID_COL).distinct()
    table_sets[name] = set(r[ID_COL] for r in ids.collect())
    print(f"  {name}: {len(table_sets[name]):,} jugadores únicos")
names      = list(table_sets.keys())
all_ids    = reduce(lambda a, b: a | b, table_sets.values())
common_ids = reduce(lambda a, b: a & b, table_sets.values())

# para cada jugador, en qué tablas aparece
memberships = []
for player_id in all_ids:
    membership = [n for n in names if player_id in table_sets[n]]
    memberships.append(membership)

data = from_memberships(memberships)

# ── UPSET PLOT ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))
upset = UpSet(data, subset_size="count", show_counts=True, sort_by="cardinality")
upset.plot(fig=fig)

plt.suptitle(
    f"Intersecciones entre tablas físicas  |  Unión: {len(all_ids)}  |  En las 4 tablas: {len(common_ids)}",
    fontsize=12, fontweight="bold", y=1.02
)
plt.subplots_adjust(top=0.9)
plt.show()

# ── RESUMEN ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print(f"  RESUMEN")
print(f"{'=' * 50}")
for n in names:
    print(f"  {n:<12}: {len(table_sets[n]):>3} jugadores únicos")
print(f"{'─' * 50}")
print(f"  Unión (≥1 tabla) : {len(all_ids):>3} jugadores")
print(f"  En las 4 tablas  : {len(common_ids):>3} jugadores")
for n in names:
    solo = len(table_sets[n] - reduce(lambda a, b: a | b, [v for k, v in table_sets.items() if k != n]))
    print(f"  Solo en {n:<10}: {solo:>3} jugadores")

# COMMAND ----------

#%pip install upsetplot
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_memberships
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
 
# ── MEMBERSHIPS (PLAYER-SEASON) ───────────────────────────────────────────────
table_sets = {}
for name, df in PHYSICAL_TABLES.items():
    # Crear combinación única de jugador-temporada
    player_seasons = df.select(ID_COL, SEASON_COL).distinct()
    table_sets[name] = set((r[ID_COL], r[SEASON_COL]) for r in player_seasons.collect())
    print(f"  {name}: {len(table_sets[name]):,} combinaciones jugador-temporada")
 
names      = list(table_sets.keys())
all_combinations = reduce(lambda a, b: a | b, table_sets.values())
common_combinations = reduce(lambda a, b: a & b, table_sets.values())
 
# Para cada combinación jugador-temporada, en qué tablas aparece
memberships = []
for player_season in all_combinations:
    membership = [n for n in names if player_season in table_sets[n]]
    memberships.append(membership)
 
data = from_memberships(memberships)
 
# ── UPSET PLOT ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))
upset = UpSet(data, subset_size="count", show_counts=True, sort_by="cardinality")
upset.plot(fig=fig)
 
plt.suptitle(
    f"Intersecciones entre tablas físicas (jugador-temporada)  |  Unión: {len(all_combinations)}  |  En las 4 tablas: {len(common_combinations)}",
    fontsize=12, fontweight="bold", y=1.02
)
plt.subplots_adjust(top=0.9)
plt.show()
 
# ── RESUMEN ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  RESUMEN (JUGADOR-TEMPORADA)")
print(f"{'=' * 60}")
for n in names:
    print(f"  {n:<12}: {len(table_sets[n]):>4} combinaciones jugador-temporada")
print(f"{'─' * 60}")
print(f"  Unión (≥1 tabla)        : {len(all_combinations):>4} combinaciones")
print(f"  En las 4 tablas         : {len(common_combinations):>4} combinaciones")
for n in names:
    solo = len(table_sets[n] - reduce(lambda a, b: a | b, [v for k, v in table_sets.items() if k != n]))
    print(f"  Solo en {n:<14}: {solo:>4} combinaciones")
 
# ── ANÁLISIS POR TEMPORADA ────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  DESGLOSE POR TEMPORADA")
print(f"{'=' * 60}")
 
# Extraer todas las temporadas únicas
all_seasons = set()
for combinations in table_sets.values():
    all_seasons.update([season for _, season in combinations])
all_seasons = sorted(all_seasons)
 
for season in all_seasons:
    print(f"\n  Temporada: {season}")
    print(f"  {'-' * 56}")
    season_sets = {}
    for name, combinations in table_sets.items():
        season_sets[name] = set([player_id for player_id, s in combinations if s == season])
        print(f"    {name:<12}: {len(season_sets[name]):>3} jugadores")
    
    season_union = reduce(lambda a, b: a | b, season_sets.values())
    season_common = reduce(lambda a, b: a & b, season_sets.values())
    print(f"    {'─' * 52}")
    print(f"    Unión (≥1 tabla) : {len(season_union):>3} jugadores")
    print(f"    En las 4 tablas  : {len(season_common):>3} jugadores")

# COMMAND ----------

SEASONS_ORDERED = ["25-26", "24-25", "23-24"]

# ── PASO 1: para cada jugador y tabla, coger la temporada más reciente ────────
availability = {}
for name, df_agg in player_season_dfs.items():
    availability[name] = set(
        (r[ID_COL], r[SEASON_COL])
        for r in df_agg.select(ID_COL, SEASON_COL).collect()
    )

all_ids = reduce(
    lambda a, b: a | b,
    [set(pid for pid, _ in v) for v in availability.values()]
)

# quedarnos solo con los 286 que tienen datos en las 4 tablas
ids_4_tablas = {
    pid for pid in all_ids
    if all(
        any((pid, s) in availability[name] for s in SEASONS_ORDERED)
        for name in PHYSICAL_TABLES
    )
}
print(f"Jugadores con datos en las 4 tablas (cualquier temporada): {len(ids_4_tablas):,}")

# para cada jugador y tabla, la temporada más reciente disponible
player_table_season = {}
for name in PHYSICAL_TABLES:
    for player_id in ids_4_tablas:
        for season in SEASONS_ORDERED:
            if (player_id, season) in availability[name]:
                player_table_season[(player_id, name)] = season
                break

# ── PASO 2: join tabla a tabla con su propia best_season ─────────────────────
import pandas as pd

df_joined = spark.createDataFrame(
    pd.DataFrame({ID_COL: list(ids_4_tablas)})
)

for name, df_agg in player_season_dfs.items():
    num_cols = player_season_dfs_clean[name]["numeric_cols"]

    # lookup id -> best_season para esta tabla
    season_map_pd = pd.DataFrame([
        {ID_COL: pid, f"_best_{name}": player_table_season[(pid, name)]}
        for pid in ids_4_tablas
    ])
    df_season_map = spark.createDataFrame(season_map_pd)

    # renombrar cols con prefijo
    df_table = df_agg.select(ID_COL, SEASON_COL, *num_cols)
    for c in num_cols:
        df_table = df_table.withColumnRenamed(c, f"{name}__{c}")

    df_joined = (
        df_joined
        .join(df_season_map, on=ID_COL, how="left")
        .join(
            df_table,
            on=(
                (df_joined[ID_COL] == df_table[ID_COL]) &
                (F.col(f"_best_{name}") == df_table[SEASON_COL])
            ),
            how="left"
        )
        .drop(df_table[ID_COL])
        .drop(SEASON_COL)
        .drop(f"_best_{name}")
    )

df_joined = df_joined.cache()

print(f"\n{'=' * 60}")
print(f"  RESULTADO DEL JOIN")
print(f"{'=' * 60}")
print(f"  Filas totales:    {df_joined.count():,}")
print(f"  Columnas totales: {len(df_joined.columns)}")
display(df_joined.limit(5))

# COMMAND ----------

# ── NULOS TRAS JOIN ───────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"NULOS — df_joined (286 jugadores x 4 tablas físicas)")
print('=' * 60)

numeric_cols_joined = [
    f.name for f in df_joined.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in [ID_COL]
]

total_rows = df_joined.count()

null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols_joined]

nulls_pd = df_joined.agg(*null_exprs).toPandas().T.reset_index()
nulls_pd.columns = ["columna", "nulos_total"]
nulls_pd = nulls_pd.set_index("columna")
nulls_pd["pct_total"] = (nulls_pd["nulos_total"] / total_rows * 100).round(2)
nulls_pd = nulls_pd.sort_values("pct_total", ascending=False)

nulls_display = nulls_pd[nulls_pd["nulos_total"] > 0]

if nulls_display.empty:
    print("  Sin nulos en ninguna columna")
else:
    print(f"  Columnas con nulos: {len(nulls_display)}")
    display(nulls_display.reset_index())

print(f"\n  Total filas:            {total_rows:,}")
print(f"  Columnas analizadas:    {len(numeric_cols_joined)}")
print(f"  Columnas sin nulos:     {len(nulls_pd[nulls_pd['nulos_total'] == 0])}")
print(f"  Columnas con nulos:     {len(nulls_display)}")

# COMMAND ----------

# ── ELIMINAR COLUMNAS CON MUCHOS NULOS ───────────────────────────────────────
cols_to_drop = [
    "nordic__right_max_force_normalized_bw",
    "nordic__left_max_force_normalized_bw",
    "isoprone__right_max_force_normalized_bw",
    "isoprone__left_max_force_normalized_bw",
]

df_joined = df_joined.drop(*cols_to_drop)

# ── IMPUTAR NULOS RESIDUALES CON MEDIANA ─────────────────────────────────────
cols_to_impute = [
    "isoprone__torque_asym_pct",
    "nordic__torque_asym_pct",
    "cmj__jump_height_imp_mom_cm_stddev",
]

for c in cols_to_impute:
    median_val = df_joined.approxQuantile(c, [0.5], 0.001)[0]
    df_joined  = df_joined.fillna({c: median_val})
    print(f"  {c}: imputado con mediana = {median_val:.4f}")

df_joined = df_joined.cache()
print(f"\n  Columnas restantes: {len(df_joined.columns)}")
print(f"  Filas:              {df_joined.count():,}")

# COMMAND ----------

df_joined.write.saveAsTable(
    "dev_gold.pitchvision.agregado_physical_tests"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Agregado physical tests

# COMMAND ----------

TABLE_AGREGADOS = "dev_gold.pitchvision.agregado_physical_tests"
df_joined = spark.table(TABLE_AGREGADOS)
display(df_joined)

# COMMAND ----------

print(f"\n{'=' * 60}")
print(f"TABLA: aggregated_physical_tests")
print('=' * 60)

numeric_cols_aggregated_physical_test = [
    f.name for f in df_joined.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]
print(f"  Columnas numéricas: {len(numeric_cols_aggregated_physical_test)}")
print(f"  {numeric_cols_aggregated_physical_test}")

print(f"\n Jugadores únicos totales:")
print(f"  {df_joined.select(ID_COL).distinct().count():,}")

print(f"\n  Total filas: {df_joined.count():,}")
display(df_joined.limit(5))

# COMMAND ----------

percentile_rows = []

for c in numeric_cols_aggregated_physical_test:
    try:
        p = df_joined.approxQuantile(c, [0.05, 0.25, 0.50, 0.75, 0.95], 0.01)
        percentile_rows.append((c, p[0], p[1], p[2], p[3], p[4]))
    except Exception as e:
        print(f"No se pudieron calcular percentiles para {c}: {e}")

if percentile_rows:
    percentiles_df = spark.createDataFrame(
        percentile_rows,
        ["columna", "p05", "p25", "p50_mediana", "p75", "p95"]
    )
    display(percentiles_df)
else:
    print("No hay percentiles disponibles.")

# COMMAND ----------

ID_COL = "id_unico_deportista"

# Convertimos percentiles_df a diccionario local con más percentiles
percentiles_dict = {
    row["columna"]: {
        "p05": row["p05"],
        "p25": row["p25"],
        "p50": row["p50_mediana"],
        "p75": row["p75"],
        "p95": row["p95"]
    }
    for row in percentiles_df.collect()
}

extreme_severity_rows = []

for c in numeric_cols_aggregated_physical_test:
    try:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]

        iqr = p75 - p25

        if iqr == 0 or iqr is None:
            continue

        tmp = (
            df_joined
            .select(c)
            .where(F.col(c).isNotNull())
            .withColumn(
                "exceso_extremo",
                F.when(F.col(c) < F.lit(p05), F.lit(p05) - F.col(c))
                 .when(F.col(c) > F.lit(p95), F.col(c) - F.lit(p95))
                 .otherwise(F.lit(0.0))
            )
            .withColumn(
                "exceso_extremo_normalizado",
                F.col("exceso_extremo") / F.lit(iqr)
            )
        )

        stats = tmp.agg(
            F.count("*").alias("n_no_nulos"),
            F.sum(F.when(F.col("exceso_extremo") > 0, 1).otherwise(0)).alias("n_fuera_p05_p95"),
            F.avg("exceso_extremo_normalizado").alias("media_exceso_norm"),
            F.max("exceso_extremo_normalizado").alias("max_exceso_norm"),
            F.expr("percentile_approx(exceso_extremo_normalizado, 0.95)").alias("p95_exceso_norm")
        ).collect()[0]

        extreme_severity_rows.append((
            c,
            p05,
            p25,
            p75,
            p95,
            iqr,
            stats["n_no_nulos"],
            stats["n_fuera_p05_p95"],
            float(stats["media_exceso_norm"]) if stats["media_exceso_norm"] is not None else None,
            float(stats["p95_exceso_norm"]) if stats["p95_exceso_norm"] is not None else None,
            float(stats["max_exceso_norm"]) if stats["max_exceso_norm"] is not None else None
        ))

    except Exception as e:
        print(f"No se pudo calcular severidad para {c}: {e}")

extreme_severity_df = spark.createDataFrame(
    extreme_severity_rows,
    [
        "columna",
        "p05",
        "p25",
        "p75",
        "p95",
        "iqr",
        "n_no_nulos",
        "n_fuera_p05_p95",
        "media_exceso_normalizado",
        "p95_exceso_normalizado",
        "max_exceso_normalizado"
    ]
)

display(
    extreme_severity_df
    .orderBy(F.col("max_exceso_normalizado").desc())
)

# COMMAND ----------

# ============================================================
# PINTAR DISTRIBUCIONES DE VARIABLES CON max_exceso_normalizado > 5
# ============================================================

import matplotlib.pyplot as plt

THRESHOLD_EXCESO = 5
MAX_POINTS = 10000

extreme_cols_rows = (
    extreme_severity_df
    .filter(F.col("max_exceso_normalizado") > THRESHOLD_EXCESO)
    .orderBy(F.col("max_exceso_normalizado").desc())
    .collect()
)

top_extreme_cols = [row["columna"] for row in extreme_cols_rows]

print(f"Variables con max_exceso_normalizado > {THRESHOLD_EXCESO}:")
print(f"Total variables seleccionadas: {len(top_extreme_cols)}")
print(top_extreme_cols)

if len(top_extreme_cols) == 0:
    print(f"No hay variables con max_exceso_normalizado > {THRESHOLD_EXCESO}")

else:
    for c in top_extreme_cols:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]
        iqr = p75 - p25

        plot_df = (
            df_joined
            .select(c)
            .where(F.col(c).isNotNull())
            .limit(MAX_POINTS)
            .toPandas()
        )

        if plot_df.empty:
            print(f"No hay datos para plotear en columna: {c}")
            continue

        min_val = plot_df[c].min()
        max_val = plot_df[c].max()

        lower_excess = max(0, p05 - min_val) / iqr if iqr != 0 else 0
        upper_excess = max(0, max_val - p95) / iqr if iqr != 0 else 0

        max_exceso = max(lower_excess, upper_excess)

        plt.figure(figsize=(8, 5))

        plt.hist(plot_df[c], bins=50, alpha=0.8)
        plt.axvline(p05, linestyle="--", linewidth=2, label=f"p05 = {p05:.2f}")
        plt.axvline(p95, linestyle="--", linewidth=2, label=f"p95 = {p95:.2f}")

        plt.xlabel(c)
        plt.ylabel("Frecuencia")
        plt.title(
            f"Distribuci\u00f3n de {c}\n"
            f"max_exceso_normalizado \u2248 {max_exceso:.2f} IQR | "
            f"Exceso inferior: {lower_excess:.2f} IQR | "
            f"Exceso superior: {upper_excess:.2f} IQR"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# Winsorización con límites a 3*IQR de los percentiles 05 y 95

winsorized_df = df_joined

for c in numeric_cols_aggregated_physical_test:
    try:
        p05 = percentiles_dict[c]["p05"]
        p25 = percentiles_dict[c]["p25"]
        p75 = percentiles_dict[c]["p75"]
        p95 = percentiles_dict[c]["p95"]
        iqr = p75 - p25

        lower_limit = p05 - 3 * iqr
        upper_limit = p95 + 3 * iqr

        winsorized_df = winsorized_df.withColumn(
            c,
            F.when(F.col(c) < lower_limit, lower_limit)
             .when(F.col(c) > upper_limit, upper_limit)
             .otherwise(F.col(c))
        )
    except Exception as e:
        print(f"No se pudo winsorizar columna {c}: {e}")

print("Winsorización aplicada con límites a 3*IQR de p05 y p95.")
display(winsorized_df.limit(10))

# COMMAND ----------

df_cleaned = winsorized_df


# COMMAND ----------

# Vectorizamos columnas numéricas
assembler = VectorAssembler(
    inputCols=numeric_cols_aggregated_physical_test,
    outputCol="features_corr",
    handleInvalid="keep"
)

df_vector_corr = assembler.transform(df_cleaned.select(*numeric_cols_aggregated_physical_test))

# Calculamos matriz de correlación
corr_matrix = Correlation.corr(
    df_vector_corr,
    "features_corr",
    method="pearson"
).head()[0]

# Convertimos a pandas para visualizar
corr_array = corr_matrix.toArray()

corr_pd = pd.DataFrame(
    corr_array,
    index=numeric_cols_aggregated_physical_test,
    columns=numeric_cols_aggregated_physical_test
)

display(corr_pd)

# COMMAND ----------

CORR_THRESHOLD = 0.95
high_corr_pairs = []

for i in range(len(numeric_cols_aggregated_physical_test)):
    for j in range(i + 1, len(numeric_cols_aggregated_physical_test)):
        corr_value = corr_pd.iloc[i, j]
        
        if abs(corr_value) >= CORR_THRESHOLD:
            high_corr_pairs.append({
                "variable_1": numeric_cols_aggregated_physical_test[i],
                "variable_2": numeric_cols_aggregated_physical_test[j],
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

cols_to_drop_corr_pairs = [
    # eliminar todas las versiones "best"
    "cmj__concentric_impulse_ns_best",
    "cmj__peak_power_bm_w_kg_best",
    "cmj__concentric_mean_power_bm_w_kg_best",
    "cmj__rsi_modified_imp_mom_best",
    "cmj__jump_height_imp_mom_cm_best",
    "cmj__concentric_peak_force_bm_n_kg_best",
    "cmj__eccentric_braking_impulse_ns_best",
    "cmj__concentric_duration_s_best",
    "slj__rsi_modified_best",
    "slj__jump_height_imp_mom_cm_best",
    "slj__peak_power_bm_w_kg_best",

    # pesos corporales (eliminar ambos)
    "cmj__weight_kg",
    "slj__weight_kg",

    # derivadas altamente redundantes (dejamos solo una representación)
    # criterio: conservar impulso y eliminar redundantes de potencia/RSI duplicadas
    "cmj__concentric_mean_power_bm_w_kg_best",  # ya eliminado por best pero por consistencia
]

# DROP
df_reduced = df_cleaned.drop(*cols_to_drop_corr_pairs)

print(f"Columnas eliminadas: {len(cols_to_drop_corr_pairs)}")
print(f"Columnas restantes: {len(df_reduced.columns)}")

# COMMAND ----------

df_reduced.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.agregado_physical_test_cleaned")

# COMMAND ----------

# MAGIC %md
# MAGIC # Agregado antropometrica x physical tests

# COMMAND ----------

df1=spark.table("dev_gold.pitchvision.cmj_agregado_por_jugador")
display(df1)

# COMMAND ----------

# %pip install matplotlib_venn
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


ids_anthro = set(
    [row[ID_COL] for row in df_anthropometry.select(ID_COL).distinct().collect()]
)

ids_tests = set(
    [row[ID_COL] for row in df_agregated_physical_tests.select(ID_COL).distinct().collect()]
)

# ── CONJUNTOS ───────────────────────────────────────────────────────

only_anthro = ids_anthro - ids_tests
only_tests  = ids_tests - ids_anthro
both        = ids_anthro & ids_tests

# ── DIAGRAMA DE VENN ────────────────────────────────────────────────

plt.figure(figsize=(6, 6))

venn2(
    subsets=(
        len(only_anthro),
        len(only_tests),
        len(both)
    ),
    set_labels=("Anthropometry", "Physical Tests")
)

plt.title(f"Solape de IDs | Total únicos: {len(ids_anthro | ids_tests)}")
plt.show()

# ── RESUMEN ─────────────────────────────────────────────────────────

print("Solo anthropometry:", len(only_anthro))
print("Solo physical tests:", len(only_tests))
print("En ambos:", len(both))

# COMMAND ----------

display(df_anthropometry)
display(df_agregated_physical_tests)

# COMMAND ----------

df = df_anthropometry.join(
    df_agregated_physical_tests,
    on=ID_COL,
    how="inner"
)

print("Filas en antropometry:", df_anthropometry.count())
print(f"Jugadores únicos antropometry:")
print(f"{df_anthropometry.select(ID_COL).distinct().count():,}\n\n")
print("Filas en tests:", df_agregated_physical_tests.count())
print(f"Jugadores únicos tests:")
print(f"{df_agregated_physical_tests.select(ID_COL).distinct().count():,}\n\n")
print("Filas en intersección:", df.count())
print(f"Jugadores únicos join:")
print(f"{df.select(ID_COL).distinct().count():,}")

# COMMAND ----------

print(f"\n{'=' * 60}")
print(f"TABLA: agregada")
print('=' * 60)

numeric_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, NumericType)
    and f.name not in EXCLUDE_ALWAYS
]
print(f"  Columnas numéricas: {len(numeric_cols)}")
print(f"  {numeric_cols}")

print(f"\n Jugadores únicos totales:")
print(f"  {df.select(ID_COL).distinct().count():,}")
print(f"\n  Jugadores únicos por temporada:")
numeric_cols = (
    df
    .groupBy(SEASON_COL)
    .agg(F.countDistinct(ID_COL).alias("jugadores_unicos"))
    .orderBy(SEASON_COL)
)
for row in numeric_cols.collect():
    print(f"    {row[SEASON_COL]}: {row['jugadores_unicos']:,} jugadores")

print(f"\n  Total filas: {df.count():,}")
display(df.limit(5))

# COMMAND ----------

#  Nulos por columna

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]

nulls_df = df.agg(*null_exprs)
#da la vuelta a la tabla para que quede en horizontal y sea más legible
nulls_long = nulls_df.select(
    F.explode(
        F.array(*[
            F.struct(
                F.lit(c).alias("columna"),
                F.col(c).alias("nulos")
            )
            for c in df.columns
        ])
    ).alias("x")
).select("x.*")

num_rows = df.count()

nulls_report = (
    nulls_long
    .withColumn("filas_totales", F.lit(num_rows))
    .withColumn(
        "porcentaje_nulos",
        F.round((F.col("nulos") / F.col("filas_totales")) * 100, 2)
    )
    .orderBy(F.col("porcentaje_nulos").desc())
)

display(nulls_report)

# COMMAND ----------

ID_COL = "id_unico_deportista"
DATE_COL = "test_date"
AGE_COL = "age_years"

EXCLUDE_COLS = [
    ID_COL,
    AGE_COL,
    "season",
    "is_restricted"
]

# Aseguramos que test_date sea tipo date
df_imputed = df.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))

# Columnas numéricas a imputar, excepto edad e identificadores
numeric_cols_to_impute = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in EXCLUDE_COLS
]

print(f"Columnas numéricas a imputar con mediciones del mismo jugador: {len(numeric_cols_to_impute)}")
print(numeric_cols_to_impute)

# VENTANAS TEMPORALES POR JUGADOR
w_forward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(Window.unboundedPreceding, 0)
)

w_backward = (
    Window
    .partitionBy(ID_COL)
    .orderBy(DATE_COL)
    .rowsBetween(0, Window.unboundedFollowing)
)

# 1. IMPUTACIÓN ESPECIAL PARA age_years
#    Ajusta edad según diferencia entre test_date y la fecha
#    donde sí existe age_years.
prev_age_struct = F.last(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_forward)

next_age_struct = F.first(
    F.when(
        F.col(AGE_COL).isNotNull(),
        F.struct(
            F.col(DATE_COL).alias("ref_date"),
            F.col(AGE_COL).alias("ref_age")
        )
    ),
    ignorenulls=True
).over(w_backward)

df_imputed = (
    df_imputed
    .withColumn("_prev_age_ref", prev_age_struct)
    .withColumn("_next_age_ref", next_age_struct)
)

# Si usamos una medición anterior:
age_from_prev = (
    F.col("_prev_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_prev_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

# Si usamos una medición posterior:
age_from_next = (
    F.col("_next_age_ref.ref_age") +
    F.floor(
        F.months_between(
            F.col(DATE_COL),
            F.col("_next_age_ref.ref_date")
        ) / 12
    ).cast("int")
)

df_imputed = (
    df_imputed
    .withColumn(
        AGE_COL,
        F.coalesce(
            F.col(AGE_COL),
            age_from_prev,
            age_from_next
        ).cast("int")
    )
    .drop("_prev_age_ref", "_next_age_ref")
)

print("Imputación de age_years terminada.")

# 2. IMPUTACIÓN DEL RESTO DE VARIABLES NUMÉRICAS
for c in numeric_cols_to_impute:
    last_value = F.last(F.col(c), ignorenulls=True).over(w_forward)
    next_value = F.first(F.col(c), ignorenulls=True).over(w_backward)

    df_imputed = df_imputed.withColumn(
        c,
        F.coalesce(
            F.col(c),
            last_value,
            next_value
        )
    )

print("Imputación del resto de variables numéricas terminada.")

# 3. CACHEAR DF
df_imputed = df_imputed.cache()

print(f"Filas finales: {df_imputed.count():,}")

display(df_imputed.limit(20))

# COMMAND ----------

df_check = df_imputed

total_rows = df_check.count()

null_exprs = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df_check.columns
]

nulls_row = df_check.agg(*null_exprs)

null_report_after = (
    nulls_row
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("columna"),
                    F.col(c).alias("nulos_restantes")
                )
                for c in df_check.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .withColumn(
        "pct_nulos_restantes",
        F.round(F.col("nulos_restantes") / F.lit(total_rows) * 100, 2)
    )
    .orderBy(F.col("nulos_restantes").desc())
)

display(
    null_report_after.filter(F.col("nulos_restantes") > 0)
)

# COMMAND ----------

exclude_cols = [
    "id_unico_deportista",
    "test_date",
    "season",
    "is_restricted",
    "age_years"
]

numeric_cols_check = [
    field.name
    for field in df_imputed.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in exclude_cols
]

print(f"Columnas numéricas revisadas: {len(numeric_cols_check)}")

# 1. Una sola acción Spark: traer datos necesarios a pandas
pdf = (
    df_imputed
    .select(AGE_COL, *numeric_cols_check)
    .toPandas()
)

# 2. Crear grupos de edad en pandas
bins = [13, 16, 19, 40]
labels = ["13-15", "16-18", "19-39"]

pdf["age_group_imputation_check"] = pd.cut(
    pdf[AGE_COL],
    bins=bins,
    labels=labels,
    right=False
)

pdf["age_group_imputation_check"] = (
    pdf["age_group_imputation_check"]
    .astype("object")
    .fillna("SIN_GRUPO")
)

# 3. Calcular nulos por grupo y variable en pandas
rows = []

for age_group, group_df in pdf.groupby("age_group_imputation_check"):
    filas_grupo = len(group_df)

    null_counts = group_df[numeric_cols_check].isna().sum()

    for col, nulos in null_counts.items():
        if nulos > 0:
            rows.append({
                "columna": col,
                "age_group": age_group,
                "filas_grupo": filas_grupo,
                "nulos": int(nulos),
                "pct_nulos": round(nulos / filas_grupo * 100, 2)
            })

nulls_by_age_group_pd = (
    pd.DataFrame(rows)
    .sort_values(["nulos", "columna", "age_group"], ascending=[False, True, True])
)

display(nulls_by_age_group_pd)

# COMMAND ----------

pdf_age = (
    df_imputed
    .select(ID_COL, AGE_COL)
    .dropDuplicates([ID_COL])
    .select(AGE_COL)
    .toPandas()
)

plt.figure(figsize=(8,4))
plt.hist(pdf_age[AGE_COL], bins=20)
plt.title("Distribución de edad (distinct por ID)")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

pdf_age.describe()

# COMMAND ----------



# COMMAND ----------


