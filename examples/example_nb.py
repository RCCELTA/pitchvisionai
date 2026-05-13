# Databricks notebook source
# MAGIC %md
# MAGIC ## Example notebook
# MAGIC - Read tables
# MAGIC - Common transformations
# MAGIC - Join 2 tables
# MAGIC - Write the results

# COMMAND ----------

from pyspark.sql import functions as sf

# COMMAND ----------

# Read the tables
gps = spark.table("dev_silver.pitchvision.v_gps_data")
injury = spark.table("dev_silver.pitchvision.v_injury")
cmj = spark.table("dev_silver.pitchvision.v_physical_test_cmj")
slj = spark.table("dev_silver.pitchvision.v_physical_test_slj")
isoprone = spark.table("dev_silver.pitchvision.v_physical_test_isoprone")
nordic = spark.table("dev_silver.pitchvision.v_physical_test_nordic")
antro = spark.table("dev_silver.pitchvision.v_anthropometry")

# COMMAND ----------

# Rows count of every table
print(f"gps rows:      {gps.count():,}")
print(f"injury rows:   {injury.count():,}")
print(f"cmj rows:      {cmj.count():,}")
print(f"slj rows:      {slj.count():,}")
print(f"isoprone rows: {isoprone.count():,}")
print(f"nordic rows:   {nordic.count():,}")
print(f"antro rows:    {antro.count():,}")

# COMMAND ----------

gps.printSchema()
gps.display()

# COMMAND ----------

gps_subset = gps \
    .select("id_unico_deportista", "activity_date", "total_distance", "max_velocity")\
    .groupBy("id_unico_deportista") \
    .agg(
        sf.count("*").alias("activity_count"),
        sf.sum("total_distance").alias("total_distance"),
        sf.max("max_velocity").alias("max_velocity")
    )

injury_per_player = injury \
    .groupBy("id_unico_deportista") \
    .agg(
        sf.count("*").alias("injuries"),
        sf.max("injury_date").alias("last_injury_date"),
    )

joined = gps_subset\
    .join(injury_per_player, "id_unico_deportista", "left")\
    .fillna(0, subset=["injuries"])

joined.write.mode("overwrite").saveAsTable("dev_gold.pitchvision.example")
