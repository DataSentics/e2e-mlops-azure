# Databricks notebook source
import os

from pyspark.sql import functions as F

# COMMAND ----------

cwd = os.getcwd()
env = os.environ["ENV"]

# COMMAND ----------

spark.sql(f"drop DATABASE if EXISTS {env}.mlops_demo cascade")

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {env}.mlops_demo")

# COMMAND ----------

df = (
    (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(f"file:{cwd}/demo/data/churn.csv")
    )
    .withColumn("ZipCode", F.col("ZipCode").cast("string").alias("ZipCode"))
    .withColumn("Count", F.col("Count").cast("double").alias("Count"))
    .withColumn(
        "TenureMonths", F.col("TenureMonths").cast("double").alias("TenureMonths")
    )
    .withColumn("ChurnValue", F.col("ChurnValue").cast("double").alias("ChurnValue"))
    .withColumn("ChurnScore", F.col("ChurnScore").cast("double").alias("ChurnScore"))
    .withColumn("CLTV", F.col("CLTV").cast("double").alias("CLTV"))
)

# COMMAND ----------

df.write.saveAsTable(f"{env}.mlops_demo.silver_customer")

# COMMAND ----------

df.select("CustomerID").write.saveAsTable(f"{env}.mlops_demo.silver_customer_ids")
