# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Customer feature table

# COMMAND ----------

import os

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

fs = FeatureStoreClient()

env = os.environ["ENV"]
feature_table_name = f"{env}.mlops_demo.customer_features"

# COMMAND ----------

df = spark.table(f"{env}.mlops_demo.silver_customer").drop("ChurnLabel")

# COMMAND ----------

fs.create_table(feature_table_name, primary_keys=["CustomerID"], schema=df.schema)

# COMMAND ----------

fs.write_table(feature_table_name, df, mode="merge")
