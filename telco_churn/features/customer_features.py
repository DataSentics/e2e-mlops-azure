# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Customer feature table

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

feature_table_name = "e2e_mlops_demo.customer_features"

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

df = spark.table("e2e_mlops_demo.bronze_customers_churn.bronze_customers_churn").drop("ChurnLabel")

# COMMAND ----------

# MAGIC %sql
# MAGIC create database IF NOT EXISTS e2e_mlops_demo

# COMMAND ----------

fs.create_table(feature_table_name, primary_keys=["CustomerID"], schema=df.schema)

# COMMAND ----------

fs.write_table(feature_table_name, df, mode="merge")
