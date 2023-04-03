# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Batch model inference

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

model_name = "customer_churn_random_forest"
model_uri = f"models:/{model_name}/None"
ids_table = "e2e_mlops_demo.bronze_customers_churn.churn_inference"

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

df_ids = spark.table(ids_table)

# COMMAND ----------

df_inference = fs.score_batch(model_uri, df_ids)

# COMMAND ----------

df_inference.select("CustomerID", "prediction").write.mode("overwrite").saveAsTable("e2e_mlops_demo.churn_prediction")
