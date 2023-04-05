# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Batch model inference

# COMMAND ----------

import os

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

env = os.environ["ENV"]
feature_table_name = f"{env}.mlops_demo.customer_features"
prediction_table_name = f"{env}.mlops_demo.churn_prediction"

primary_key = "CustomerID"
label = "ChurnLabel"
entity_name = "customer"
model_name = "churn"
model_type = "random_forest"

full_model_name = f"{entity_name}_{model_name}_{model_type}"

# COMMAND ----------

model_uri = f"models:/{full_model_name}/Production"

ids_table = f"{env}.mlops_demo.silver_customer_ids"

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

df_ids = spark.table(ids_table)

# COMMAND ----------

df_inference = fs.score_batch(model_uri, df_ids)

# COMMAND ----------

df_inference.select("CustomerID", "prediction").write.mode("overwrite").saveAsTable(prediction_table_name)
