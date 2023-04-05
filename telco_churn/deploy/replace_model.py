# Databricks notebook source
from mlflow.tracking import MlflowClient

# COMMAND ----------

model_name = "customer_churn_random_forest"

# COMMAND ----------

mlflow_client = MlflowClient()

# COMMAND ----------

def move_model_to_production(model_name):
    registered_model = mlflow_client.get_registered_model(name=model_name)
    latest_versions_list = registered_model.latest_versions

    for model_version in latest_versions_list:
        if model_version.current_stage != "Archived":
            mlflow_client.transition_model_version_stage(
                name=model_name, version=model_version.version, stage="Production"
            )

# COMMAND ----------

move_model_to_production(model_name)
