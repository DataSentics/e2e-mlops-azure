# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # `model_train`
# MAGIC 
# MAGIC Pipeline to execute model training. Params, metrics and model artifacts will be tracking to MLflow Tracking.
# MAGIC Optionally, the resulting model will be registered to MLflow Model Registry if provided.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup
from mlflow.tracking import MlflowClient
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from dataclasses import dataclass
from typing import List, Dict, Any
import pprint

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
import numpy as np

# COMMAND ----------

feature_table_name = "e2e_mlops_demo.customer_features"
primary_key = "CustomerID"
label = "ChurnLabel"

# COMMAND ----------

fs = FeatureStoreClient()
mlflow_client = MlflowClient()

# COMMAND ----------

df_ids = spark.table("e2e_mlops_demo.customer_labels")

# COMMAND ----------

training_set = fs.create_training_set(
    df_ids,
    [
        FeatureLookup(
            table_name=feature_table_name,
            lookup_key=[primary_key],
        )
    ],
    label=label,
)

# COMMAND ----------

df = training_set.load_df().toPandas()

# COMMAND ----------

df_replaced = df.replace([np.inf, -np.inf], np.nan)
df_replaced_dropna = df_replaced.dropna()

X = df_replaced_dropna.drop(label, axis=1)
y = df_replaced_dropna[label].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.25,
    stratify=y,
)

# COMMAND ----------

model_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "auto",
    "random_state": 42,
}

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric_transformer",
            SimpleImputer(strategy="median"),
            make_column_selector(dtype_exclude="object"),
        ),
        (
            "categorical_transformer",
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include="object"),
        ),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)

rf_classifier = RandomForestClassifier(**model_params)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier),
    ]
)

# COMMAND ----------

model_name = "customer_churn_random_forest"
run_name = f"dev_{model_name}"

mlflow.sklearn.autolog(log_input_examples=True, silent=True)
with mlflow.start_run(run_name=run_name) as mlflow_run:
    model = pipeline.fit(X_train, y_train)

    fs.log_model(
        model,
        model_name,
        flavor=mlflow.sklearn,
        training_set=training_set,
        input_example=X_train[:100],
        signature=infer_signature(X_train, y_train),
    )

    mlflow.register_model(f"runs:/{mlflow_run.info.run_id}/{model_name}", name=model_name)

# COMMAND ----------

registered_model = mlflow_client.get_registered_model(name=model_name)
latest_versions_list = registered_model.latest_versions

for model_version in latest_versions_list:
    if model_version.current_stage != "Archived":
        mlflow_client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )
