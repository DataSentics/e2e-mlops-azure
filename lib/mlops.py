import json
import os

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from mlflow.tracking import MlflowClient


dbutils = DBUtils(SparkSession.getActiveSession())
mlflow_client = MlflowClient()


def get_env():
    return os.environ.get("ENV", "none")


def get_commit_hash():
    return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())["extraContext"]


def get_commit_hash_id():
    return get_commit_hash()["mlflowGitCommit"][-4:]


def get_experiment_name(entity_name, model_name, model_type):
    env = get_env()
    commit_hash_id = get_commit_hash_id()
    return f"{env}_{entity_name}_{model_name}_{model_type}_{commit_hash_id}"
