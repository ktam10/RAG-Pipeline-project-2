import mlflow
import os
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://3.19.56.68:5000")
REGISTERED_MODEL_NAME = "rag-embedding-model"
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Create the registered model if it doesn't exist
try:
    client.create_registered_model(
        name=REGISTERED_MODEL_NAME,
        description="Embedding model used for RAG pipeline chunk indexing"
    )
    print(f"Registered model '{REGISTERED_MODEL_NAME}' created.")
except Exception:
    print(f"Registered model '{REGISTERED_MODEL_NAME}' already exists. Continuing.")

# Log a run with the embedding model name as a tag
mlflow.set_experiment("rag-embedding-experiments")


with mlflow.start_run(run_name="register-embedding-v1") as run:

    # Log the embedding model name as a tag
    mlflow.set_tag("embedding_model_name", EMBEDDING_MODEL_NAME)
    mlflow.log_param("embedding_model",    EMBEDDING_MODEL_NAME)
    mlflow.log_param("version",            "1")
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

# Create Version 1 in the registry pointing to this run
version = client.create_model_version(
    name=REGISTERED_MODEL_NAME,
    source=f"runs:/{run_id}",
    run_id=run_id,
    tags={"embedding_model_name": EMBEDDING_MODEL_NAME}
)
print(f"Version {version.version} created.")

# Promote to Production
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=version.version,
    stage="Production"
)
print(f"Version {version.version} promoted to Production.")
print(f"Embedding model '{EMBEDDING_MODEL_NAME}' is now registered and in Production.")