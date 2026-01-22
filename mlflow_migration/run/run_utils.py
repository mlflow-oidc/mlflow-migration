import os
import tempfile
from mlflow_migration.common import mlflow_utils, io_utils
from mlflow_migration.common.find_artifacts import find_run_model_names


def get_model_name(artifact_path):
    idx = artifact_path.find("artifacts")
    idx += len("artifacts") + 1
    return artifact_path[idx:]


def update_mlmodel_fields(mlflow_client, run_id):
    """
    Updates MLmodel files in a run to fix fields that reference source server entities.

    :param mlflow_client: MLflow client for the destination server.
    :param run_id: Run ID on the destination server.

    This function performs two corrections:
    1. Updates the run_id field to reference the new run on the destination server.
    2. Removes the model_id field (if present) to prevent MLflow 3.x from attempting
       to fetch a LoggedModel entity that doesn't exist on the destination server.

    This is necessary because MLflow runs don't track their models directly, so we
    recursively search the run's artifact directory for all MLmodel files.
    """
    mlmodel_paths = find_run_model_names(mlflow_client, run_id)
    for model_path in mlmodel_paths:
        download_uri = f"runs:/{run_id}/{model_path}/MLmodel"
        local_path = mlflow_utils.download_artifacts(mlflow_client, download_uri)
        mlmodel = io_utils.read_file(local_path, "yaml")
        mlmodel["run_id"] = run_id
        # Remove model_id if present to prevent LoggedModel lookup on target server
        if "model_id" in mlmodel:
            del mlmodel["model_id"]
        with tempfile.TemporaryDirectory() as dir:
            output_path = os.path.join(dir, "MLmodel")
            io_utils.write_file(output_path, mlmodel, "yaml")
            if model_path == "MLmodel":
                model_path = ""
            mlflow_client.log_artifact(run_id, output_path, model_path)
