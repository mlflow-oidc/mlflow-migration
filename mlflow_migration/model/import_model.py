"""
Imports a registered model, its versions and the version's run.
"""

import os
import click
import re
import json

import mlflow
from mlflow.exceptions import RestException

from mlflow_migration.common.click_options import (
    opt_input_dir,
    opt_model,
    opt_experiment_name,
    opt_import_permissions,
    opt_delete_model,
    opt_import_source_tags,
    opt_verbose,
)
from mlflow_migration.common import utils, io_utils, model_utils
from mlflow_migration.common import filesystem as _fs
from mlflow_migration.common.mlflow_utils import MlflowTrackingUriTweak
from mlflow_migration.common.source_tags import (
    set_source_tags_for_field,
    fmt_timestamps,
)
from mlflow_migration.common import MlflowExportImportException
from mlflow_migration.client.client_utils import create_mlflow_client, create_dbx_client
from mlflow_migration.run.import_run import import_run
from mlflow_migration.bulk import rename_utils

from mlflow_migration.model_version.import_model_version import _import_model_version


_logger = utils.getLogger(__name__)


def _set_source_tags_for_field(dct, tags):
    set_source_tags_for_field(dct, tags)
    fmt_timestamps("creation_timestamp", dct, tags)
    fmt_timestamps("last_updated_timestamp", dct, tags)


def _is_logged_model_id(path):
    """
    Detect MLflow 3.x LoggedModel ID (format: m-<32 hex chars>).

    :param path: The path component to check
    :return: True if the path matches the LoggedModel ID pattern
    """
    return bool(re.match(r"^m-[a-f0-9]{32}$", path))


def _resolve_logged_model_path_for_run(input_dir, run_id, logged_model_id):
    """
    Resolve LoggedModel ID to actual artifact path using run.json metadata.

    This function is adapted for the model import context where runs are stored
    in directories named by their run_id within the input_dir.

    :param input_dir: The input directory containing the exported model
    :param run_id: The run ID containing the model version
    :param logged_model_id: The LoggedModel ID to resolve (e.g., m-0728b41ab24e491db0bcc28f5d4d9afd)
    :return: The resolved artifact path or None if not found
    """
    run_dir = os.path.join(input_dir, run_id)
    run_json_path = os.path.join(run_dir, "run.json")
    artifacts_dir = os.path.join(run_dir, "artifacts")

    # Strategy 1: Parse mlflow.log-model.history from run.json
    if os.path.exists(run_json_path):
        try:
            run_data = io_utils.read_file_mlflow(run_json_path)
            tags = run_data.get("tags", {})
            # Tags can be either a dict or a list of {key, value} dicts
            if isinstance(tags, list):
                tags = {t["key"]: t["value"] for t in tags}
            log_model_history = tags.get("mlflow.log-model.history")
            if log_model_history:
                history = json.loads(log_model_history)
                for entry in history:
                    artifact_path = entry.get("artifact_path")
                    if artifact_path:
                        mlmodel_path = os.path.join(
                            artifacts_dir, artifact_path, "MLmodel"
                        )
                        if os.path.exists(mlmodel_path):
                            _logger.info(
                                f"Resolved LoggedModel '{logged_model_id}' to '{artifact_path}' via log-model.history"
                            )
                            return artifact_path
        except Exception as e:
            _logger.warning(f"Failed to parse log-model.history: {e}")

    # Strategy 2: Scan for MLmodel files in artifacts directory
    if os.path.exists(artifacts_dir):
        for root, _, files in os.walk(artifacts_dir):
            if "MLmodel" in files:
                rel_path = os.path.relpath(root, artifacts_dir)
                if rel_path != ".":
                    _logger.info(
                        f"Resolved LoggedModel '{logged_model_id}' to '{rel_path}' via directory scan"
                    )
                    return rel_path

    return None


def import_model(
    model_name,
    experiment_name,
    input_dir,
    delete_model=False,
    import_permissions=False,
    import_source_tags=False,
    verbose=False,
    await_creation_for=None,
    mlflow_client=None,
):
    importer = ModelImporter(
        import_source_tags=import_source_tags,
        import_permissions=import_permissions,
        await_creation_for=await_creation_for,
        mlflow_client=mlflow_client,
    )
    return importer.import_model(
        model_name=model_name,
        input_dir=input_dir,
        experiment_name=experiment_name,
        delete_model=delete_model,
        verbose=verbose,
    )


class BaseModelImporter:
    """Base class of ModelImporter subclasses."""

    def __init__(
        self,
        mlflow_client=None,
        import_permissions=False,
        import_source_tags=False,
        await_creation_for=None,
    ):
        """
        :param mlflow_client: MLflow client or if None create default client.
        :param import_permissions: Import Databricks permissions.
        :param import_source_tags: Import source information for MLFlow objects and create tags in destination object.
        :param await_creation_for: Seconds to wait for model version crreation.
        """
        self.mlflow_client = mlflow_client or create_mlflow_client()
        self.dbx_client = create_dbx_client(self.mlflow_client)
        self.import_source_tags = import_source_tags
        self.import_permissions = import_permissions
        self.import_source_tags = import_source_tags
        self.await_creation_for = await_creation_for

    def _import_model(self, model_name, input_dir, delete_model=False):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param delete_model: Delete current model before importing versions.
        :param verbose: Verbose.
        :return: Model import manifest.
        """
        path = os.path.join(input_dir, "model.json")
        model_dct = io_utils.read_file_mlflow(path)["registered_model"]

        _logger.info("Model to import:")
        _logger.info(f"  Name: {model_dct['name']}")
        _logger.info(f"  Description: {model_dct.get('description','')}")
        _logger.info(f"  Tags: {model_dct.get('tags','')}")
        _logger.info(f"  {len(model_dct.get('versions',[]))} versions")
        _logger.info(f"  path: {path}")

        if not model_name:
            model_name = model_dct["name"]
        if delete_model:
            model_utils.delete_model(self.mlflow_client, model_name)

        created_model = model_utils.create_model(
            self.mlflow_client, model_name, model_dct, True
        )
        perms = model_dct.get("permissions")
        if created_model and self.import_permissions and perms:
            if model_utils.model_names_same_registry(model_dct["name"], model_name):
                model_utils.update_model_permissions(
                    self.mlflow_client, self.dbx_client, model_name, perms
                )
            else:
                _logger.warning(
                    f'Cannot import permissions since models \'{model_dct["name"]}\' and \'{model_name}\' must be either both Unity Catalog model names or both Workspace model names.'
                )

        return model_dct


class ModelImporter(BaseModelImporter):
    """Low-level 'single' model importer."""

    def __init__(
        self,
        mlflow_client=None,
        import_permissions=False,
        import_source_tags=False,
        await_creation_for=None,
    ):
        super().__init__(
            mlflow_client=mlflow_client,
            import_permissions=import_permissions,
            import_source_tags=import_source_tags,
            await_creation_for=await_creation_for,
        )

    def import_model(
        self, model_name, input_dir, experiment_name, delete_model=False, verbose=False
    ):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param experiment_name: The name of the experiment.
        :param delete_model: Delete current model before importing versions.
        :param import_source_tags: Import source information for registered model and its versions ad tags in destination object.
        :param verbose: Verbose.
        :return: Model import manifest.
        """
        model_dct = self._import_model(model_name, input_dir, delete_model)
        _logger.info("Importing versions:")
        for vr in model_dct.get("versions", []):
            try:
                run_id = self._import_run(input_dir, experiment_name, vr)
                if run_id:
                    self.import_version(model_name, vr, run_id, input_dir=input_dir)
            except RestException as e:
                msg = {
                    "model": model_name,
                    "version": vr["version"],
                    "src_run_id": vr["run_id"],
                    "experiment": experiment_name,
                    "RestException": str(e),
                }
                _logger.error(f"Failed to import model version: {msg}")
                import traceback

                traceback.print_exc()
        if verbose:
            model_utils.dump_model_versions(self.mlflow_client, model_name)

    def _import_run(self, input_dir, experiment_name, vr):
        run_id = vr["run_id"]
        source = vr["source"]
        current_stage = vr["current_stage"]
        run_artifact_uri = vr.get("_run_artifact_uri", None)
        run_dir = _fs.mk_local_path(os.path.join(input_dir, run_id))
        if not os.path.exists(run_dir):
            msg = {
                "model": vr["name"],
                "version": vr["version"],
                "experiment": experiment_name,
                "run_id": run_id,
            }
            _logger.warning(
                f"Cannot import model version because its run folder '{run_dir}' does not exist: {msg}"
            )
            return None
        _logger.info(f"  Version {vr['version']}:")
        _logger.info(f"    current_stage: {current_stage}:")
        _logger.info("    Source run - run to import:")
        _logger.info(f"      run_id:           {run_id}")
        _logger.info(f"      run_artifact_uri: {run_artifact_uri}")
        _logger.info(f"      source:           {source}")
        model_artifact = _extract_model_path(source, run_id, input_dir=input_dir)
        _logger.info(f"      model_artifact:   {model_artifact}")

        dst_run, _ = import_run(
            input_dir=run_dir,
            experiment_name=experiment_name,
            import_source_tags=self.import_source_tags,
            mlflow_client=self.mlflow_client,
        )
        dst_run_id = dst_run.info.run_id
        run = self.mlflow_client.get_run(dst_run_id)
        _logger.info("    Destination run - imported run:")
        _logger.info(f"      run_id: {dst_run_id}")
        _logger.info(f"      run_artifact_uri: {run.info.artifact_uri}")
        source = _path_join(run.info.artifact_uri, model_artifact)
        _logger.info(f"      source:           {source}")
        return dst_run_id

    def import_version(self, model_name, src_vr, dst_run_id, input_dir=None):
        dst_run = self.mlflow_client.get_run(dst_run_id)
        model_path = _extract_model_path(
            src_vr["source"], src_vr["run_id"], input_dir=input_dir
        )
        dst_source = f"{dst_run.info.artifact_uri}/{model_path}"
        return _import_model_version(
            mlflow_client=self.mlflow_client,
            model_name=model_name,
            src_vr=src_vr,
            dst_run_id=dst_run_id,
            dst_source=dst_source,
            import_source_tags=self.import_source_tags,
        )


class BulkModelImporter(BaseModelImporter):
    """Bulk model importer."""

    def __init__(
        self,
        run_info_map,
        import_permissions=False,
        import_source_tags=False,
        experiment_renames=None,
        await_creation_for=None,
        mlflow_client=None,
    ):
        super().__init__(
            mlflow_client=mlflow_client,
            import_permissions=import_permissions,
            import_source_tags=import_source_tags,
            await_creation_for=await_creation_for,
        )
        self.run_info_map = run_info_map
        self.experiment_renames = experiment_renames

    def import_model(self, model_name, input_dir, delete_model=False, verbose=False):
        """
        :param model_name: Model name.
        :param input_dir: Input directory.
        :param delete_model: Delete current model before importing versions.
        :param verbose: Verbose.
        :return: Model import manifest.
        """
        model_dct = self._import_model(model_name, input_dir, delete_model)
        _logger.info(f"Importing {len(model_dct['versions'])} versions:")
        for vr in model_dct["versions"]:
            src_run_id = vr["run_id"]
            dst_run_info = self.run_info_map.get(src_run_id, None)
            if not dst_run_info:
                msg = {
                    "model": model_name,
                    "version": vr["version"],
                    "stage": vr["current_stage"],
                    "run_id": src_run_id,
                }
                _logger.error(
                    f"Cannot import model version {msg} since the source run_id was probably deleted."
                )
            else:
                dst_run_id = dst_run_info.run_id
                exp_name = rename_utils.rename(
                    vr["_experiment_name"], self.experiment_renames, "experiment"
                )
                try:
                    with MlflowTrackingUriTweak(self.mlflow_client):
                        mlflow.set_experiment(exp_name)
                    self.import_version(model_name, vr, dst_run_id, input_dir=input_dir)
                except RestException as e:
                    msg = {
                        "model": model_name,
                        "version": vr.get("version", []),
                        "experiment": exp_name,
                        "run_id": dst_run_id,
                        "exception": str(e),
                    }
                    _logger.error(f"Failed to import model version: {msg}")
        if verbose:
            model_utils.dump_model_versions(self.mlflow_client, model_name)

    def import_version(self, model_name, src_vr, dst_run_id, input_dir=None):
        src_run_id = src_vr["run_id"]
        model_path = _extract_model_path(
            src_vr["source"], src_run_id, input_dir=input_dir
        )  # get path to model artifact
        dst_artifact_uri = self.run_info_map[src_run_id].artifact_uri
        dst_source = f"{dst_artifact_uri}/{model_path}"
        return _import_model_version(
            mlflow_client=self.mlflow_client,
            model_name=model_name,
            src_vr=src_vr,
            dst_run_id=dst_run_id,
            dst_source=dst_source,
            import_source_tags=self.import_source_tags,
        )


def _extract_model_path(source, run_id, input_dir=None):
    """
    Extract relative path to model artifact from version source field.

    Supports MLflow 3.x LoggedModel IDs (format: m-<32 hex chars>) which are resolved
    to actual artifact paths using run metadata.

    :param source: 'source' field of registered model version
    :param run_id: Run ID in the 'source' field
    :param input_dir: Input directory containing the exported model (optional, used for LoggedModel resolution)
    :return: relative path to the model artifact
    """
    idx = source.find(run_id)
    if idx == -1:
        raise MlflowExportImportException(
            f"Cannot find run ID '{run_id}' in registered model version source field '{source}'",
            http_status_code=404,
        )

    # Use "/artifacts/" pattern to avoid matching "artifacts" in scheme names like "mlflow-artifacts:"
    pattern = "/artifacts/"
    idx = source.find(pattern)
    if idx == -1:  # Sometimes there is no 'artifacts' directory in the source
        model_path = ""
    else:
        model_path = source[idx + len(pattern) :]

    # Check for MLflow 3.x LoggedModel ID (format: m-<32 hex chars>)
    if model_path and _is_logged_model_id(model_path):
        _logger.info(f"Detected MLflow 3.x LoggedModel ID: {model_path}")
        if input_dir:
            resolved = _resolve_logged_model_path_for_run(input_dir, run_id, model_path)
            if resolved:
                return resolved
        _logger.warning(
            f"Could not resolve LoggedModel ID '{model_path}', falling back to 'model'"
        )
        return "model"

    return model_path


def _path_join(x, y):
    """Account for DOS backslash"""
    path = os.path.join(x, y)
    if path.startswith("dbfs:"):
        path = path.replace("\\", "/")
    return path


@click.command()
@opt_input_dir
@opt_model
@opt_experiment_name
@opt_delete_model
@opt_import_permissions
@opt_import_source_tags
@click.option(
    "--await-creation-for",
    help="Await creation for specified seconds.",
    type=int,
    default=None,
    show_default=True,
)
@opt_verbose
def main(
    input_dir,
    model,
    experiment_name,
    delete_model,
    import_permissions,
    import_source_tags,
    await_creation_for,
    verbose,
):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")
    import_model(
        model_name=model,
        experiment_name=experiment_name,
        input_dir=input_dir,
        delete_model=delete_model,
        import_permissions=import_permissions,
        import_source_tags=import_source_tags,
        await_creation_for=await_creation_for,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
