"""
Unit tests for MLflow 3.x LoggedModel ID detection and resolution.

These tests verify the functionality added to handle MLflow 3.x LoggedModel IDs
in model version source fields during import operations.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from mlflow_migration.run.run_utils import update_mlmodel_fields
from mlflow_migration.model_version.import_model_version import (
    _is_logged_model_id,
    _resolve_logged_model_path,
    _extract_model_path as extract_model_path_version,
)
from mlflow_migration.model.import_model import (
    _is_logged_model_id as is_logged_model_id_model,
    _resolve_logged_model_path_for_run,
    _extract_model_path as extract_model_path_model,
)
from mlflow_migration.common import MlflowExportImportException


class TestIsLoggedModelId:
    """Tests for _is_logged_model_id() function."""

    def test_valid_logged_model_id(self):
        """Test detection of valid MLflow 3.x LoggedModel IDs."""
        valid_ids = [
            "m-0728b41ab24e491db0bcc28f5d4d9afd",
            "m-38acc6ecb8484e2b9268c3dc003c0832",
            "m-00000000000000000000000000000000",
            "m-ffffffffffffffffffffffffffffffff",
            "m-abcdef0123456789abcdef0123456789",
        ]
        for logged_model_id in valid_ids:
            assert (
                _is_logged_model_id(logged_model_id) is True
            ), f"Expected '{logged_model_id}' to be detected as LoggedModel ID"

    def test_invalid_logged_model_id_regular_paths(self):
        """Test that regular model paths are not detected as LoggedModel IDs."""
        regular_paths = [
            "model",
            "my_model",
            "models/sklearn",
            "artifact/model",
            "my_model/subdir",
        ]
        for path in regular_paths:
            assert (
                _is_logged_model_id(path) is False
            ), f"Expected '{path}' to NOT be detected as LoggedModel ID"

    def test_invalid_logged_model_id_wrong_format(self):
        """Test that incorrectly formatted IDs are not detected."""
        invalid_ids = [
            "m-short",  # Too short
            "m-0728b41ab24e491db0bcc28f5d4d9afdXX",  # Too long (34 hex chars)
            "M-0728b41ab24e491db0bcc28f5d4d9afd",  # Uppercase M
            "0728b41ab24e491db0bcc28f5d4d9afd",  # Missing m- prefix
            "n-0728b41ab24e491db0bcc28f5d4d9afd",  # Wrong prefix
            "m_0728b41ab24e491db0bcc28f5d4d9afd",  # Underscore instead of dash
            "m-0728b41ab24e491db0bcc28f5d4d9afg",  # Invalid hex char 'g'
            "",  # Empty string
        ]
        for invalid_id in invalid_ids:
            assert (
                _is_logged_model_id(invalid_id) is False
            ), f"Expected '{invalid_id}' to NOT be detected as LoggedModel ID"

    def test_both_modules_have_same_behavior(self):
        """Test that both module implementations behave identically."""
        test_cases = [
            "m-0728b41ab24e491db0bcc28f5d4d9afd",
            "model",
            "m-short",
            "",
        ]
        for path in test_cases:
            result_version = _is_logged_model_id(path)
            result_model = is_logged_model_id_model(path)
            assert (
                result_version == result_model
            ), f"Modules differ for '{path}': version={result_version}, model={result_model}"


class TestResolveLoggedModelPath:
    """Tests for _resolve_logged_model_path() function."""

    def test_resolve_via_log_model_history_dict_tags(self):
        """Test resolution using mlflow.log-model.history tag (dict format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "my_custom_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: my_custom_model\n")

            # Create run.json with log-model.history tag (dict format)
            log_model_history = json.dumps([{"artifact_path": "my_custom_model"}])
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test resolution
            result = _resolve_logged_model_path(
                tmpdir, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result == "my_custom_model"

    def test_resolve_via_log_model_history_list_tags(self):
        """Test resolution using mlflow.log-model.history tag (list format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "sklearn_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: sklearn_model\n")

            # Create run.json with log-model.history tag (list format)
            log_model_history = json.dumps([{"artifact_path": "sklearn_model"}])
            run_data = {
                "tags": [
                    {"key": "mlflow.log-model.history", "value": log_model_history},
                    {"key": "other_tag", "value": "other_value"},
                ]
            }
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test resolution
            result = _resolve_logged_model_path(
                tmpdir, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result == "sklearn_model"

    def test_resolve_via_directory_scan_fallback(self):
        """Test resolution via directory scan when log-model.history is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure without run.json
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "xgboost_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: xgboost_model\n")

            # Test resolution (should use directory scan)
            result = _resolve_logged_model_path(
                tmpdir, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result == "xgboost_model"

    def test_resolve_returns_none_when_no_model_found(self):
        """Test that None is returned when no model can be found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir)

            # Test resolution
            result = _resolve_logged_model_path(
                tmpdir, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result is None

    def test_resolve_with_nested_model_directory(self):
        """Test resolution with nested model directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "models", "production", "v1")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: models/production/v1\n")

            # Test resolution (should find nested path)
            result = _resolve_logged_model_path(
                tmpdir, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result == os.path.join("models", "production", "v1")


class TestResolveLoggedModelPathForRun:
    """Tests for _resolve_logged_model_path_for_run() function."""

    def test_resolve_for_run_with_run_id_directory(self):
        """Test resolution using run_id based directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123def456"

            # Create directory structure: {input_dir}/{run_id}/
            run_dir = os.path.join(tmpdir, run_id)
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: model\n")

            # Create run.json
            log_model_history = json.dumps([{"artifact_path": "model"}])
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test resolution
            result = _resolve_logged_model_path_for_run(
                tmpdir, run_id, "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result == "model"

    def test_resolve_for_run_returns_none_when_run_dir_missing(self):
        """Test that None is returned when run directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _resolve_logged_model_path_for_run(
                tmpdir, "nonexistent_run_id", "m-0728b41ab24e491db0bcc28f5d4d9afd"
            )
            assert result is None


class TestExtractModelPathVersion:
    """Tests for _extract_model_path() in import_model_version.py."""

    def test_extract_regular_model_path(self):
        """Test extraction of regular model paths."""
        source = "mlflow-artifacts:/0/abc123/artifacts/model"
        result = extract_model_path_version(source)
        assert result == "model"

    def test_extract_nested_model_path(self):
        """Test extraction of nested model paths."""
        source = "s3://bucket/mlflow/0/abc123/artifacts/models/sklearn/v1"
        result = extract_model_path_version(source)
        assert result == os.path.normpath("models/sklearn/v1")

    def test_extract_logged_model_id_with_resolution(self):
        """Test extraction and resolution of LoggedModel ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "my_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: my_model\n")

            # Create run.json
            log_model_history = json.dumps([{"artifact_path": "my_model"}])
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test extraction with LoggedModel ID
            source = "mlflow-artifacts:/0/abc123/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_version(source, input_dir=tmpdir)
            assert result == "my_model"

    def test_extract_logged_model_id_fallback_to_model(self):
        """Test that unresolvable LoggedModel ID falls back to 'model'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty directory structure (no model to resolve)
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir)

            source = "mlflow-artifacts:/0/abc123/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_version(source, input_dir=tmpdir)
            assert result == "model"

    def test_extract_logged_model_id_without_input_dir_falls_back(self):
        """Test that LoggedModel ID without input_dir falls back to 'model'."""
        source = (
            "mlflow-artifacts:/0/abc123/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
        )
        result = extract_model_path_version(source, input_dir=None)
        assert result == "model"

    def test_extract_raises_on_invalid_source(self):
        """Test that invalid source raises exception."""
        with pytest.raises(MlflowExportImportException):
            extract_model_path_version(None)
        with pytest.raises(MlflowExportImportException):
            extract_model_path_version("")

    def test_extract_basename_logged_model_id(self):
        """Test extraction of LoggedModel ID from basename (fallback path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "custom_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: custom_model\n")

            # Create run.json
            log_model_history = json.dumps([{"artifact_path": "custom_model"}])
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test with source that doesn't have /artifacts/ pattern (triggers basename fallback)
            source = "dbfs:/models/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_version(source, input_dir=tmpdir)
            assert result == "custom_model"


class TestExtractModelPathModel:
    """Tests for _extract_model_path() in import_model.py."""

    def test_extract_regular_model_path(self):
        """Test extraction of regular model paths."""
        run_id = "abc123def456"
        source = f"mlflow-artifacts:/0/{run_id}/artifacts/model"
        result = extract_model_path_model(source, run_id)
        assert result == "model"

    def test_extract_logged_model_id_with_resolution(self):
        """Test extraction and resolution of LoggedModel ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123def456"

            # Create directory structure
            run_dir = os.path.join(tmpdir, run_id)
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "xgb_model")
            os.makedirs(model_dir)

            # Create MLmodel file
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: xgb_model\n")

            # Create run.json
            log_model_history = json.dumps([{"artifact_path": "xgb_model"}])
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test extraction with LoggedModel ID
            source = f"mlflow-artifacts:/0/{run_id}/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_model(source, run_id, input_dir=tmpdir)
            assert result == "xgb_model"

    def test_extract_logged_model_id_fallback_to_model(self):
        """Test that unresolvable LoggedModel ID falls back to 'model'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123def456"

            # Create empty run directory (no model to resolve)
            run_dir = os.path.join(tmpdir, run_id)
            os.makedirs(run_dir)

            source = f"mlflow-artifacts:/0/{run_id}/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_model(source, run_id, input_dir=tmpdir)
            assert result == "model"

    def test_extract_raises_when_run_id_not_in_source(self):
        """Test that exception is raised when run_id is not found in source."""
        with pytest.raises(MlflowExportImportException):
            extract_model_path_model(
                "mlflow-artifacts:/0/different_run/artifacts/model", "abc123"
            )

    def test_extract_without_artifacts_pattern(self):
        """Test extraction when 'artifacts' pattern is missing."""
        run_id = "abc123def456"
        source = f"mlflow-artifacts:/0/{run_id}/model"
        result = extract_model_path_model(source, run_id)
        # When artifacts pattern is missing, model_path becomes empty string
        assert result == ""


class TestIntegrationScenarios:
    """Integration-like tests for realistic MLflow 3.x scenarios."""

    def test_mlflow_3x_xgboost_model_scenario(self):
        """Test realistic MLflow 3.x XGBoost model migration scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "38acc6ecb8484e2b9268c3dc003c0832"
            logged_model_id = "m-0728b41ab24e491db0bcc28f5d4d9afd"

            # Simulate exported directory structure from MLflow 3.x
            run_dir = os.path.join(tmpdir, run_id)
            artifacts_dir = os.path.join(run_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "model")  # Original artifact_path
            os.makedirs(model_dir)

            # Create MLmodel file (as exported)
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write(
                    """artifact_path: model
flavors:
  python_function:
    loader_module: mlflow.xgboost
  xgboost:
    xgb_version: 2.0.0
"""
                )

            # Create model.xgb file
            with open(os.path.join(model_dir, "model.xgb"), "w") as f:
                f.write("binary xgboost model data")

            # Create run.json with log-model.history (as exported by MLflow 3.x)
            log_model_history = json.dumps(
                [
                    {
                        "artifact_path": "model",
                        "flavors": {"xgboost": {}, "python_function": {}},
                        "logged_model_id": logged_model_id,
                    }
                ]
            )
            run_data = {
                "info": {"run_id": run_id},
                "tags": {"mlflow.log-model.history": log_model_history},
            }
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Simulate the source field from model version (MLflow 3.x format)
            source = f"mlflow-artifacts:/28/{run_id}/artifacts/{logged_model_id}"

            # Test extraction - should resolve to "model" (the original artifact_path)
            result = extract_model_path_model(source, run_id, input_dir=tmpdir)
            assert result == "model"

    def test_mlflow_2x_compatibility(self):
        """Test that MLflow 2.x style sources still work correctly."""
        run_id = "abc123def456"

        # MLflow 2.x style source (uses artifact_path directly, not LoggedModel ID)
        source = f"mlflow-artifacts:/0/{run_id}/artifacts/model"
        result = extract_model_path_model(source, run_id)
        assert result == "model"

        source = f"s3://my-bucket/mlflow/{run_id}/artifacts/sklearn_model"
        result = extract_model_path_model(source, run_id)
        assert result == "sklearn_model"

        source = f"dbfs:/databricks/mlflow/{run_id}/artifacts/models/production"
        result = extract_model_path_model(source, run_id)
        assert result == "models/production"

    def test_multiple_models_in_single_run(self):
        """Test scenario with multiple models logged in a single run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure for model version import
            run_dir = os.path.join(tmpdir, "run")
            artifacts_dir = os.path.join(run_dir, "artifacts")

            # Create two model directories
            model1_dir = os.path.join(artifacts_dir, "classifier")
            model2_dir = os.path.join(artifacts_dir, "regressor")
            os.makedirs(model1_dir)
            os.makedirs(model2_dir)

            # Create MLmodel files
            with open(os.path.join(model1_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: classifier\n")
            with open(os.path.join(model2_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: regressor\n")

            # Create run.json with multiple models in log-model.history
            log_model_history = json.dumps(
                [
                    {"artifact_path": "classifier", "logged_model_id": "m-aaaa"},
                    {"artifact_path": "regressor", "logged_model_id": "m-bbbb"},
                ]
            )
            run_data = {"tags": {"mlflow.log-model.history": log_model_history}}
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump(run_data, f)

            # Test - first model in history should be returned (classifier)
            source = "mlflow-artifacts:/0/run123/artifacts/m-0728b41ab24e491db0bcc28f5d4d9afd"
            result = extract_model_path_version(source, input_dir=tmpdir)
            assert result == "classifier"


class TestUpdateMlmodelFields:
    """Tests for update_mlmodel_fields() function in run_utils.py."""

    def test_removes_model_id_from_mlmodel(self):
        """Test that model_id is removed from MLmodel file during import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MLmodel file with model_id (as exported from MLflow 3.x)
            mlmodel_content = {
                "artifact_path": "model",
                "run_id": "old-run-id-from-source",
                "model_id": "m-0728b41ab24e491db0bcc28f5d4d9afd",
                "flavors": {"python_function": {"loader_module": "mlflow.sklearn"}},
            }

            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            mlmodel_path = os.path.join(model_dir, "MLmodel")

            import yaml

            with open(mlmodel_path, "w") as f:
                yaml.dump(mlmodel_content, f)

            # Track what gets written
            written_content = {}

            def mock_log_artifact(run_id, local_path, artifact_path):
                with open(local_path, "r") as f:
                    written_content["mlmodel"] = yaml.safe_load(f)

            # Create mock client
            mock_client = MagicMock()
            mock_client.download_artifacts.return_value = mlmodel_path
            mock_client.log_artifact.side_effect = mock_log_artifact

            new_run_id = "new-run-id-on-target"

            # Patch find_run_model_names to return our model path
            with patch(
                "mlflow_migration.run.run_utils.find_run_model_names",
                return_value=["model"],
            ):
                with patch(
                    "mlflow_migration.run.run_utils.mlflow_utils.download_artifacts",
                    return_value=mlmodel_path,
                ):
                    update_mlmodel_fields(mock_client, new_run_id)

            # Verify model_id was removed and run_id was updated
            assert "mlmodel" in written_content
            assert written_content["mlmodel"]["run_id"] == new_run_id
            assert "model_id" not in written_content["mlmodel"]
            assert written_content["mlmodel"]["artifact_path"] == "model"

    def test_handles_mlmodel_without_model_id(self):
        """Test that MLmodel files without model_id are handled correctly (MLflow 2.x)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MLmodel file without model_id (MLflow 2.x style)
            mlmodel_content = {
                "artifact_path": "model",
                "run_id": "old-run-id-from-source",
                "flavors": {"python_function": {"loader_module": "mlflow.sklearn"}},
            }

            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            mlmodel_path = os.path.join(model_dir, "MLmodel")

            import yaml

            with open(mlmodel_path, "w") as f:
                yaml.dump(mlmodel_content, f)

            # Track what gets written
            written_content = {}

            def mock_log_artifact(run_id, local_path, artifact_path):
                with open(local_path, "r") as f:
                    written_content["mlmodel"] = yaml.safe_load(f)

            # Create mock client
            mock_client = MagicMock()
            mock_client.download_artifacts.return_value = mlmodel_path
            mock_client.log_artifact.side_effect = mock_log_artifact

            new_run_id = "new-run-id-on-target"

            # Patch find_run_model_names to return our model path
            with patch(
                "mlflow_migration.run.run_utils.find_run_model_names",
                return_value=["model"],
            ):
                with patch(
                    "mlflow_migration.run.run_utils.mlflow_utils.download_artifacts",
                    return_value=mlmodel_path,
                ):
                    update_mlmodel_fields(mock_client, new_run_id)

            # Verify run_id was updated and no model_id was added
            assert "mlmodel" in written_content
            assert written_content["mlmodel"]["run_id"] == new_run_id
            assert "model_id" not in written_content["mlmodel"]

    def test_handles_no_models_in_run(self):
        """Test that runs without models are handled gracefully."""
        mock_client = MagicMock()

        with patch(
            "mlflow_migration.run.run_utils.find_run_model_names", return_value=[]
        ):
            # Should not raise any exceptions
            update_mlmodel_fields(mock_client, "some-run-id")

        # No artifacts should be logged
        mock_client.log_artifact.assert_not_called()
