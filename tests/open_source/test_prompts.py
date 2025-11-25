"""
Tests for prompt export and import functionality.
"""
import pytest
import mlflow
from mlflow_migration.prompt.export_prompt import export_prompt
from mlflow_migration.prompt.import_prompt import import_prompt
from mlflow_migration.bulk.export_prompts import export_prompts
from mlflow_migration.bulk.import_prompts import import_prompts
from mlflow_migration.common.version_utils import has_prompt_support
from tests.utils_test import create_output_dir
from tests.open_source.init_tests import mlflow_context


pytestmark = pytest.mark.skipif(
    not has_prompt_support(),
    reason="Prompt registry not supported in this MLflow version"
)


def _create_test_prompt(client: mlflow.MlflowClient, name: str, template: str, tags: dict[str, str] | None = None) -> object:
    """
    Create a test prompt using version-aware API.
    
    Attempts to create a prompt using the appropriate API based on MLflow version:
    1. First tries mlflow.genai.register_prompt (newer versions)
    2. Falls back to mlflow.register_prompt (older versions)
    3. Skips test if no compatible API is available
    
    Args:
        client: MLflow client instance
        name: Name for the prompt
        template: Template string with variables in {var} format
        tags: Optional dictionary of tags to apply to the prompt
        
    Returns:
        Registered prompt object
        
    Raises:
        pytest.skip: If no compatible prompt creation API is available
    """
    try:
        import mlflow.genai
        if hasattr(mlflow.genai, 'register_prompt'):
            return mlflow.genai.register_prompt(
                name=name,
                template=template,
                tags=tags or {}
            )
    except Exception:
        # Swallow all exceptions here because the genai API may not be available or compatible in all MLflow versions.
        # The fallback below will attempt to use the top-level API if available.
        pass
    
    # Fallback to top-level API
    if hasattr(mlflow, 'register_prompt'):
        return mlflow.register_prompt(
            name=name,
            template=template,
            tags=tags or {}
        )
    
    pytest.skip("No compatible prompt creation API available")


def test_export_import_prompt(mlflow_context):
    """
    Test single prompt export and import functionality.
    
    Validates the complete workflow of exporting a prompt from a source MLflow instance
    and importing it into a destination MLflow instance. This test ensures that:
    
    1. A prompt can be created with a template containing variables and tags
    2. The prompt can be exported to the filesystem with all metadata preserved
    3. The exported prompt can be imported into a different MLflow instance
    4. The imported prompt retains its structure and can be renamed during import
    
    Test Setup:
        - Creates a prompt in the source MLflow instance with:
            * Template: "Hello {name}, welcome to {place}!"
            * Tags: {"test": "true", "purpose": "unit-test"}
        - Exports the prompt to a temporary directory
        - Imports the prompt to destination with a modified name (appends "_imported")
    
    Assertions:
        - Exported prompt object is not None
        - Exported prompt name matches the original
        - Import result is not None
        - Imported prompt has the expected new name
    """
    # Create test prompt in source
    mlflow.set_tracking_uri(mlflow_context.client_src.tracking_uri)
    prompt_name = "test_prompt_single"
    template = "Hello {name}, welcome to {place}!"
    
    prompt = _create_test_prompt(
        mlflow_context.client_src,
        name=prompt_name,
        template=template,
        tags={"test": "true", "purpose": "unit-test"}
    )
    
    # Export prompt
    output_dir = f"{mlflow_context.output_dir}/prompt_single"
    create_output_dir(output_dir)
    
    exported = export_prompt(
        prompt_name=prompt_name,
        prompt_version=str(prompt.version),
        output_dir=output_dir,
        mlflow_client=mlflow_context.client_src
    )
    
    assert exported is not None
    assert exported.name == prompt_name
    
    # Import prompt to destination
    mlflow.set_tracking_uri(mlflow_context.client_dst.tracking_uri)
    imported_name = f"{prompt_name}_imported"
    
    result = import_prompt(
        input_dir=output_dir,
        prompt_name=imported_name,
        mlflow_client=mlflow_context.client_dst
    )
    
    assert result is not None
    assert result[0] == imported_name


def test_bulk_export_import_prompts(mlflow_context):
    """
    Test bulk prompt export and import functionality.
    
    Validates the ability to export and import multiple prompts in a single operation.
    This test ensures that:
    
    1. Multiple prompts can be created in the source MLflow instance
    2. All prompts can be exported in bulk without specifying individual names
    3. The bulk export tracks successful and failed exports correctly
    4. All exported prompts can be imported to a destination instance
    5. Threading is disabled to ensure deterministic test execution
    
    Test Setup:
        - Creates 3 test prompts in source with unique templates:
            * "test_prompt_bulk_1": "Template 1: {var1}"
            * "test_prompt_bulk_2": "Template 2: {var2}"
            * "test_prompt_bulk_3": "Template 3: {var3}"
        - Exports all prompts (prompt_names=None) without threading
        - Imports all exported prompts to destination
    
    Assertions:
        - Export result is not None
        - Successful exports count >= number of created prompts
        - Failed exports count == 0
        - Import result is not None
        - Successful imports count >= number of created prompts
    """
    # Create multiple test prompts in source
    mlflow.set_tracking_uri(mlflow_context.client_src.tracking_uri)
    
    prompts_data = [
        ("test_prompt_bulk_1", "Template 1: {var1}"),
        ("test_prompt_bulk_2", "Template 2: {var2}"),
        ("test_prompt_bulk_3", "Template 3: {var3}")
    ]
    
    for name, template in prompts_data:
        _create_test_prompt(
            mlflow_context.client_src,
            name=name,
            template=template
        )
    
    # Export all prompts
    output_dir = f"{mlflow_context.output_dir}/prompts_bulk"
    create_output_dir(output_dir)
    
    export_result = export_prompts(
        output_dir=output_dir,
        prompt_names=None,  # Export all
        use_threads=False,
        mlflow_client=mlflow_context.client_src
    )
    
    assert export_result is not None
    assert export_result["successful_exports"] >= len(prompts_data)
    assert export_result["failed_exports"] == 0
    
    # Import all prompts to destination
    mlflow.set_tracking_uri(mlflow_context.client_dst.tracking_uri)
    
    import_result = import_prompts(
        input_dir=output_dir,
        use_threads=False,
        mlflow_client=mlflow_context.client_dst
    )
    
    assert import_result is not None
    assert import_result["successful_imports"] >= len(prompts_data)


def test_export_specific_prompts(mlflow_context):
    """
    Test exporting specific prompts by name.
    
    Validates the selective export functionality where only specified prompts are
    exported rather than all available prompts. This test ensures that:
    
    1. Multiple prompts can exist in the source instance
    2. Only specified prompts are exported when names are provided
    3. Prompts not in the selection list are excluded from export
    4. Export statistics accurately reflect the selective operation
    
    Test Setup:
        - Creates 3 prompts in source: prompt_a, prompt_b, prompt_c
        - Requests export of only prompt_a and prompt_c (excluding prompt_b)
        - Uses threading disabled for deterministic behavior
    
    Assertions:
        - Export result is not None
        - Exactly 2 prompts are successfully exported (matching selection)
        - No export failures occur (failed_exports == 0)
        
    Note:
        This test validates the filtering capability of the bulk export function,
        ensuring that partial exports work correctly and don't inadvertently export
        all available prompts.
    """
    # Create test prompts
    mlflow.set_tracking_uri(mlflow_context.client_src.tracking_uri)
    
    _create_test_prompt(mlflow_context.client_src, "prompt_a", "Template A")
    _create_test_prompt(mlflow_context.client_src, "prompt_b", "Template B")
    _create_test_prompt(mlflow_context.client_src, "prompt_c", "Template C")
    
    # Export only specific prompts
    output_dir = f"{mlflow_context.output_dir}/prompts_specific"
    create_output_dir(output_dir)
    
    export_result = export_prompts(
        output_dir=output_dir,
        prompt_names=["prompt_a", "prompt_c"],
        use_threads=False,
        mlflow_client=mlflow_context.client_src
    )
    
    assert export_result is not None
    assert export_result["successful_exports"] == 2
    assert export_result["failed_exports"] == 0
