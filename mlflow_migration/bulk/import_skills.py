"""
Imports multiple MLflow skills (with all their versions) from a directory.
"""

import os
import sys
import click
import mlflow
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mlflow_migration.common import utils, io_utils
from mlflow_migration.common.click_options import opt_input_dir
from mlflow_migration.common.version_utils import has_skill_support, log_version_info
from mlflow_migration.common.skill_utils import (
    SKILL_FILE,
    skill_not_supported_message,
)
from mlflow_migration.client.client_utils import create_mlflow_client
from mlflow_migration.skill.import_skill import import_skill

_logger = utils.getLogger(__name__)


def import_skills(
    input_dir: str,
    delete_skill: bool = False,
    use_threads: bool = False,
    mlflow_client: mlflow.MlflowClient | None = None,
) -> dict[str, Any]:
    """
    Import multiple skills from a directory.

    :param input_dir: Input directory containing exported skills.
    :param delete_skill: Delete existing skill before importing.
    :param use_threads: Use multithreading for import.
    :param mlflow_client: MLflow client.
    :return: Summary of import results.
    """

    if not has_skill_support():
        _logger.warning(skill_not_supported_message())
        return {"unsupported": True, "mlflow_version": mlflow.__version__}

    mlflow_client = mlflow_client or create_mlflow_client()
    log_version_info()

    try:
        skill_dirs = _find_skill_directories(input_dir)

        _logger.info(f"Found {len(skill_dirs)} skills to import")

        def import_single(skill_dir: dict[str, str]) -> tuple[str, list[int]] | None:
            _logger.info(f"Importing skill from: {skill_dir['name']}")
            return import_skill(
                input_dir=skill_dir["path"],
                delete_skill=delete_skill,
                mlflow_client=mlflow_client,
            )

        # Each skill is imported by one thread so the version order within a skill is preserved
        max_workers = utils.get_threads(use_threads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(import_single, skill_dirs))

        successful = [r for r in results if r is not None]
        summary = {
            "total_skills": len(skill_dirs),
            "successful_imports": len(successful),
            "failed_imports": len(results) - len(successful),
            "imported_versions": sum(len(versions) for _, versions in successful),
        }

        io_utils.write_export_file(input_dir, "import_summary.json", __file__, summary)

        _logger.info(f"Skill import completed: {summary}")
        return summary

    except Exception as e:
        _logger.error(f"Bulk skill import failed: {str(e)}")
        return {"error": str(e)}


def _find_skill_directories(input_dir: str) -> list[dict[str, str]]:
    """Find all skill directories (those containing a skill.json file) in the input directory."""
    if not os.path.exists(input_dir):
        raise Exception(f"Input directory does not exist: {input_dir}")

    skill_dirs: list[dict[str, str]] = []
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, SKILL_FILE)):
                skill_dirs.append({"name": item, "path": item_path})
                _logger.info(f"Found skill directory: {item}")
            else:
                _logger.debug(f"Skipping directory '{item}': no {SKILL_FILE} found")
    return skill_dirs


@click.command()
@opt_input_dir
@click.option(
    "--delete-skill",
    help="Delete existing skill before importing.",
    type=bool,
    default=False,
)
@click.option(
    "--use-threads", help="Use multithreading for import.", is_flag=True, default=False
)
def main(input_dir, delete_skill, use_threads):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")

    result = import_skills(
        input_dir=input_dir, delete_skill=delete_skill, use_threads=use_threads
    )

    # Check for failures
    if result is None:
        _logger.error("Skill import failed with unknown error")
        sys.exit(1)
    elif "unsupported" in result:
        _logger.error(skill_not_supported_message())
        sys.exit(1)
    elif "error" in result:
        _logger.error(f"Skill import failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
