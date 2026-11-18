"""
Exports multiple MLflow skills (with all their versions) to a directory.
"""

import os
import sys
import click
import mlflow
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mlflow_migration.common import utils, io_utils
from mlflow_migration.common.click_options import opt_output_dir
from mlflow_migration.common.version_utils import has_skill_support, log_version_info
from mlflow_migration.common.skill_utils import (
    format_skill_identity,
    parse_skill_identity,
    search_all_skills,
    skill_dir_name,
    skill_not_supported_message,
)
from mlflow_migration.client.client_utils import create_mlflow_client
from mlflow_migration.skill.export_skill import export_skill

_logger = utils.getLogger(__name__)


def export_skills(
    output_dir: str,
    skill_names: list[str] | None = None,
    alias: str | None = None,
    use_threads: bool = False,
    mlflow_client: mlflow.MlflowClient | None = None,
) -> dict[str, Any]:
    """
    Export multiple skills to a directory.

    :param output_dir: Output directory.
    :param skill_names: List of skill identities ('name' or '@organization/name') to export.
        If None, exports all skills.
    :param alias: Export only the version this alias (e.g. 'champion') points to.
        Skills without this alias are skipped. If None, all versions are exported.
    :param use_threads: Use multithreading for export.
    :param mlflow_client: MLflow client.
    :return: Summary of export results.
    """

    if not has_skill_support():
        _logger.warning(skill_not_supported_message())
        return {"unsupported": True, "mlflow_version": mlflow.__version__}

    mlflow_client = mlflow_client or create_mlflow_client()
    log_version_info()

    try:
        if skill_names:
            identities = [parse_skill_identity(name) for name in skill_names if name]
        else:
            identities = [
                (skill.organization or "", skill.name)
                for skill in search_all_skills(mlflow_client)
            ]

        skipped = []
        if alias:
            skipped = [
                identity
                for identity in identities
                if not _has_alias(mlflow_client, identity, alias)
            ]
            for organization, name in skipped:
                _logger.info(
                    f"Skipping skill '{format_skill_identity(name, organization)}': no alias '{alias}'"
                )
            identities = [i for i in identities if i not in skipped]

        _logger.info(f"Found {len(identities)} skills to export")

        os.makedirs(output_dir, exist_ok=True)

        def export_single(identity: tuple[str, str]) -> object | None:
            organization, name = identity
            return export_skill(
                skill_name=name,
                organization=organization,
                alias=alias,
                output_dir=os.path.join(output_dir, skill_dir_name(name, organization)),
                mlflow_client=mlflow_client,
            )

        max_workers = utils.get_threads(use_threads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(export_single, identities))

        failed = [
            format_skill_identity(name, organization)
            for (organization, name), result in zip(identities, results)
            if result is None
        ]
        summary = {
            "total_skills": len(identities),
            "successful_exports": len(identities) - len(failed),
            "failed_exports": len(failed),
            "failed_skills": failed,
        }
        if alias:
            summary["alias"] = alias
            summary["skipped_skills"] = [
                format_skill_identity(name, organization)
                for organization, name in skipped
            ]

        io_utils.write_export_file(output_dir, "skills_summary.json", __file__, summary)

        _logger.info(f"Skill export completed: {summary}")
        return summary

    except Exception as e:
        _logger.error(f"Bulk skill export failed: {str(e)}")
        return {"error": str(e)}


def _has_alias(
    mlflow_client: mlflow.MlflowClient, identity: tuple[str, str], alias: str
) -> bool:
    """
    Whether a skill has the alias. A skill that cannot be read is not skipped so that
    its export fails and is reported.
    """
    organization, name = identity
    try:
        skill = mlflow_client.get_skill(name=name, organization=organization)
    except Exception:
        return True
    return alias in (skill.aliases or {})


@click.command()
@opt_output_dir
@click.option(
    "--skills",
    help="Skill names: 'all' for all skills, comma-delimited list (e.g., 'skill1,@my-org/skill2'), \
or file path ending with '.txt' containing skill names (one per line). \
A skill scoped to an organization is written as '@organization/name'.",
    type=str,
    required=True,
)
@click.option(
    "--alias",
    help="Export only the version this alias points to (e.g. 'champion'). Skills without \
this alias are skipped. Default is to export all versions.",
    type=str,
    required=False,
)
@click.option(
    "--use-threads", help="Use multithreading for export.", is_flag=True, default=False
)
def main(output_dir, skills, alias, use_threads):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")

    # Handle 'all', file, or comma-separated list
    if skills.endswith(".txt"):
        with open(skills, "r", encoding="utf-8") as f:
            skill_names_list = [line.strip() for line in f if line.strip()]
    elif skills.lower() == "all":
        skill_names_list = None  # None means export all
    else:
        skill_names_list = [name.strip() for name in skills.split(",")]

    result = export_skills(
        output_dir=output_dir,
        skill_names=skill_names_list,
        alias=alias,
        use_threads=use_threads,
    )

    # Check for failures
    if result is None:
        _logger.error("Skill export failed with unknown error")
        sys.exit(1)
    elif "unsupported" in result:
        _logger.error(skill_not_supported_message())
        sys.exit(1)
    elif "error" in result:
        _logger.error(f"Skill export failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
