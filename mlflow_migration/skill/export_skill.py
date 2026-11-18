"""
Exports an MLflow skill to a directory: either all its versions or only the version an
alias (e.g. 'champion') points to.

Skills live in the MLflow Skill Registry (MLflow 3.18.0+). The content of a skill version
is either hosted by MLflow itself (uploaded from a local directory) or lives in an external
source (Git, OCI or ZIP). MLflow-hosted content is downloaded into the export directory.
External sources are exported by reference since the registry only stores a pointer to them.
"""

import os
import shutil
import tempfile
import click
import mlflow
from mlflow.exceptions import RestException

from mlflow_migration.common import utils, io_utils
from mlflow_migration.common.click_options import opt_output_dir
from mlflow_migration.common.timestamp_utils import adjust_timestamps
from mlflow_migration.common.version_utils import has_skill_support, log_version_info
from mlflow_migration.common.skill_utils import (
    SKILL_FILE,
    format_skill_identity,
    search_all_skill_versions,
    skill_not_supported_message,
)
from mlflow_migration.client.client_utils import create_mlflow_client

_logger = utils.getLogger(__name__)

_TIMESTAMP_KEYS = ["creation_timestamp", "last_updated_timestamp"]


def export_skill(
    skill_name: str,
    output_dir: str,
    organization: str = "",
    alias: str | None = None,
    mlflow_client: mlflow.MlflowClient | None = None,
) -> object | None:
    """
    Export a skill to a directory.

    :param skill_name: Name of the skill to export.
    :param output_dir: Output directory.
    :param organization: Optional organization the skill is scoped to.
    :param alias: Export only the version this alias (e.g. 'champion') points to.
        If None, all versions are exported.
    :param mlflow_client: MLflow client.
    :return: Skill object or None if export failed.
    """

    if not has_skill_support():
        raise Exception(skill_not_supported_message())

    mlflow_client = mlflow_client or create_mlflow_client()
    log_version_info()

    organization = organization or ""
    identity = format_skill_identity(skill_name, organization)

    try:
        skill = mlflow_client.get_skill(name=skill_name, organization=organization)
        aliases = dict(skill.aliases) if skill.aliases else {}
        if alias:
            version = mlflow_client.get_skill_version_by_alias(
                name=skill_name, alias=alias, organization=organization
            )
            versions = [version]
            # Keep only the aliases of the exported version
            aliases = {
                k: v for k, v in aliases.items() if int(v) == int(version.version)
            }
            _logger.info(
                f"Exporting skill '{identity}' version {version.version} (alias '{alias}')"
            )
        else:
            versions = search_all_skill_versions(
                mlflow_client, skill_name, organization
            )
            _logger.info(
                f"Exporting skill '{identity}' with {len(versions)} version(s)"
            )

        os.makedirs(output_dir, exist_ok=True)
        versions_info = [
            _export_version(mlflow_client, version, output_dir, identity)
            for version in versions
        ]

        skill_info = {
            "name": skill.name,
            "organization": skill.organization or "",
            "description": skill.description,
            "icons": skill.icons,
            "workspace": skill.workspace,
            "status": _to_str(skill.status),
            "tags": dict(skill.tags) if skill.tags else {},
            "aliases": aliases,
            "latest_version": skill.latest_version,
            "source_type": _to_str(skill.source_type),
            "created_by": skill.created_by,
            "last_updated_by": skill.last_updated_by,
            "creation_timestamp": skill.creation_timestamp,
            "last_updated_timestamp": skill.last_updated_timestamp,
            "versions": versions_info,
        }
        adjust_timestamps(skill_info, _TIMESTAMP_KEYS)

        io_utils.write_export_file(
            output_dir,
            SKILL_FILE,
            __file__,
            {"skill": skill_info},
            {"export_alias": alias} if alias else None,
        )

        _logger.info(f"Successfully exported skill '{identity}'")
        return skill

    except RestException as e:
        _logger.error(
            f"Skill export failed: {{'skill': '{identity}', 'RestException': {e.json}}}"
        )
        return None
    except Exception as e:
        _logger.error(
            f"Skill export failed: {{'skill': '{identity}', 'Exception': {str(e)}}}"
        )
        return None


def _to_str(value) -> str | None:
    return None if value is None else str(value)


def _source_to_dict(source) -> dict | str | None:
    """Convert a typed skill source (GitSource, OCISource, ZipSource, MlflowSource) to a dict."""
    if source is None or isinstance(source, str):
        return source
    return source.to_dict()


def _export_version(
    mlflow_client: mlflow.MlflowClient, version, output_dir: str, identity: str
) -> dict:
    """Build the export data of a skill version and download its content if MLflow hosts it."""
    from mlflow.entities.skill_source import MlflowSource

    version_info = {
        "version": version.version,
        "source_type": _to_str(version.source_type),
        "source": _source_to_dict(version.source),
        "digest": version.digest,
        "status": _to_str(version.status),
        "tags": dict(version.tags) if version.tags else {},
        "aliases": list(version.aliases) if version.aliases else [],
        "content_path": None,
        "created_by": version.created_by,
        "last_updated_by": version.last_updated_by,
        "creation_timestamp": version.creation_timestamp,
        "last_updated_timestamp": version.last_updated_timestamp,
    }
    adjust_timestamps(version_info, _TIMESTAMP_KEYS)

    if isinstance(version.source, MlflowSource):
        content_path = os.path.join("versions", str(version.version), "content")
        try:
            _download_content(
                mlflow_client, version, os.path.join(output_dir, content_path)
            )
            version_info["content_path"] = content_path
        except Exception as e:
            _logger.error(
                f"Failed to download content of skill '{identity}' version {version.version}: {e}"
            )
    return version_info


def _download_content(
    mlflow_client: mlflow.MlflowClient, version, content_dir: str
) -> None:
    """
    Download the MLflow-hosted content of a skill version into content_dir.
    Uses the tracking URI of the client so the export does not depend on the global tracking URI.
    """
    from mlflow.artifacts import download_artifacts
    from mlflow.genai.skill_content.digest import compute_tree_digest

    source = version.source
    artifact_uri = source.artifact_path.rstrip("/")
    if source.subpath:
        artifact_uri = f"{artifact_uri}/{source.subpath.strip('/')}"

    if os.path.exists(content_dir):
        shutil.rmtree(content_dir)
    os.makedirs(os.path.dirname(content_dir), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded = download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=tmp_dir,
            tracking_uri=mlflow_client.tracking_uri,
        )
        if not os.path.isdir(downloaded):
            raise Exception(f"Skill content '{artifact_uri}' is not a directory")
        shutil.move(downloaded, content_dir)

    if version.digest and compute_tree_digest(content_dir) != version.digest:
        _logger.warning(
            f"Content of skill '{version.name}' version {version.version} does not match "
            f"its recorded digest {version.digest}"
        )


@click.command()
@click.option(
    "--skill-name", help="Name of the skill to export.", type=str, required=True
)
@click.option(
    "--organization",
    help="Organization the skill is scoped to. Default is no organization.",
    type=str,
    default="",
    show_default=False,
)
@click.option(
    "--alias",
    help="Export only the version this alias points to (e.g. 'champion'). Default is to export all versions.",
    type=str,
    required=False,
)
@opt_output_dir
def main(skill_name, organization, alias, output_dir):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")

    export_skill(
        skill_name=skill_name,
        organization=organization,
        alias=alias,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
