"""
Imports an MLflow skill and all its versions from a directory.

Versions whose content was hosted by the source MLflow server are uploaded to the destination
server (which must serve artifacts). Versions with an external source (Git, OCI or ZIP) are
registered by reference with their recorded digest.
"""

import os
import tempfile
import click
import mlflow

from mlflow_migration.common import utils, io_utils
from mlflow_migration.common.click_options import opt_input_dir
from mlflow_migration.common.version_utils import has_skill_support, log_version_info
from mlflow_migration.common.skill_utils import (
    SKILL_FILE,
    delete_skill as delete_skill_util,
    format_skill_identity,
    search_all_skill_versions,
    skill_exists,
    skill_not_supported_message,
)
from mlflow_migration.client.client_utils import create_mlflow_client

_logger = utils.getLogger(__name__)


def import_skill(
    input_dir: str,
    skill_name: str | None = None,
    organization: str | None = None,
    delete_skill: bool = False,
    mlflow_client: mlflow.MlflowClient | None = None,
) -> tuple[str, list[int]] | None:
    """
    Import a skill and its versions from an exported directory.

    Version numbers are assigned by the destination server and can differ from the source
    version numbers. Aliases (e.g. 'champion') are set on the imported versions.
    A version that already exists in the destination skill (same source and content digest)
    is not registered again: it is reused, so repeating an import does not duplicate versions.

    :param input_dir: Input directory containing the exported skill.
    :param skill_name: Optional new name for the imported skill. If None, uses original name.
    :param organization: Optional new organization for the imported skill. If None, uses
        original organization.
    :param delete_skill: Delete existing skill before importing.
    :param mlflow_client: MLflow client.
    :return: Imported skill name and list of imported version numbers or None if import failed.
    """

    if not has_skill_support():
        raise Exception(skill_not_supported_message())

    mlflow_client = mlflow_client or create_mlflow_client()
    log_version_info()

    try:
        skill_path = os.path.join(input_dir, SKILL_FILE)
        if not os.path.exists(skill_path):
            raise Exception(f"Skill export file not found: {skill_path}")

        skill_info = io_utils.read_file_mlflow(skill_path).get("skill", {})

        name = skill_name or skill_info.get("name")
        if not name:
            raise Exception("No skill name specified and none found in export data")
        if organization is None:
            organization = skill_info.get("organization") or ""
        identity = format_skill_identity(name, organization)

        _logger.info(f"Importing skill '{identity}' from: {input_dir}")

        if delete_skill:
            delete_skill_util(mlflow_client, name, organization)

        _create_or_update_skill(mlflow_client, name, organization, skill_info)

        for key, value in (skill_info.get("tags") or {}).items():
            try:
                mlflow_client.set_skill_tag(
                    name=name, key=key, value=value, organization=organization
                )
            except Exception as e:
                _logger.warning(f"Failed to set tag '{key}' on skill '{identity}': {e}")

        # Maps source version number to destination version number
        version_map: dict[int, int] = {}
        versions = sorted(
            skill_info.get("versions") or [], key=lambda v: int(v["version"])
        )
        dst_versions = search_all_skill_versions(mlflow_client, name, organization)
        for version_info in versions:
            dst_version = _import_version(
                mlflow_client, name, organization, version_info, input_dir, dst_versions
            )
            if dst_version is not None:
                version_map[int(version_info["version"])] = dst_version

        _import_aliases(
            mlflow_client, name, organization, skill_info.get("aliases"), version_map
        )

        failed = len(versions) - len(version_map)
        if failed:
            _logger.warning(
                f"Imported skill '{identity}' with {failed} of {len(versions)} version(s) failed"
            )
        else:
            _logger.info(
                f"Successfully imported skill '{identity}' with {len(version_map)} version(s)"
            )
        return name, list(version_map.values())

    except Exception as e:
        _logger.error(f"Skill import failed: {str(e)}")
        return None


def _create_or_update_skill(
    mlflow_client: mlflow.MlflowClient, name: str, organization: str, skill_info: dict
) -> None:
    description = skill_info.get("description")
    icons = skill_info.get("icons")
    if skill_exists(mlflow_client, name, organization):
        _logger.info(
            f"Skill '{format_skill_identity(name, organization)}' already exists: "
            "new versions are added to it"
        )
        kwargs = {}
        if description is not None:
            kwargs["description"] = description
        if icons is not None:
            kwargs["icons"] = icons
        if kwargs:
            mlflow_client.update_skill(name=name, organization=organization, **kwargs)
    else:
        mlflow_client.create_skill(
            name=name, organization=organization, description=description, icons=icons
        )


def _build_source(version_info: dict):
    """Rebuild the external source (Git, OCI or ZIP) of an exported skill version."""
    from mlflow.entities.skill_source import GitSource, OCISource, ZipSource

    source = version_info.get("source")
    source_type = version_info.get("source_type")
    if source is None or isinstance(source, str):
        return source
    source_classes = {"git": GitSource, "oci": OCISource, "zip": ZipSource}
    if source_type not in source_classes:
        raise Exception(f"Unsupported skill source type '{source_type}'")
    return source_classes[source_type].from_dict(source)


def _import_version(
    mlflow_client: mlflow.MlflowClient,
    name: str,
    organization: str,
    version_info: dict,
    input_dir: str,
    dst_versions: list,
) -> int | None:
    """
    Import one skill version. Returns the destination version number or None if it failed.
    If the destination skill already has the same version (one of dst_versions), it is reused.
    """
    identity = format_skill_identity(name, organization)
    src_version = version_info.get("version")
    status = version_info.get("status") or "active"
    # A version can only be registered as 'active' or 'draft': other statuses are set afterwards
    initial_status = "draft" if status == "draft" else "active"

    existing = None
    try:
        if version_info.get("source_type") == "mlflow":
            from mlflow.genai.skill_content.digest import compute_tree_digest

            content_path = version_info.get("content_path")
            if not content_path:
                raise Exception("its content was not exported")
            content_dir = os.path.join(input_dir, content_path)
            if not os.path.isdir(content_dir):
                raise Exception(f"content directory not found: {content_dir}")
            digest = compute_tree_digest(content_dir)
            existing = next(
                (
                    v
                    for v in dst_versions
                    if str(v.source_type) == "mlflow" and v.digest == digest
                ),
                None,
            )
            version = existing or _upload_content(
                mlflow_client, name, organization, content_dir, digest, initial_status
            )
        else:
            source = _build_source(version_info)
            digest = version_info.get("digest")
            existing = next(
                (
                    v
                    for v in dst_versions
                    if str(v.source_type) != "mlflow"
                    and v.source == source
                    and v.digest == digest
                ),
                None,
            )
            version = existing or mlflow_client.create_skill_version(
                name=name,
                organization=organization,
                source=source,
                digest=digest,
                status=initial_status,
            )
    except Exception as e:
        _logger.error(
            f"Failed to import version {src_version} of skill '{identity}': {e}"
        )
        return None

    dst_version = int(version.version)
    if existing:
        _logger.info(
            f"Skill '{identity}' version {src_version} already exists as version {dst_version}: reusing it"
        )
    else:
        dst_versions.append(version)
        _logger.info(
            f"Imported skill '{identity}' version {src_version} as version {dst_version}"
        )

    # The status of a reused version is left as it is in the destination
    if not existing and status != initial_status:
        try:
            mlflow_client.update_skill_version(
                name=name,
                version=dst_version,
                organization=organization,
                status=status,
            )
        except Exception as e:
            _logger.warning(
                f"Failed to set status '{status}' on skill '{identity}' version {dst_version}: {e}"
            )

    for key, value in (version_info.get("tags") or {}).items():
        try:
            mlflow_client.set_skill_version_tag(
                name=name,
                version=dst_version,
                key=key,
                value=value,
                organization=organization,
            )
        except Exception as e:
            _logger.warning(
                f"Failed to set tag '{key}' on skill '{identity}' version {dst_version}: {e}"
            )

    return dst_version


def _upload_content(
    mlflow_client: mlflow.MlflowClient,
    name: str,
    organization: str,
    content_dir: str,
    digest: str,
    status: str,
):
    """Package a skill content directory and upload it as a new MLflow-hosted skill version."""
    from mlflow.genai.skill_content.archive import package_skill_tree

    with tempfile.TemporaryDirectory(prefix="mlflow-migration-skill-") as tmp_dir:
        archive = package_skill_tree(
            content_dir, os.path.join(tmp_dir, "content.tar.gz")
        )
        return mlflow_client.create_skill_version(
            name=name,
            organization=organization,
            source=str(archive),
            digest=digest,
            status=status,
        )


def _import_aliases(
    mlflow_client: mlflow.MlflowClient,
    name: str,
    organization: str,
    aliases: dict[str, int] | None,
    version_map: dict[int, int],
) -> None:
    identity = format_skill_identity(name, organization)
    for alias, src_version in (aliases or {}).items():
        dst_version = version_map.get(int(src_version))
        if dst_version is None:
            _logger.warning(
                f"Skipping alias '{alias}' of skill '{identity}': version {src_version} was not imported"
            )
            continue
        try:
            mlflow_client.set_skill_alias(
                name=name, alias=alias, version=dst_version, organization=organization
            )
        except Exception as e:
            _logger.warning(f"Failed to set alias '{alias}' on skill '{identity}': {e}")


@click.command()
@opt_input_dir
@click.option(
    "--skill-name",
    help="Optional new name for the imported skill. If not specified, uses original name.",
    type=str,
    required=False,
)
@click.option(
    "--organization",
    help="Optional new organization for the imported skill. If not specified, uses original organization.",
    type=str,
    required=False,
)
@click.option(
    "--delete-skill",
    help="Delete existing skill before importing.",
    type=bool,
    default=False,
)
def main(input_dir, skill_name, organization, delete_skill):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")

    import_skill(
        input_dir=input_dir,
        skill_name=skill_name,
        organization=organization,
        delete_skill=delete_skill,
    )


if __name__ == "__main__":
    main()
