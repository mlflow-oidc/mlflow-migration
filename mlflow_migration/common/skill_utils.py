"""
Skill registry utilities.

The MLflow Skill Registry (MLflow 3.18.0+) stores versioned agent skills. A skill is
identified by its name and an optional organization, written as 'name' or '@organization/name'.
"""

import mlflow
from mlflow_migration.common import utils

# File name of the export file of a skill
SKILL_FILE = "skill.json"

# Maximum results per page when searching skills and skill versions
MAX_SEARCH_RESULTS = 100

_logger = utils.getLogger(__name__)


def skill_not_supported_message() -> str:
    return f"Skill registry not supported in MLflow {mlflow.__version__} (requires 3.18.0+)"


def parse_skill_identity(identity: str) -> tuple[str, str]:
    """
    Parse a skill identity into (organization, name).

    :param identity: Either 'name' or '@organization/name'.
    :return: Tuple of organization (empty string if none) and skill name.
    """
    identity = identity.strip()
    if identity.startswith("@"):
        organization, sep, name = identity[1:].partition("/")
        if not sep or not organization or not name:
            raise ValueError(
                f"Invalid skill identity '{identity}': expected 'name' or '@organization/name'"
            )
        return organization, name
    return "", identity


def format_skill_identity(name: str, organization: str | None = "") -> str:
    """Format a skill identity as 'name' or '@organization/name'."""
    return f"@{organization}/{name}" if organization else name


def skill_dir_name(name: str, organization: str | None = "") -> str:
    """
    Directory name of an exported skill. Neither skill nor organization names can contain
    underscores so 'organization__name' is unambiguous.
    """
    return f"{organization}__{name}" if organization else name


def search_all_skills(client: mlflow.MlflowClient) -> list:
    """Get all skills of the registry."""
    skills = []
    page_token = None
    while True:
        page = client.search_skills(
            max_results=MAX_SEARCH_RESULTS, page_token=page_token
        )
        skills.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return skills


def search_all_skill_versions(
    client: mlflow.MlflowClient, name: str, organization: str = ""
) -> list:
    """Get all (non-deleted) versions of a skill sorted by ascending version number."""
    versions = []
    page_token = None
    while True:
        page = client.search_skill_versions(
            name=name,
            organization=organization,
            max_results=MAX_SEARCH_RESULTS,
            page_token=page_token,
        )
        versions.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return sorted(versions, key=lambda v: int(v.version))


def skill_exists(
    client: mlflow.MlflowClient, name: str, organization: str = ""
) -> bool:
    try:
        client.get_skill(name=name, organization=organization)
        return True
    except Exception:
        return False


def delete_skill(
    client: mlflow.MlflowClient, name: str, organization: str = ""
) -> None:
    """Delete a skill and all its versions. Does nothing if the skill does not exist."""
    identity = format_skill_identity(name, organization)
    if not skill_exists(client, name, organization):
        return
    try:
        _logger.info(f"Deleting skill '{identity}' and its versions")
        client.delete_skill(name=name, organization=organization)
    except Exception as e:
        _logger.warning(f"Failed to delete skill '{identity}': {e}")
