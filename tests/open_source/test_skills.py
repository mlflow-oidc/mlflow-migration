"""
Tests for skill export and import functionality.
"""

import os
import pytest
import mlflow
from mlflow_migration.skill.export_skill import export_skill
from mlflow_migration.skill.import_skill import import_skill
from mlflow_migration.bulk.export_skills import export_skills
from mlflow_migration.bulk.import_skills import import_skills
from mlflow_migration.common.io_utils import read_file_mlflow
from mlflow_migration.common.skill_utils import (
    delete_skill,
    parse_skill_identity,
    search_all_skill_versions,
    skill_dir_name,
)
from mlflow_migration.common.version_utils import has_skill_support
from tests.utils_test import create_output_dir
from tests.open_source.init_tests import mlflow_context

pytestmark = pytest.mark.skipif(
    not has_skill_support(),
    reason="Skill registry not supported in this MLflow version",
)

_GIT_URL = "https://github.com/acme/skills.git"
_GIT_DIGEST = "a" * 64


def _write_skill_dir(tmp_path, name: str, body: str = "Do the task.") -> str:
    """Create a local skill directory with a SKILL.md manifest and a supporting file."""
    skill_dir = tmp_path / f"{name}-{len(os.listdir(tmp_path))}"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n\n# {name}\n\n{body}\n"
    )
    (skill_dir / "scripts" / "run.py").write_text(f"print('{body}')\n")
    return str(skill_dir)


def _register_local_skill(
    client, tmp_path, name: str, body: str = "Do the task.", fresh: bool = True
):
    """
    Register a skill version with MLflow-hosted content. Skips if the server cannot host it.
    If fresh, an existing skill of that name (from a previous test run) is deleted first.
    """
    mlflow.set_tracking_uri(client.tracking_uri)
    if fresh:
        delete_skill(client, name)
    try:
        return mlflow.genai.register_skill(
            source=_write_skill_dir(tmp_path, name, body), name=name
        )
    except Exception as e:
        pytest.skip(f"Tracking server cannot host skill content: {e}")


def _read_content(root: str) -> dict[str, str]:
    content = {}
    for dir_path, _, files in os.walk(root):
        for file in files:
            path = os.path.join(dir_path, file)
            with open(path, "r", encoding="utf-8") as f:
                content[os.path.relpath(path, root)] = f.read()
    return content


def test_parse_skill_identity():
    assert parse_skill_identity("code-review") == ("", "code-review")
    assert parse_skill_identity("@acme/code-review") == ("acme", "code-review")
    assert skill_dir_name("code-review") == "code-review"
    assert skill_dir_name("code-review", "acme") == "acme__code-review"
    with pytest.raises(ValueError):
        parse_skill_identity("@acme")


def test_export_import_skill(mlflow_context, tmp_path):
    """
    Export a skill with MLflow-hosted and Git-sourced versions, tags, an alias and a
    deprecated version, import it under a new name and compare source and destination.
    """
    src, dst = mlflow_context.client_src, mlflow_context.client_dst
    name = "test-skill-single"
    imported_name = f"{name}-imported"

    v1 = _register_local_skill(src, tmp_path, name)
    v2 = _register_local_skill(
        src, tmp_path, name, body="Do the task better.", fresh=False
    )
    v3 = src.create_skill_version(
        name=name,
        source=mlflow.genai.GitSource(_GIT_URL, ref="v1", subpath="skills/single"),
        digest=_GIT_DIGEST,
        status="draft",
    )
    src.update_skill(name=name, description="A test skill")
    src.set_skill_tag(name=name, key="team", value="ml")
    src.set_skill_version_tag(name=name, version=v2.version, key="reviewed", value="y")
    src.set_skill_alias(name=name, alias="production", version=v2.version)
    src.update_skill_version(name=name, version=v1.version, status="deprecated")

    output_dir = f"{mlflow_context.output_dir}/skill_single"
    create_output_dir(output_dir)

    exported = export_skill(skill_name=name, output_dir=output_dir, mlflow_client=src)
    assert exported is not None
    assert exported.name == name

    skill_info = read_file_mlflow(os.path.join(output_dir, "skill.json"))["skill"]
    assert [v["version"] for v in skill_info["versions"]] == [1, 2, 3]
    assert skill_info["versions"][0]["content_path"] == "versions/1/content"
    assert skill_info["versions"][2]["content_path"] is None
    assert os.path.exists(os.path.join(output_dir, "versions/2/content/SKILL.md"))

    delete_skill(dst, imported_name)
    result = import_skill(
        input_dir=output_dir, skill_name=imported_name, mlflow_client=dst
    )
    assert result == (imported_name, [1, 2, 3])

    skill = dst.get_skill(name=imported_name)
    assert skill.description == "A test skill"
    assert skill.tags == {"team": "ml"}
    assert skill.aliases == {"production": v2.version}

    src_versions = search_all_skill_versions(src, name)
    dst_versions = search_all_skill_versions(dst, imported_name)
    assert len(dst_versions) == 3
    for src_version, dst_version in zip(src_versions, dst_versions):
        assert dst_version.version == src_version.version
        assert dst_version.status == src_version.status
        assert dst_version.digest == src_version.digest
        assert dst_version.source_type == src_version.source_type
        assert dst_version.tags == src_version.tags
    assert str(dst_versions[0].status) == "deprecated"
    assert dst_versions[2].source == v3.source

    # Content of the imported MLflow-hosted version is the same as the exported content
    mlflow.set_tracking_uri(dst.tracking_uri)
    pulled = mlflow.genai.skills.pull(
        f"skills:/{imported_name}/2", destination=str(tmp_path / "pulled")
    )
    assert _read_content(pulled) == _read_content(
        os.path.join(output_dir, "versions/2/content")
    )
    assert "Do the task better." in _read_content(pulled)["SKILL.md"]


def test_import_skill_delete_existing(mlflow_context, tmp_path):
    """Importing twice reuses the existing version. Deleting first recreates the skill."""
    src, dst = mlflow_context.client_src, mlflow_context.client_dst
    name = "test-skill-delete"
    _register_local_skill(src, tmp_path, name)

    output_dir = f"{mlflow_context.output_dir}/skill_delete"
    create_output_dir(output_dir)
    assert export_skill(skill_name=name, output_dir=output_dir, mlflow_client=src)

    delete_skill(dst, name)
    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [1])
    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [1])
    assert import_skill(input_dir=output_dir, delete_skill=True, mlflow_client=dst) == (
        name,
        [1],
    )
    assert len(search_all_skill_versions(dst, name)) == 1


def test_promote_champion(mlflow_context, tmp_path):
    """
    Promotion workflow: the version marked 'champion' is exported and imported into
    another environment where it becomes the 'champion'. Then a newer version is promoted.
    """
    src, dst = mlflow_context.client_src, mlflow_context.client_dst
    name = "test-skill-champion"
    output_dir = f"{mlflow_context.output_dir}/skill_champion"
    delete_skill(dst, name)

    _register_local_skill(src, tmp_path, name, body="First.")
    v2 = _register_local_skill(src, tmp_path, name, body="Second.", fresh=False)
    v3 = _register_local_skill(src, tmp_path, name, body="Third.", fresh=False)
    src.set_skill_alias(name=name, alias="champion", version=v2.version)
    src.set_skill_alias(name=name, alias="challenger", version=v3.version)

    # Without the alias nothing is exported
    create_output_dir(output_dir)
    assert (
        export_skill(
            skill_name=name, output_dir=output_dir, alias="nope", mlflow_client=src
        )
        is None
    )

    # Promote version 2
    create_output_dir(output_dir)
    assert export_skill(
        skill_name=name, output_dir=output_dir, alias="champion", mlflow_client=src
    )
    skill_info = read_file_mlflow(os.path.join(output_dir, "skill.json"))["skill"]
    assert [v["version"] for v in skill_info["versions"]] == [2]
    assert skill_info["aliases"] == {"champion": 2}
    assert os.listdir(os.path.join(output_dir, "versions")) == ["2"]

    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [1])
    champion = dst.get_skill_version_by_alias(name=name, alias="champion")
    assert champion.version == 1
    assert champion.digest == v2.digest

    # Importing the same champion again does not create a new version
    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [1])
    assert len(search_all_skill_versions(dst, name)) == 1

    # Promote version 3: the destination gets a new version and the alias moves to it
    src.set_skill_alias(name=name, alias="champion", version=v3.version)
    create_output_dir(output_dir)
    assert export_skill(
        skill_name=name, output_dir=output_dir, alias="champion", mlflow_client=src
    )
    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [2])
    champion = dst.get_skill_version_by_alias(name=name, alias="champion")
    assert champion.version == 2
    assert champion.digest == v3.digest
    assert dst.get_skill(name=name).aliases == {"champion": 2, "challenger": 2}

    # Roll back to version 2: the already imported version is reused
    src.set_skill_alias(name=name, alias="champion", version=v2.version)
    create_output_dir(output_dir)
    assert export_skill(
        skill_name=name, output_dir=output_dir, alias="champion", mlflow_client=src
    )
    assert import_skill(input_dir=output_dir, mlflow_client=dst) == (name, [1])
    assert dst.get_skill_version_by_alias(name=name, alias="champion").version == 1
    assert len(search_all_skill_versions(dst, name)) == 2


def test_bulk_export_champions(mlflow_context, tmp_path):
    """Bulk export with an alias exports only skills having the alias."""
    src, dst = mlflow_context.client_src, mlflow_context.client_dst
    with_alias, without_alias = "test-skill-champ-yes", "test-skill-champ-no"
    _register_local_skill(src, tmp_path, with_alias)
    v2 = _register_local_skill(src, tmp_path, with_alias, body="Two.", fresh=False)
    _register_local_skill(src, tmp_path, without_alias)
    src.set_skill_alias(name=with_alias, alias="champion", version=v2.version)

    output_dir = f"{mlflow_context.output_dir}/skills_champions"
    create_output_dir(output_dir)
    export_result = export_skills(
        output_dir=output_dir,
        skill_names=[with_alias, without_alias],
        alias="champion",
        mlflow_client=src,
    )
    assert export_result["successful_exports"] == 1
    assert export_result["failed_exports"] == 0
    assert export_result["skipped_skills"] == [without_alias]
    assert not os.path.exists(os.path.join(output_dir, without_alias))

    delete_skill(dst, with_alias)
    import_result = import_skills(input_dir=output_dir, mlflow_client=dst)
    assert import_result["successful_imports"] == 1
    champion = dst.get_skill_version_by_alias(name=with_alias, alias="champion")
    assert champion.digest == v2.digest


def test_export_skill_not_found(mlflow_context):
    output_dir = f"{mlflow_context.output_dir}/skill_not_found"
    create_output_dir(output_dir)
    exported = export_skill(
        skill_name="test-skill-does-not-exist",
        output_dir=output_dir,
        mlflow_client=mlflow_context.client_src,
    )
    assert exported is None


def test_bulk_export_import_skills(mlflow_context, tmp_path):
    """Export all skills, including one scoped to an organization, and import them."""
    src, dst = mlflow_context.client_src, mlflow_context.client_dst
    names = ["test-skill-bulk-1", "test-skill-bulk-2"]
    for name in names:
        _register_local_skill(src, tmp_path, name)

    organization = "test-org"
    org_name = "test-skill-bulk-org"
    delete_skill(src, org_name, organization)
    src.create_skill_version(
        name=org_name,
        organization=organization,
        source=mlflow.genai.GitSource(_GIT_URL, ref="main"),
        digest=_GIT_DIGEST,
    )

    output_dir = f"{mlflow_context.output_dir}/skills_bulk"
    create_output_dir(output_dir)

    export_result = export_skills(
        output_dir=output_dir, skill_names=None, use_threads=False, mlflow_client=src
    )
    assert export_result["successful_exports"] >= len(names) + 1
    assert export_result["failed_exports"] == 0
    assert os.path.exists(os.path.join(output_dir, "test-org__test-skill-bulk-org"))

    import_result = import_skills(
        input_dir=output_dir, delete_skill=True, use_threads=False, mlflow_client=dst
    )
    assert import_result["successful_imports"] == export_result["successful_exports"]
    assert import_result["failed_imports"] == 0

    for name in names:
        assert len(search_all_skill_versions(dst, name)) == 1
    org_versions = search_all_skill_versions(dst, org_name, organization)
    assert len(org_versions) == 1
    assert org_versions[0].organization == organization


def test_export_specific_skills(mlflow_context, tmp_path):
    """Only the specified skills are exported."""
    src = mlflow_context.client_src
    for name in ["test-skill-a", "test-skill-b", "test-skill-c"]:
        _register_local_skill(src, tmp_path, name)

    output_dir = f"{mlflow_context.output_dir}/skills_specific"
    create_output_dir(output_dir)

    export_result = export_skills(
        output_dir=output_dir,
        skill_names=["test-skill-a", "test-skill-c"],
        use_threads=True,
        mlflow_client=src,
    )
    assert export_result["successful_exports"] == 2
    assert export_result["failed_exports"] == 0
    assert sorted(d for d in os.listdir(output_dir) if not d.endswith(".json")) == [
        "test-skill-a",
        "test-skill-c",
    ]
