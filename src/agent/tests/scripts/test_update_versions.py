from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent


def load_update_versions_module():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "update_versions.py"
    spec = importlib.util.spec_from_file_location("update_versions", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def test_update_repo_versions_updates_all_target_files(tmp_path):
    module = load_update_versions_module()
    repo_root = tmp_path / "repo"

    write_file(
        repo_root / "src/agent/pyproject.toml",
        """
        [project]
        version = "1.1.0b13"
        """,
    )
    write_file(
        repo_root / "src/plugins/aidev_bkplugin/pyproject.toml",
        """
        [project]
        version = "1.1.0b9"
        dependencies = [
            "aidev-agent>=1.1.0b9",
        ]
        """,
    )
    write_file(
        repo_root / "src/plugins/aidev_wxbot/pyproject.toml",
        """
        [project]
        version = "1.1.0b2"
        dependencies = [
            "aidev-bkplugin>=1.1.0b3",
        ]
        """,
    )
    write_file(
        repo_root / "src/plugins/aidev_wxbot/aidev_wxbot/__init__.py",
        """
        __version__ = "1.0.0"
        """,
    )
    write_file(
        repo_root / "src/plugins/aidev_ai_blueking/pyproject.toml",
        """
        [project]
        version = "2.0.0-dev.31"
        dependencies = [
            "aidev-agent>=1.1.0b5",
        ]
        """,
    )
    write_file(
        repo_root / "template/{{cookiecutter.project_name}}/pyproject.toml",
        """
        [project]
        version = "1.1.0rc5"
        dependencies = [
            "aidev-agent==1.1.0b12",
            "aidev-bkplugin==1.1.0b9",
            "aidev-wxbot==1.1.0b2",
            "aidev-ai-blueking==2.0.0.dev31",
        ]
        """,
    )
    write_file(
        repo_root / "template/{{cookiecutter.project_name}}/requirements.txt",
        """
        aidev-agent==1.1.0b13
        aidev-ai-blueking==2.0.0.dev31
        aidev-bkplugin==1.1.0b9
        aidev-wxbot==1.1.0b2
        """,
    )
    write_file(
        repo_root / "template/{{cookiecutter.project_name}}/bk_plugin/versions/assistant.py",
        """
        class CommonAgent:
            class Meta:
                version = "1.0.0assistant"
        """,
    )
    write_file(
        repo_root / "template/{{cookiecutter.project_name}}/readme.md",
        """
        curl -X POST http://127.0.0.1:8000/bk_plugin/invoke/1.0.0assistant
        curl -X POST https://example.com/prod/invoke/1.0.0assistant/
        """,
    )

    updated_files = module.update_repo_versions(repo_root, "2.0.0b1")

    assert {path.as_posix() for path in updated_files} == {
        "src/agent/pyproject.toml",
        "src/plugins/aidev_bkplugin/pyproject.toml",
        "src/plugins/aidev_wxbot/pyproject.toml",
        "src/plugins/aidev_wxbot/aidev_wxbot/__init__.py",
        "src/plugins/aidev_ai_blueking/pyproject.toml",
        "template/{{cookiecutter.project_name}}/pyproject.toml",
        "template/{{cookiecutter.project_name}}/requirements.txt",
    }
    assert 'version = "2.0.0b1"' in (repo_root / "src/agent/pyproject.toml").read_text(encoding="utf-8")
    assert '"aidev-agent>=2.0.0b1"' in (repo_root / "src/plugins/aidev_bkplugin/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"aidev-bkplugin>=2.0.0b1"' in (repo_root / "src/plugins/aidev_wxbot/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "2.0.0b1"' in (repo_root / "src/plugins/aidev_wxbot/aidev_wxbot/__init__.py").read_text(
        encoding="utf-8"
    )
    assert '"aidev-agent>=2.0.0b1"' in (repo_root / "src/plugins/aidev_ai_blueking/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'version = "2.0.0b1"' in (repo_root / "template/{{cookiecutter.project_name}}/pyproject.toml").read_text(
        encoding="utf-8"
    )
    requirements_text = (repo_root / "template/{{cookiecutter.project_name}}/requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "aidev-agent==2.0.0b1" in requirements_text
    assert "aidev-ai-blueking==2.0.0b1" in requirements_text
    assert "aidev-bkplugin==2.0.0b1" in requirements_text
    assert "aidev-wxbot==2.0.0b1" in requirements_text
    assistant_text = (repo_root / "template/{{cookiecutter.project_name}}/bk_plugin/versions/assistant.py").read_text(
        encoding="utf-8"
    )
    assert 'version = "1.0.0assistant"' in assistant_text
    readme_text = (repo_root / "template/{{cookiecutter.project_name}}/readme.md").read_text(encoding="utf-8")
    assert "invoke/1.0.0assistant" in readme_text
