import re
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_targets():
    return set(re.findall(r"^([A-Za-z0-9_-]+):", (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8"), flags=re.MULTILINE))


def example_environment():
    values = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            assert separator and key not in values
            values[key] = value
    return values


def test_public_environment_covers_runtime_ports_and_has_no_credentials():
    values = example_environment()

    assert values["API_PORT"] == "8000"
    assert values["DASHBOARD_PORT"] == "8501"
    assert values["RAGOPS_API_PORT"] == "8000"
    assert values["RAGOPS_API_URL"] == "http://127.0.0.1:8000"
    assert values["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert values["QDRANT_HTTP_PORT"] == "6333"
    assert values["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5000"
    assert values["MLFLOW_PORT"] == "5000"
    assert values["RAGOPS_LLM_PROVIDER"] == "template"
    assert values["OPENAI_API_KEY"] == values["GEMINI_API_KEY"] == ""


def test_compose_is_isolatable_and_uses_host_visible_runtime_state():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert all("container_name" not in service for service in services.values())
    assert services["qdrant"]["image"] == "qdrant/qdrant:v1.18.2"
    assert "${QDRANT_HTTP_PORT:-6333}:6333" in services["qdrant"]["ports"]
    assert "${QDRANT_GRPC_PORT:-6334}:6334" in services["qdrant"]["ports"]
    assert "${MLFLOW_PORT:-5000}:5000" in services["mlflow"]["ports"]
    assert services["api"]["depends_on"] == ["qdrant"]
    assert "./data/processed:/app/data/processed:ro" in services["api"]["volumes"]
    assert "./data/traces:/app/data/traces" in services["api"]["volumes"]
    assert "ragops_trace_data" not in compose["volumes"]


def test_makefile_exposes_the_documented_clean_workflow_and_loads_env_for_runtime_commands():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    required = {"fetch-sources", "validate-sources", "ingest", "build-index", "evaluate", "test", "serve", "dashboard", "test-operations"}

    assert required <= make_targets()
    assert 'ENV_FILE ?= $(CURDIR)/.env' in makefile
    for target in ("evaluate", "index", "serve", "dashboard"):
        body = re.search(rf"^{target}:.*?(?=^[A-Za-z0-9_-]+:|\Z)", makefile, flags=re.MULTILINE | re.DOTALL).group(0)
        assert "$(RUN_ENV)" in body


def test_documented_make_commands_exist():
    targets = make_targets()
    for path in [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"^make\s+([A-Za-z0-9_-]+)", text, flags=re.MULTILINE):
            assert target in targets, f"Unknown Make target {target!r} in {path.relative_to(PROJECT_ROOT)}"


def test_user_facing_clean_setup_has_no_manual_corpus_step():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    operations = (PROJECT_ROOT / "docs/operations.md").read_text(encoding="utf-8")
    for command in ("make setup", "make fetch-sources", "docker compose up -d --build", "make ingest", "make build-index", "make evaluate", "make test", "make dashboard"):
        assert command in operations
    assert "make fetch-sources" in readme
    assert "place the pinned source snapshots" not in readme.lower()


def test_supported_python_range_matches_the_tested_runtime():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert project["requires-python"] == ">=3.11,<3.13"
    assert "Python 3.11 and 3.12 are supported" in readme
    assert "FROM python:3.12-slim" in (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_setup_declares_fetch_and_dashboard_runtime_boundaries():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "requests>=2.32.0,<3.0" in project["dependencies"]
    streamlit = "streamlit>=1.58.0,<1.59.0"
    assert streamlit in project["optional-dependencies"]["dashboard"]
    assert streamlit in project["optional-dependencies"]["dev"]
    assert "requests>=2.32.0,<3.0" in (PROJECT_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")


def test_runtime_paths_have_no_stale_todos():
    paths = [PROJECT_ROOT / "dashboard/app.py", *sorted((PROJECT_ROOT / "src/ragops/api").glob("*.py")), *sorted((PROJECT_ROOT / "src/ragops/generation").glob("*.py")), *sorted((PROJECT_ROOT / "src/ragops/routing").glob("*.py"))]
    stale = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"\b(?:TODO|FIXME|XXX)\b", line, flags=re.IGNORECASE):
                stale.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")
    assert stale == []
