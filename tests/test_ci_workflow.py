from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/ci.yml"
README_PATH = PROJECT_ROOT / "README.md"

EXPECTED_JOB_COMMANDS = {
    "lint": "python -m ruff check src tests scripts dashboard",
    "unit": "make test-unit-ci PYTHON=python",
    "api": "make test-api-ci PYTHON=python",
    "evaluation-smoke": "make test-evaluation-smoke EVAL_GATE_PYTHON=python",
    "evaluation-gate": "make eval-gate EVAL_GATE_PYTHON=python",
}


def load_workflow():
    """Load GitHub's YAML keys as strings rather than YAML 1.1 booleans."""
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_ci_runs_all_required_checks_for_pull_requests():
    workflow = load_workflow()

    assert workflow["name"] == "CI"
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert set(workflow["jobs"]) == set(EXPECTED_JOB_COMMANDS)


def test_ci_jobs_are_bounded_read_only_python_312_jobs():
    workflow = load_workflow()

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert "github.ref" in workflow["concurrency"]["group"]
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == "10"
        assert "needs" not in job
        steps = job["steps"]
        assert steps[0]["uses"] == "actions/checkout@v6"
        assert steps[1]["uses"] == "actions/setup-python@v6"
        assert steps[1]["with"] == {
            "python-version": "3.12",
            "cache": "pip",
            "cache-dependency-path": "requirements-ci.txt",
        }


def test_ci_jobs_install_one_pinned_dependency_boundary_and_run_expected_commands():
    workflow = load_workflow()

    for job_name, expected_command in EXPECTED_JOB_COMMANDS.items():
        run_steps = [step["run"] for step in workflow["jobs"][job_name]["steps"] if "run" in step]
        assert run_steps == ["python -m pip install -r requirements-ci.txt", expected_command]


def test_ci_forces_offline_template_execution_without_secrets_or_services():
    workflow = load_workflow()

    assert workflow["env"] == {
        "PYTHONHASHSEED": "0",
        "RAGOPS_LLM_PROVIDER": "template",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in workflow_text
    assert "docker" not in workflow_text.casefold()
    assert "services:" not in workflow_text


def test_readme_ci_badge_targets_the_checked_workflow():
    readme = README_PATH.read_text(encoding="utf-8")

    badge = "[![CI](https://github.com/NeguseNegest/Ragops-Control-Plane/actions/workflows/ci.yml/badge.svg)](https://github.com/NeguseNegest/Ragops-Control-Plane/actions/workflows/ci.yml)"
    assert badge in readme
