import hashlib
import json
from pathlib import Path

import pytest

from scripts.fetch_sources import check_local_sources, fetch_sources, load_source_manifest


def write_manifest(project_root: Path, *, raw_path="data/raw/example", mapping_source="docs"):
    text = b"Pinned Qdrant documentation\n"
    payload = [
        {
            "source_name": "Example Git docs",
            "source_type": "git_sparse_checkout",
            "repo_url": "https://example.com/docs.git",
            "commit": "a" * 40,
            "path_mappings": [{"source": mapping_source, "destination": "docs"}],
            "raw_path": raw_path,
        },
        {
            "source_name": "Example text docs",
            "source_type": "llms_full_txt",
            "url": "https://example.com/llms.txt",
            "sha256": hashlib.sha256(text).hexdigest(),
            "raw_path": "data/raw/text/llms.txt",
        },
    ]
    path = project_root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, text


def test_manifest_loads_pinned_sources(tmp_path):
    path, _ = write_manifest(tmp_path)

    sources = load_source_manifest(path, project_root=tmp_path)

    assert [source["install_root"] for source in sources] == [Path("data/raw/example"), Path("data/raw/text")]
    assert sources[0]["path_mappings"][0] == {"source": Path("docs"), "destination": Path("docs")}


def test_checked_manifest_uses_immutable_official_git_sources():
    sources = load_source_manifest()

    assert [source["source_name"] for source in sources] == [
        "Qdrant documentation",
        "FastAPI documentation",
        "MLflow documentation",
    ]
    assert all(source["source_type"] == "git_sparse_checkout" for source in sources)
    assert all(source["repo_url"].startswith("https://github.com/") for source in sources)
    assert all(len(source["commit"]) == 40 for source in sources)
    assert sources[0]["path_mappings"] == [
        {
            "source": Path("qdrant-landing/static/llms-full.txt"),
            "destination": Path("qdrant_llms_full.txt"),
            "kind": "file",
        }
    ]


@pytest.mark.parametrize("raw_path,mapping_source", [("../outside", "docs"), ("data/raw/example", "../outside")])
def test_manifest_rejects_paths_outside_declared_source_root(tmp_path, raw_path, mapping_source):
    path, _ = write_manifest(tmp_path, raw_path=raw_path, mapping_source=mapping_source)

    with pytest.raises(ValueError, match="safe project-relative path|inside data/raw"):
        load_source_manifest(path, project_root=tmp_path)


def test_fetch_is_staged_and_requires_force_for_replacement(tmp_path):
    path, text = write_manifest(tmp_path)
    sources = load_source_manifest(path, project_root=tmp_path)

    def fake_git(source, staging_root, _work_root):
        destination = staging_root / source["raw_path"] / source["path_mappings"][0]["destination"]
        destination.mkdir(parents=True)
        (destination / "index.md").write_text("first", encoding="utf-8")

    def fake_text(source, staging_root, _work_root):
        destination = staging_root / source["raw_path"]
        destination.parent.mkdir(parents=True)
        destination.write_bytes(text)

    installed = fetch_sources(sources, project_root=tmp_path, git_fetcher=fake_git, text_fetcher=fake_text)

    assert installed == [tmp_path / "data/raw/example", tmp_path / "data/raw/text"]
    assert (tmp_path / "data/raw/example/docs/index.md").read_text(encoding="utf-8") == "first"
    assert check_local_sources(sources, project_root=tmp_path) == [("Example Git docs", 1), ("Example text docs", 1)]
    with pytest.raises(FileExistsError, match="--force"):
        fetch_sources(sources, project_root=tmp_path, git_fetcher=fake_git, text_fetcher=fake_text)

    fetch_sources(sources, project_root=tmp_path, force=True, git_fetcher=fake_git, text_fetcher=fake_text)


def test_local_check_rejects_changed_download(tmp_path):
    path, _ = write_manifest(tmp_path)
    sources = load_source_manifest(path, project_root=tmp_path)
    git_docs = tmp_path / "data/raw/example/docs"
    git_docs.mkdir(parents=True)
    (git_docs / "index.md").write_text("docs", encoding="utf-8")
    text_path = tmp_path / "data/raw/text/llms.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check_local_sources(sources, project_root=tmp_path)


def test_local_check_accepts_a_git_mapped_file(tmp_path):
    path, _ = write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))[:1]
    payload[0]["path_mappings"] = [{"source": "static/llms-full.txt", "destination": "qdrant_llms_full.txt"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    sources = load_source_manifest(path, project_root=tmp_path)
    mapped_file = tmp_path / "data/raw/example/qdrant_llms_full.txt"
    mapped_file.parent.mkdir(parents=True)
    mapped_file.write_text("pinned", encoding="utf-8")

    assert check_local_sources(sources, project_root=tmp_path) == [("Example Git docs", 1)]
