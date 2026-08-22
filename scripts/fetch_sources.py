"""Fetch the pinned documentation corpus declared in the source manifest."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data/manifests/source_manifest.json")


def _relative_path(value, label):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a safe project-relative path: {value!r}")
    return path


def load_source_manifest(path=DEFAULT_MANIFEST, project_root=PROJECT_ROOT):
    project_root = Path(project_root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else project_root / path
    try:
        sources = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Source manifest does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Source manifest is invalid JSON: {error}") from error
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source manifest must contain a non-empty JSON list.")

    names = set()
    install_roots = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source manifest entry must be an object.")
        name = str(source.get("source_name", "")).strip()
        if not name or name in names:
            raise ValueError(f"Source names must be non-empty and unique: {name!r}")
        names.add(name)

        raw_path = _relative_path(source.get("raw_path", ""), f"{name} raw_path")
        if raw_path.parts[:2] != ("data", "raw") or len(raw_path.parts) < 3:
            raise ValueError(f"{name} raw_path must be inside data/raw/<source>.")
        source["raw_path"] = raw_path
        source["install_root"] = Path(*raw_path.parts[:3])
        if source["install_root"] in install_roots:
            raise ValueError(f"Each source must own one unique data/raw directory: {source['install_root']}")
        install_roots.add(source["install_root"])

        source_type = source.get("source_type")
        if source_type == "git_sparse_checkout":
            repo_url = str(source.get("repo_url", "")).strip()
            commit = str(source.get("commit", "")).strip().lower()
            mappings = source.get("path_mappings")
            if not repo_url.startswith("https://") or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError(f"{name} requires an HTTPS repo_url and a 40-character commit.")
            if not isinstance(mappings, list) or not mappings:
                raise ValueError(f"{name} requires at least one path mapping.")
            destinations = set()
            for mapping in mappings:
                if not isinstance(mapping, dict) or not {"source", "destination"} <= set(mapping) or set(mapping) - {"source", "destination", "kind"}:
                    raise ValueError(f"{name} path mappings require source, destination, and optional kind fields only.")
                if mapping.get("kind", "directory") not in {"directory", "file"}:
                    raise ValueError(f"{name} path mapping kind must be directory or file.")
                mapping["source"] = _relative_path(mapping["source"], f"{name} source mapping")
                mapping["destination"] = _relative_path(mapping["destination"], f"{name} destination mapping")
                if mapping["destination"] in destinations:
                    raise ValueError(f"{name} path mapping destinations must be unique.")
                destinations.add(mapping["destination"])
        elif source_type == "llms_full_txt":
            url = str(source.get("url", "")).strip()
            digest = str(source.get("sha256", "")).strip().lower()
            if not url.startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name} requires an HTTPS URL and a 64-character SHA256.")
            source["sha256"] = digest
        else:
            raise ValueError(f"Unsupported source_type for {name}: {source_type!r}")
    return sources


def _run_git(arguments):
    subprocess.run(["git", "-c", "gc.auto=0", *map(str, arguments)], check=True)


def fetch_git_source(source, staging_root, work_root):
    repository = work_root / re.sub(r"[^a-z0-9]+", "-", source["source_name"].lower()).strip("-")
    repository.mkdir(parents=True)
    _run_git(["-C", repository, "init", "--quiet"])
    _run_git(["-C", repository, "remote", "add", "origin", source["repo_url"]])
    file_checkout = any(mapping.get("kind") == "file" for mapping in source["path_mappings"])
    if file_checkout and any(mapping.get("kind", "directory") != "file" for mapping in source["path_mappings"]):
        raise ValueError(f"{source['source_name']} cannot mix file and directory path mappings.")
    sparse_mode = "--no-cone" if file_checkout else "--cone"
    _run_git(["-C", repository, "sparse-checkout", "init", sparse_mode])
    _run_git(["-C", repository, "sparse-checkout", "set", *[mapping["source"] for mapping in source["path_mappings"]]])
    _run_git(["-C", repository, "fetch", "--depth", "1", "--filter=blob:none", "origin", source["commit"]])
    _run_git(["-C", repository, "checkout", "--quiet", "--detach", "FETCH_HEAD"])

    resolved_commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "FETCH_HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if resolved_commit != source["commit"]:
        raise RuntimeError(f"Fetched {source['source_name']} commit {resolved_commit}, expected {source['commit']}.")

    target_root = staging_root / source["raw_path"]
    for mapping in source["path_mappings"]:
        source_path = repository / mapping["source"]
        expected_kind = mapping.get("kind", "directory")
        if (expected_kind == "directory" and not source_path.is_dir()) or (expected_kind == "file" and not source_path.is_file()):
            raise RuntimeError(f"Pinned {expected_kind} source path is missing: {source_path}")
        destination = target_root / mapping["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination)
        else:
            shutil.copy2(source_path, destination)


def fetch_text_source(source, staging_root, _work_root):
    destination = staging_root / source["raw_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with requests.get(
        source["url"],
        headers={"User-Agent": "ragops-control-plane-source-fetch/1.0"},
        stream=True,
        timeout=60,
    ) as response, destination.open("wb") as output:
        response.raise_for_status()
        for block in response.iter_content(chunk_size=1024 * 1024):
            if not block:
                continue
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != source["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {source['source_name']}: received {digest.hexdigest()}, expected {source['sha256']}.")


def _remove_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def fetch_sources(sources, project_root=PROJECT_ROOT, force=False, git_fetcher=fetch_git_source, text_fetcher=fetch_text_source):
    project_root = Path(project_root).resolve()
    final_roots = [project_root / source["install_root"] for source in sources]
    existing = [path for path in final_roots if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Source directories already exist: {joined}. Use --force to replace only these pinned source directories.")

    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ragops-source-fetch-", dir=data_root) as temporary:
        temporary = Path(temporary)
        staging_root = temporary / "stage"
        work_root = temporary / "work"
        staging_root.mkdir()
        work_root.mkdir()
        for source in sources:
            if source["source_type"] == "git_sparse_checkout":
                git_fetcher(source, staging_root, work_root)
            else:
                text_fetcher(source, staging_root, work_root)

        for source, final_root in zip(sources, final_roots, strict=True):
            staged_root = staging_root / source["install_root"]
            if not staged_root.exists():
                raise RuntimeError(f"Fetcher did not stage the expected source root: {staged_root}")
            if final_root.exists():
                _remove_path(final_root)
            final_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_root), str(final_root))
    return final_roots


def check_local_sources(sources, project_root=PROJECT_ROOT):
    project_root = Path(project_root).resolve()
    summaries = []
    for source in sources:
        if source["source_type"] == "git_sparse_checkout":
            root = project_root / source["raw_path"]
            destinations = [root / mapping["destination"] for mapping in source["path_mappings"]]
            missing = [
                path
                for path in destinations
                if not path.is_file() and (not path.is_dir() or not any(item.is_file() for item in path.rglob("*")))
            ]
            if missing:
                raise FileNotFoundError(f"Missing local paths for {source['source_name']}: {', '.join(map(str, missing))}")
            file_count = sum(
                1 if destination.is_file() else sum(1 for item in destination.rglob("*") if item.is_file())
                for destination in destinations
            )
        else:
            path = project_root / source["raw_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing local source: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source["sha256"]:
                raise ValueError(f"Local SHA256 mismatch for {source['source_name']}: {digest}")
            file_count = 1
        summaries.append((source["source_name"], file_count))
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-only", action="store_true", help="Validate the manifest without network access.")
    action.add_argument("--check-local", action="store_true", help="Validate already fetched source paths and checksums.")
    parser.add_argument("--force", action="store_true", help="Replace only the source directories declared by the manifest.")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = load_source_manifest(args.manifest, project_root=args.project_root)
    if args.validate_only:
        print(f"Valid source manifest: {len(sources)} pinned sources.")
        return
    if args.check_local:
        for name, file_count in check_local_sources(sources, project_root=args.project_root):
            print(f"{name}: {file_count} files")
        return
    installed = fetch_sources(sources, project_root=args.project_root, force=args.force)
    for path in installed:
        print(f"Installed {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Source fetch failed: {error}") from error
