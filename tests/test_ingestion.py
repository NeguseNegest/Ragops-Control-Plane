import json
from pathlib import Path

from ragops.ingestion.loaders import is_supported_path, iter_documents, load_document
from ragops.schemas import DocumentChunk
from scripts.ingest import write_embedded_chunks


def write_raw_file(raw_root: Path, relative_path: str, text: str) -> Path:
    path = raw_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_file_loads_as_documentation(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(
        raw_root,
        "fastapi/docs/tutorial/example.md",
        "# FastAPI Tutorial\n\nCreate an app with FastAPI.",
    )

    document = load_document("fastapi/docs/tutorial/example.md", raw_root)

    assert document is not None
    assert "Create an app with FastAPI." in document.text
    assert document.metadata["extension"] == ".md"
    assert document.metadata["source_name"] == "fastapi"
    assert document.metadata["content_type"] == "documentation"
    assert document.metadata["relative_path"] == "fastapi/docs/tutorial/example.md"


def test_mdx_file_removes_frontmatter_and_imports(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(
        raw_root,
        "mlflow/docs/guide/index.mdx",
        "---\ntitle: Tracking\n---\n"
        "import Tabs from '@theme/Tabs';\n\n"
        "# Tracking\n\n"
        "Log metrics with MLflow.",
    )

    document = load_document("mlflow/docs/guide/index.mdx", raw_root)

    assert document is not None
    assert "title: Tracking" not in document.text
    assert "import Tabs" not in document.text
    assert "Log metrics with MLflow." in document.text
    assert document.metadata["extension"] == ".mdx"


def test_html_file_strips_tags_and_ignored_navigation(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(
        raw_root,
        "qdrant/docs/page.html",
        "<html><body><nav>Menu</nav><h1>Search</h1><p>Vector docs</p></body></html>",
    )

    document = load_document("qdrant/docs/page.html", raw_root)

    assert document is not None
    assert "Search" in document.text
    assert "Vector docs" in document.text
    assert "Menu" not in document.text
    assert "<h1>" not in document.text


def test_rst_and_txt_files_load_as_documentation(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(raw_root, "mlflow/docs/reference.rst", "MLflow\n======\n")
    write_raw_file(raw_root, "qdrant/qdrant_llms_full.txt", "Qdrant collections")

    rst_document = load_document("mlflow/docs/reference.rst", raw_root)
    txt_document = load_document("qdrant/qdrant_llms_full.txt", raw_root)

    assert rst_document is not None
    assert rst_document.metadata["content_type"] == "documentation"
    assert rst_document.metadata["extension"] == ".rst"

    assert txt_document is not None
    assert txt_document.metadata["content_type"] == "documentation"
    assert txt_document.metadata["extension"] == ".txt"


def test_python_files_only_load_from_fastapi_docs_src(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(
        raw_root,
        "fastapi/docs_src/tutorial/example.py",
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
    )
    write_raw_file(
        raw_root,
        "mlflow/docs/scripts/build.py",
        "print('build docs')\n",
    )

    fastapi_document = load_document("fastapi/docs_src/tutorial/example.py", raw_root)
    mlflow_document = load_document("mlflow/docs/scripts/build.py", raw_root)

    assert fastapi_document is not None
    assert fastapi_document.metadata["content_type"] == "code_example"
    assert fastapi_document.metadata["language"] == "python"
    assert "app = FastAPI()" in fastapi_document.text

    assert not is_supported_path("mlflow/docs/scripts/build.py", raw_root=".")
    assert mlflow_document is None


def test_unsupported_and_empty_files_are_skipped(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(raw_root, "fastapi/docs/logo.png", "not really an image")
    write_raw_file(raw_root, "fastapi/docs/empty.md", " \n\n ")

    assert not is_supported_path("fastapi/docs/logo.png", raw_root=".")
    assert load_document("fastapi/docs/logo.png", raw_root) is None
    assert load_document("fastapi/docs/empty.md", raw_root) is None


def test_iter_documents_skips_unsupported_and_empty_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(raw_root, "fastapi/docs/b.md", "# B")
    write_raw_file(raw_root, "fastapi/docs/a.md", "# A")
    write_raw_file(raw_root, "fastapi/docs/empty.md", " ")
    write_raw_file(raw_root, "fastapi/docs/image.svg", "<svg />")

    documents = list(iter_documents(raw_root))

    assert [document.metadata["relative_path"] for document in documents] == [
        "fastapi/docs/a.md",
        "fastapi/docs/b.md",
    ]


def test_document_ids_are_deterministic(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_raw_file(raw_root, "qdrant/docs/search.md", "# Search")

    first_document = load_document("qdrant/docs/search.md", raw_root)
    second_document = load_document("qdrant/docs/search.md", raw_root)

    assert first_document is not None
    assert second_document is not None
    assert first_document.document_id == second_document.document_id


def test_write_embedded_chunks_writes_jsonl_with_embedding_metadata(tmp_path: Path):
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="alpha beta",
            token_count=2,
            chunk_hash="hash-1",
            metadata={"relative_path": "docs/a.md"},
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="gamma delta",
            token_count=2,
            chunk_hash="hash-2",
            metadata={"relative_path": "docs/a.md"},
        ),
    ]

    def fake_embedder(texts, model_name, batch_size, show_progress_bar):
        assert texts == ["alpha beta", "gamma delta"]
        assert model_name == "fake-model"
        assert batch_size == 2
        assert show_progress_bar
        return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    output_path = tmp_path / "processed" / "chunks.jsonl"

    write_embedded_chunks(
        chunks,
        output_path,
        embedding_model="fake-model",
        embedding_batch_size=2,
        embedder=fake_embedder,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "chunk-1"
    assert rows[0]["embedding"] == [0.1, 0.2, 0.3]
    assert rows[0]["metadata"]["embedding_model"] == "fake-model"
    assert rows[0]["metadata"]["embedding_dimension"] == 3
