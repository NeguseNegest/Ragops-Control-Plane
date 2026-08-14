import json
import math
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragops.pipeline_registry import PipelineVersion
from ragops.tracing.context import COMPONENT_TIMING_FIELDS, ComponentLatencies

TRACE_SCHEMA_VERSION = 3
DEFAULT_TRACE_DB_PATH = Path("data/traces/ragops_traces.sqlite3")
TRACE_TABLES = frozenset({"traces", "retrieved_chunks", "feedback"})

_TRACE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    endpoint TEXT NOT NULL CHECK (endpoint IN ('retrieve', 'query')),
    query TEXT NOT NULL,
    requested_top_k INTEGER NOT NULL CHECK (requested_top_k > 0),
    pipeline_name TEXT NOT NULL CHECK (length(trim(pipeline_name)) > 0),
    pipeline_version TEXT NOT NULL CHECK (length(trim(pipeline_version)) > 0),
    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
    retrieved_chunk_count INTEGER NOT NULL CHECK (retrieved_chunk_count >= 0),
    answer TEXT,
    total_latency_ms REAL NOT NULL CHECK (total_latency_ms >= 0),
    embedding_ms REAL CHECK (embedding_ms IS NULL OR embedding_ms >= 0),
    dense_ms REAL CHECK (dense_ms IS NULL OR dense_ms >= 0),
    bm25_ms REAL CHECK (bm25_ms IS NULL OR bm25_ms >= 0),
    fusion_ms REAL CHECK (fusion_ms IS NULL OR fusion_ms >= 0),
    reranker_ms REAL CHECK (reranker_ms IS NULL OR reranker_ms >= 0),
    generation_ms REAL CHECK (generation_ms IS NULL OR generation_ms >= 0),
    error_type TEXT,
    error_message TEXT,
    CHECK (
        (status = 'success' AND error_type IS NULL AND error_message IS NULL)
        OR
        (status = 'error' AND error_type IS NOT NULL AND error_message IS NOT NULL)
    ),
    CHECK (endpoint = 'query' OR answer IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_status_created_at ON traces(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_pipeline ON traces(pipeline_name, pipeline_version, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieved_chunks (
    trace_id TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    chunk_id TEXT NOT NULL CHECK (length(trim(chunk_id)) > 0),
    document_id TEXT NOT NULL CHECK (length(trim(document_id)) > 0),
    text TEXT NOT NULL CHECK (length(text) > 0),
    score REAL NOT NULL,
    source_url TEXT,
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
    used_for_generation INTEGER NOT NULL DEFAULT 0 CHECK (used_for_generation IN (0, 1)),
    PRIMARY KEY (trace_id, rank),
    UNIQUE (trace_id, chunk_id),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_chunk_id ON retrieved_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_document_id ON retrieved_chunks(document_id);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rating INTEGER CHECK (rating IN (-1, 1)),
    comment TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata_json)),
    CHECK (rating IS NOT NULL OR (comment IS NOT NULL AND length(trim(comment)) > 0)),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_trace_created_at ON feedback(trace_id, created_at DESC);
"""

_MIGRATE_1_TO_2_SQL = """
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE;

CREATE TABLE traces_new (
    trace_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    endpoint TEXT NOT NULL CHECK (endpoint IN ('retrieve', 'query')),
    query TEXT NOT NULL,
    requested_top_k INTEGER NOT NULL CHECK (requested_top_k > 0),
    pipeline_name TEXT NOT NULL CHECK (length(trim(pipeline_name)) > 0),
    pipeline_version TEXT NOT NULL CHECK (length(trim(pipeline_version)) > 0),
    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
    retrieved_chunk_count INTEGER NOT NULL CHECK (retrieved_chunk_count >= 0),
    answer TEXT,
    total_latency_ms REAL NOT NULL CHECK (total_latency_ms >= 0),
    error_type TEXT,
    error_message TEXT,
    CHECK (
        (status = 'success' AND error_type IS NULL AND error_message IS NULL)
        OR
        (status = 'error' AND error_type IS NOT NULL AND error_message IS NOT NULL)
    ),
    CHECK (endpoint = 'query' OR answer IS NULL)
);

INSERT INTO traces_new (
    trace_id, created_at, completed_at, endpoint, query, requested_top_k,
    pipeline_name, pipeline_version, status, retrieved_chunk_count,
    answer, total_latency_ms, error_type, error_message
)
SELECT
    trace_id, created_at, completed_at, endpoint, query, requested_top_k,
    pipeline_name, pipeline_version, status, retrieved_chunk_count,
    answer, total_latency_ms, error_type, error_message
FROM traces;

DROP TABLE traces;
ALTER TABLE traces_new RENAME TO traces;
CREATE INDEX idx_traces_created_at ON traces(created_at DESC);
CREATE INDEX idx_traces_status_created_at ON traces(status, created_at DESC);
CREATE INDEX idx_traces_pipeline ON traces(pipeline_name, pipeline_version, created_at DESC);
PRAGMA user_version = 2;

COMMIT;
PRAGMA foreign_keys = ON;
"""

_MIGRATE_2_TO_3_SQL = """
BEGIN IMMEDIATE;

ALTER TABLE traces ADD COLUMN embedding_ms REAL CHECK (embedding_ms IS NULL OR embedding_ms >= 0);
ALTER TABLE traces ADD COLUMN dense_ms REAL CHECK (dense_ms IS NULL OR dense_ms >= 0);
ALTER TABLE traces ADD COLUMN bm25_ms REAL CHECK (bm25_ms IS NULL OR bm25_ms >= 0);
ALTER TABLE traces ADD COLUMN fusion_ms REAL CHECK (fusion_ms IS NULL OR fusion_ms >= 0);
ALTER TABLE traces ADD COLUMN reranker_ms REAL CHECK (reranker_ms IS NULL OR reranker_ms >= 0);
ALTER TABLE traces ADD COLUMN generation_ms REAL CHECK (generation_ms IS NULL OR generation_ms >= 0);
PRAGMA user_version = 3;

COMMIT;
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineIdentity(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: PipelineVersion

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        return value.strip()


class TraceRecord(StrictModel):
    trace_id: str
    created_at: datetime
    completed_at: datetime
    endpoint: Literal["retrieve", "query"]
    query: str
    requested_top_k: int = Field(gt=0)
    pipeline_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    pipeline_version: PipelineVersion
    status: Literal["success", "error"]
    retrieved_chunk_count: int = Field(ge=0)
    answer: str | None = None
    total_latency_ms: float = Field(ge=0)
    embedding_ms: float | None = Field(default=None, ge=0)
    dense_ms: float | None = Field(default=None, ge=0)
    bm25_ms: float | None = Field(default=None, ge=0)
    fusion_ms: float | None = Field(default=None, ge=0)
    reranker_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value):
        value = value.strip()
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError("trace_id must be a UUID.") from error
        if str(parsed) != value.lower():
            raise ValueError("trace_id must use canonical UUID form.")
        return value.lower()

    @field_validator("pipeline_name", "answer", "error_type", "error_message")
    @classmethod
    def clean_text(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Trace text fields must not be empty when provided.")
        return value

    @field_validator("created_at", "completed_at")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Trace timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("total_latency_ms", *COMPONENT_TIMING_FIELDS)
    @classmethod
    def validate_finite_latency(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Trace latencies must be finite.")
        return value

    @model_validator(mode="after")
    def validate_result(self):
        if self.completed_at < self.created_at:
            raise ValueError("Trace completion time must not precede creation time.")
        if self.status == "success" and (self.error_type is not None or self.error_message is not None):
            raise ValueError("Successful traces must not contain error fields.")
        if self.status == "error" and (self.error_type is None or self.error_message is None):
            raise ValueError("Error traces must contain error type and message.")
        if self.endpoint == "retrieve" and self.answer is not None:
            raise ValueError("Retrieve traces must not contain a generated answer.")
        if self.endpoint == "retrieve" and self.generation_ms is not None:
            raise ValueError("Retrieve traces must not contain generation latency.")
        return self

    def component_latencies(self):
        """Return the validated component-latency response/storage view."""
        return ComponentLatencies.model_validate({field: getattr(self, field) for field in COMPONENT_TIMING_FIELDS})


class RetrievedChunkTrace(StrictModel):
    rank: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float
    source_url: str | None = None
    metadata: dict = Field(default_factory=dict)
    used_for_generation: bool = False

    @field_validator("chunk_id", "document_id", "text", "source_url")
    @classmethod
    def clean_text(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Retrieved chunk text fields must not be empty when provided.")
        return value

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value):
        if not math.isfinite(value):
            raise ValueError("Retrieved chunk score must be finite.")
        return value


class FeedbackRecord(StrictModel):
    feedback_id: str
    trace_id: str
    created_at: datetime
    rating: Literal[-1, 1] | None = None
    comment: str | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("feedback_id", "trace_id")
    @classmethod
    def validate_uuid(cls, value):
        value = value.strip()
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError("Feedback and trace IDs must be UUIDs.") from error
        if str(parsed) != value.lower():
            raise ValueError("Feedback and trace IDs must use canonical UUID form.")
        return value.lower()

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Feedback timestamp must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("comment")
    @classmethod
    def clean_comment(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Feedback comment must not be empty when provided.")
        return value

    @model_validator(mode="after")
    def require_feedback_value(self):
        if self.rating is None and self.comment is None:
            raise ValueError("Feedback requires a rating or comment.")
        return self


def configured_trace_db_path(path=None, project_root=None):
    """Resolve an explicit path, environment override, or project-local default."""
    if path is None:
        environment_path = os.getenv("RAGOPS_TRACE_DB_PATH")
        path = environment_path.strip() if environment_path and environment_path.strip() else DEFAULT_TRACE_DB_PATH
    path = Path(path)
    if path.is_absolute():
        return path
    project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
    return (project_root / path).resolve()


def configured_pipeline_identity(name=None, version=None):
    """Resolve the explicitly deployed pipeline identity for trace provenance."""
    name = name if name is not None else os.getenv("RAGOPS_PIPELINE_NAME", "dense_baseline")
    version = version if version is not None else os.getenv("RAGOPS_PIPELINE_VERSION", "1.0.0")
    return PipelineIdentity(name=name, version=version)


def utc_now():
    return datetime.now(UTC)


def _json_text(payload, label):
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-serializable and finite.") from error


class TraceStore:
    """SQLite-backed trace, retrieved-evidence, and feedback repository."""

    def __init__(self, path=DEFAULT_TRACE_DB_PATH, timeout_seconds=5.0):
        self.path = Path(path).resolve()
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("SQLite timeout must be a positive finite number.")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("SQLite timeout must be a positive finite number.")

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        return connection

    def initialize(self):
        """Create the current schema or migrate and validate a compatible database."""
        if self.path.exists() and not self.path.is_file():
            raise ValueError(f"Trace database path is not a file: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current_version > TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"Trace database schema version {current_version} is newer than supported version {TRACE_SCHEMA_VERSION}."
                )
            if current_version == 0:
                connection.executescript(_TRACE_SCHEMA_SQL)
                connection.execute(f"PRAGMA user_version = {TRACE_SCHEMA_VERSION}")
                current_version = TRACE_SCHEMA_VERSION
            if current_version == 1:
                connection.executescript(_MIGRATE_1_TO_2_SQL)
                current_version = 2
            if current_version == 2:
                connection.executescript(_MIGRATE_2_TO_3_SQL)
                current_version = 3
            if current_version < TRACE_SCHEMA_VERSION:
                raise ValueError(f"No migration is available from trace schema {current_version} to {TRACE_SCHEMA_VERSION}.")
        self.validate_schema()
        return self

    def validate_schema(self):
        """Require the expected user version, tables, columns, and valid foreign keys."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Trace database does not exist: {self.path}")
        expected_columns = {
            "traces": {
                "trace_id",
                "created_at",
                "completed_at",
                "endpoint",
                "query",
                "requested_top_k",
                "pipeline_name",
                "pipeline_version",
                "status",
                "retrieved_chunk_count",
                "answer",
                "total_latency_ms",
                *COMPONENT_TIMING_FIELDS,
                "error_type",
                "error_message",
            },
            "retrieved_chunks": {
                "trace_id",
                "rank",
                "chunk_id",
                "document_id",
                "text",
                "score",
                "source_url",
                "metadata_json",
                "used_for_generation",
            },
            "feedback": {"feedback_id", "trace_id", "created_at", "rating", "comment", "metadata_json"},
        }
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != TRACE_SCHEMA_VERSION:
                raise ValueError(f"Trace database schema version is {version}; expected {TRACE_SCHEMA_VERSION}.")
            tables = {
                row["name"]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            missing_tables = sorted(TRACE_TABLES - tables)
            if missing_tables:
                raise ValueError(f"Trace database is missing tables: {missing_tables}.")
            for table, expected in expected_columns.items():
                columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
                if columns != expected:
                    raise ValueError(f"Trace database table {table!r} has unexpected columns.")
            for table in ("retrieved_chunks", "feedback"):
                foreign_keys = connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
                if not any(
                    row["table"] == "traces"
                    and row["from"] == "trace_id"
                    and row["to"] == "trace_id"
                    and row["on_delete"].upper() == "CASCADE"
                    for row in foreign_keys
                ):
                    raise ValueError(f"Trace database table {table!r} is missing its cascading trace foreign key.")
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise ValueError("Trace database contains invalid foreign-key references.")
        return True

    def record_trace(self, trace, chunks=()):
        """Atomically persist one completed trace and its ordered retrieved chunks."""
        trace = trace if isinstance(trace, TraceRecord) else TraceRecord.model_validate(trace)
        chunks = [chunk if isinstance(chunk, RetrievedChunkTrace) else RetrievedChunkTrace.model_validate(chunk) for chunk in chunks]
        expected_ranks = list(range(1, len(chunks) + 1))
        if [chunk.rank for chunk in chunks] != expected_ranks:
            raise ValueError("Retrieved trace chunks must have contiguous one-based ranks in stored order.")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Retrieved trace chunks must have unique chunk IDs.")
        if trace.retrieved_chunk_count != len(chunks):
            raise ValueError("Trace retrieved_chunk_count must match the stored chunks.")

        trace_values = (
            trace.trace_id,
            trace.created_at.isoformat(),
            trace.completed_at.isoformat(),
            trace.endpoint,
            trace.query,
            trace.requested_top_k,
            trace.pipeline_name,
            trace.pipeline_version,
            trace.status,
            trace.retrieved_chunk_count,
            trace.answer,
            trace.total_latency_ms,
            trace.embedding_ms,
            trace.dense_ms,
            trace.bm25_ms,
            trace.fusion_ms,
            trace.reranker_ms,
            trace.generation_ms,
            trace.error_type,
            trace.error_message,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces (
                    trace_id, created_at, completed_at, endpoint, query, requested_top_k,
                    pipeline_name, pipeline_version, status, retrieved_chunk_count,
                    answer, total_latency_ms, embedding_ms, dense_ms, bm25_ms,
                    fusion_ms, reranker_ms, generation_ms, error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                trace_values,
            )
            connection.executemany(
                """
                INSERT INTO retrieved_chunks (
                    trace_id, rank, chunk_id, document_id, text, score,
                    source_url, metadata_json, used_for_generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        chunk.rank,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.text,
                        chunk.score,
                        chunk.source_url,
                        _json_text(chunk.metadata, "Retrieved chunk metadata"),
                        int(chunk.used_for_generation),
                    )
                    for chunk in chunks
                ],
            )
        return trace.trace_id

    def record_feedback(self, feedback):
        """Persist feedback for an existing trace, enforcing the foreign key."""
        feedback = feedback if isinstance(feedback, FeedbackRecord) else FeedbackRecord.model_validate(feedback)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback (feedback_id, trace_id, created_at, rating, comment, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.feedback_id,
                        feedback.trace_id,
                        feedback.created_at.isoformat(),
                        feedback.rating,
                        feedback.comment,
                        _json_text(feedback.metadata, "Feedback metadata"),
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "FOREIGN KEY" in str(error):
                raise ValueError(f"Feedback trace does not exist: {feedback.trace_id}") from error
            raise
        return feedback.feedback_id

    def get_trace(self, trace_id):
        """Return one trace by ID, or None when it does not exist."""
        trace_id = str(trace_id).strip().lower()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return TraceRecord.model_validate(dict(row)) if row is not None else None

    def list_traces(self, limit=100):
        """Return the newest traces with a bounded result size."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Trace list limit must be an integer from 1 through 1000.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM traces ORDER BY created_at DESC, trace_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [TraceRecord.model_validate(dict(row)) for row in rows]

    def list_retrieved_chunks(self, trace_id):
        """Return stored chunks in their original rank order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT rank, chunk_id, document_id, text, score, source_url, metadata_json, used_for_generation "
                "FROM retrieved_chunks WHERE trace_id = ? ORDER BY rank",
                (str(trace_id).strip().lower(),),
            ).fetchall()
        return [
            RetrievedChunkTrace(
                rank=row["rank"],
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                score=row["score"],
                source_url=row["source_url"],
                metadata=json.loads(row["metadata_json"]),
                used_for_generation=bool(row["used_for_generation"]),
            )
            for row in rows
        ]

    def list_feedback(self, trace_id):
        """Return trace feedback in creation order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_id, trace_id, created_at, rating, comment, metadata_json "
                "FROM feedback WHERE trace_id = ? ORDER BY created_at, feedback_id",
                (str(trace_id).strip().lower(),),
            ).fetchall()
        return [
            FeedbackRecord(
                feedback_id=row["feedback_id"],
                trace_id=row["trace_id"],
                created_at=row["created_at"],
                rating=row["rating"],
                comment=row["comment"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def counts(self):
        """Return row counts for operational validation."""
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in sorted(TRACE_TABLES)
            }
