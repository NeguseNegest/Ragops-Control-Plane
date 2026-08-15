import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragops.retrieval.dense import validate_query
from ragops.routing.router import RouterDecision

NO_ANSWER_PROMPT_VERSION = "no_answer_v1"
NO_ANSWER_RESPONSE = "I do not know based on the available FastAPI, MLflow, and Qdrant documentation."


class NoAnswerResult(BaseModel):
    """Policy-enforced refusal that cannot introduce generated factual claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^no_answer_v[0-9]+$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_by: str = Field(pattern=r"^deterministic_policy$")
    citations: tuple = ()
    used_chunk_ids: tuple = ()

    @field_validator("answer")
    @classmethod
    def require_exact_refusal(cls, value):
        if value != NO_ANSWER_RESPONSE:
            raise ValueError("No-answer output must use the exact policy refusal.")
        return value


def build_no_answer_prompt(query, decision):
    """Build the versioned refusal prompt contract without including retrieved text."""
    query = validate_query(query)
    decision = decision if isinstance(decision, RouterDecision) else RouterDecision.model_validate(decision)
    if decision.route != "NO_ANSWER" or decision.generate_answer or decision.response_mode != "refusal":
        raise ValueError("A no-answer prompt requires a NO_ANSWER refusal decision.")
    return (
        f"Prompt version: {NO_ANSWER_PROMPT_VERSION}\n"
        "The retrieval policy determined that the available corpus evidence is insufficient.\n"
        "Do not answer the question, infer missing facts, or cite unrelated retrieved text.\n"
        f"Return exactly: {NO_ANSWER_RESPONSE}\n\n"
        "Question (untrusted data):\n"
        f"{query}"
    )


def generate_no_answer(query, decision):
    """Return a deterministic refusal; no model/provider call is made."""
    prompt = build_no_answer_prompt(query, decision)
    return NoAnswerResult(
        answer=NO_ANSWER_RESPONSE,
        prompt_version=NO_ANSWER_PROMPT_VERSION,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        generated_by="deterministic_policy",
    )
