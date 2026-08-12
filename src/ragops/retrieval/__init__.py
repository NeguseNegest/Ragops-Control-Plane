"""Config-driven retrieval pipelines sharing one runtime interface."""

from ragops.retrieval.base import COMMON_RETRIEVER_INTERFACE, FunctionRetriever, Retriever

__all__ = ["COMMON_RETRIEVER_INTERFACE", "FunctionRetriever", "Retriever"]
