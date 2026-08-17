"""Basic Streamlit playground for the RAGOps query API."""

import os

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 5
REQUEST_TIMEOUT_SECONDS = 120


def get_api_url():
    """Return the configured FastAPI base URL.

    The dashboard uses RAGOPS_API_URL when it is set and otherwise connects to
    the local API. Removing the trailing slash keeps endpoint construction
    consistent.
    """
    api_url = os.getenv("RAGOPS_API_URL", DEFAULT_API_URL).strip()

    if not api_url:
        api_url = DEFAULT_API_URL

    return api_url.rstrip("/")


def query_api(query, top_k, api_url):
    """Send one question to POST /query and return its JSON response.

    The function owns HTTP communication so the Streamlit page only needs to
    handle a successful result or one readable RuntimeError.
    """
    endpoint = f"{api_url}/query"
    payload = {"query": query, "top_k": top_k}

    try:
        response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach the API at {api_url}. Make sure the FastAPI server is running.") from error

    try:
        result = response.json()
    except ValueError as error:
        if response.ok:
            message = "The API returned an invalid response."
        else:
            message = f"The API request failed with status {response.status_code}."
        raise RuntimeError(message) from error

    if not response.ok:
        detail = result.get("detail") if isinstance(result, dict) else None
        raise RuntimeError(str(detail or f"The API request failed with status {response.status_code}."))

    if not isinstance(result, dict):
        raise RuntimeError("The API returned an unexpected response.")

    return result


def render_citations(citations):
    """Render the source list returned with an answer.

    HTTP sources become links. Local corpus paths remain plain text so the page
    does not present them as public URLs.
    """
    st.subheader("Citations")

    if not citations:
        st.caption("No citations were returned.")
        return

    for citation in citations:
        citation_id = citation.get("citation_id") or "[?]"
        title = citation.get("title") or citation.get("document_id") or "Unknown source"
        url = citation.get("url")

        if isinstance(url, str) and url.startswith(("http://", "https://")):
            st.markdown(f"{citation_id} [{title}]({url})")
        elif url:
            st.write(f"{citation_id} {title} — {url}")
        else:
            st.write(f"{citation_id} {title}")


def render_chunks(chunks):
    """Render retrieved evidence in collapsed sections.

    Each section exposes the chunk text, rank, score, source, and the small
    amount of provenance needed to understand why it was retrieved.
    """
    st.subheader("Retrieved Chunks")

    if not chunks:
        st.caption("No chunks were returned.")
        return

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        rank = chunk.get("rank", "?")
        score = chunk.get("score")
        source = chunk.get("source_url") or metadata.get("relative_path") or metadata.get("source_path") or "Unknown source"

        try:
            score_text = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score_text = "n/a"

        with st.expander(f"#{rank} · score {score_text} · {source}"):
            st.write(chunk.get("text") or "No chunk text was returned.")

            details = []
            if metadata.get("heading"):
                details.append(f"Heading: {metadata['heading']}")
            if metadata.get("relative_path"):
                details.append(f"Path: {metadata['relative_path']}")
            if chunk.get("chunk_id"):
                details.append(f"Chunk ID: {chunk['chunk_id']}")
            if details:
                st.caption(" · ".join(details))


def render_result(result):
    """Render the answer, latency, citations, and retrieved evidence.

    Missing optional lists are treated as empty so one incomplete response does
    not prevent the rest of the result from being displayed.
    """
    st.subheader("Answer")
    st.write(result.get("answer") or "No answer was returned.")

    latency_ms = result.get("latency_ms")
    try:
        latency_text = f"{float(latency_ms):.1f} ms"
    except (TypeError, ValueError):
        latency_text = "n/a"
    st.metric("Latency", latency_text)

    render_citations(result.get("citations") or [])
    render_chunks(result.get("chunks") or [])


def main():
    """Build the complete Day 13 query playground.

    The page collects one question, validates it, calls the existing FastAPI
    endpoint, and displays the grounded response. Routing and compact engineering
    analytics belong to the later Day 49 dashboard pass; caching, feedback, and
    canary views are explicitly outside the required scope.
    """
    st.set_page_config(page_title="RAGOps Control Plane", layout="wide")
    st.title("RAGOps Control Plane")
    st.caption("Ask a question about the indexed FastAPI, MLflow, and Qdrant documentation.")

    with st.form("query_form"):
        query = st.text_area("Question", placeholder="How do I create a FastAPI app?")
        top_k = st.number_input("Retrieved chunks", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return

    query = query.strip()
    if not query:
        st.warning("Enter a question before submitting.")
        return

    try:
        with st.spinner("Searching the documentation..."):
            result = query_api(query, int(top_k), get_api_url())
    except RuntimeError as error:
        st.error(str(error))
        return

    render_result(result)


if __name__ == "__main__":
    main()
