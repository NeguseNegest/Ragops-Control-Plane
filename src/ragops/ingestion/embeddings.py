import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_cache = {}

def get_model(model_name=embedding_model_name):
    
    if model_name not in model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        model_cache[model_name] = SentenceTransformer(model_name)

    return model_cache[model_name]


def embed_texts(texts, model_name=embedding_model_name, batch_size=64, show_progress_bar=False):

    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True)

    return embeddings.tolist()