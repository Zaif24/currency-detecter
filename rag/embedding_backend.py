"""
Pluggable embedding backend.

Tries sentence-transformers/all-MiniLM-L6-v2 first (dense semantic embeddings,
384-dim). If model weights cannot be downloaded (no internet / restricted
network), transparently falls back to a TF-IDF vectorizer (scikit-learn) so
the RAG pipeline still runs end-to-end. Streamlit Cloud deployments have
outbound internet access, so the primary backend is expected to load there.
"""
import os
import pickle


class SentenceTransformerBackend:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)

    def fit_encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def save(self, store_dir):
        pass  # no local state to persist; model re-downloads/caches via HF

    @classmethod
    def load(cls, store_dir):
        return cls()


class TfidfBackend:
    """Fallback: TF-IDF + SVD to get fixed-size dense vectors comparable via cosine sim."""
    name = "tfidf-svd-256 (offline fallback)"

    def __init__(self, n_components=256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.n_components = n_components
        self.svd = None

    def fit_encode(self, texts):
        from sklearn.decomposition import TruncatedSVD
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense = self.svd.fit_transform(tfidf)
        return dense

    def encode(self, texts):
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)

    def save(self, store_dir):
        with open(os.path.join(store_dir, "tfidf_backend.pkl"), "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    @classmethod
    def load(cls, store_dir):
        obj = cls()
        with open(os.path.join(store_dir, "tfidf_backend.pkl"), "rb") as f:
            data = pickle.load(f)
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        return obj


def get_embedder():
    """Used during ingestion (fit_encode)."""
    try:
        return SentenceTransformerBackend()
    except Exception as e:
        print(f"[embedding_backend] Falling back to TF-IDF ({e.__class__.__name__}: {e})")
        return TfidfBackend()


def load_embedder_for_query(store_dir, backend_name):
    """Used at query time (encode only), matching whichever backend built the index."""
    if backend_name.startswith("sentence-transformers"):
        return SentenceTransformerBackend.load(store_dir)
    return TfidfBackend.load(store_dir)
