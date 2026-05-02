import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class RetrievedDoc:
    source: str
    domain: str
    score: float
    text: str


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class SupportRetriever:
    """Simple TF-IDF retriever for local support docs."""

    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.documents: List[Dict[str, str]] = []
        self._doc_vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._load_docs()
        self._build_index()

    def _load_docs(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory does not exist: {self.docs_dir}")

        supported_ext = {".txt", ".md", ".rst", ".html"}
        for path in sorted(self.docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in supported_ext:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            domain = self._infer_domain(path, text)
            self.documents.append(
                {
                    "source": str(path.name),
                    "domain": domain,
                    "text": text,
                }
            )

        if not self.documents:
            raise ValueError(f"No support documents found in: {self.docs_dir}")
        LOGGER.info("Loaded %d support documents.", len(self.documents))

    @staticmethod
    def _infer_domain(path: Path, text: str) -> str:
        haystack = f"{path.name} {text[:300]}".lower()
        if "hackerrank" in haystack:
            return "HackerRank Support"
        if "claude" in haystack or "anthropic" in haystack:
            return "Claude Help Center"
        if "visa" in haystack:
            return "Visa Support"
        return "General Support"

    def _build_index(self) -> None:
        tokens_per_doc = [tokenize(doc["text"]) for doc in self.documents]
        n_docs = len(tokens_per_doc)

        # Document frequencies for IDF
        doc_freq: Counter = Counter()
        for tokens in tokens_per_doc:
            doc_freq.update(set(tokens))

        self._idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()
        }
        self._doc_vectors = [self._to_tfidf_vector(tokens) for tokens in tokens_per_doc]
        LOGGER.info("Retriever index built with %d terms.", len(self._idf))

    def _to_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        n_tokens = len(tokens)
        return {
            term: (count / n_tokens) * self._idf.get(term, 0.0) for term, count in tf.items()
        }

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common_terms = set(a).intersection(b)
        numerator = sum(a[t] * b[t] for t in common_terms)
        a_norm = math.sqrt(sum(v * v for v in a.values()))
        b_norm = math.sqrt(sum(v * v for v in b.values()))
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return numerator / (a_norm * b_norm)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        q_tokens = tokenize(query)
        q_vec = self._to_tfidf_vector(q_tokens)
        if not q_vec:
            return []

        scored: List[Tuple[int, float]] = []
        for idx, d_vec in enumerate(self._doc_vectors):
            score = self._cosine_similarity(q_vec, d_vec)
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_scored = scored[:top_k]
        return [
            RetrievedDoc(
                source=self.documents[idx]["source"],
                domain=self.documents[idx]["domain"],
                score=score,
                text=self.documents[idx]["text"],
            )
            for idx, score in top_scored
        ]
