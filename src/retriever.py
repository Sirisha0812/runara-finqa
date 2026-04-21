"""Document-local hybrid retrieval for FinQA evidence chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from src.config import config
from src.logger import LoggerContext, get_logger

logger = get_logger(__name__)


@dataclass
class EvidenceChunk:
    """A single retrievable span from a financial document."""

    chunk_id: str
    chunk_type: str
    text: str
    metadata: Dict[str, Any]


class FinQARetriever:
    """Hybrid retriever over chunks from the current FinQA document."""

    def __init__(self) -> None:
        self.embedding_model_name = config.vector_store.embedding_model
        self.embedding_model: SentenceTransformer | None = None

        logger.info(
            "retriever_initialized",
            retrieval_scope="document_local",
            embedding_model=self.embedding_model_name,
            dense_available=SentenceTransformer is not None,
            sparse_available=BM25Okapi is not None,
        )

    def _load_embedding_model(self) -> None:
        if SentenceTransformer is None:
            return
        if self.embedding_model is None:
            with LoggerContext(
                logger,
                "load_embedding_model",
                model=self.embedding_model_name,
            ):
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9\.\-%]+", text.lower())

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def build_chunks(self, example: Dict[str, Any]) -> List[EvidenceChunk]:
        """Split a FinQA example into evidence chunks for retrieval."""
        chunks: List[EvidenceChunk] = []

        for idx, paragraph in enumerate(example.get("pre_text", []) or []):
            text = self._normalize_whitespace(str(paragraph))
            if text:
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"pre_{idx}",
                        chunk_type="pre_text",
                        text=text,
                        metadata={"position": idx},
                    )
                )

        table = example.get("table", []) or []
        if table:
            headers = [self._normalize_whitespace(str(cell)) for cell in table[0]]
            if headers:
                header_text = " | ".join(headers)
                chunks.append(
                    EvidenceChunk(
                        chunk_id="table_header",
                        chunk_type="table_header",
                        text=f"Table columns: {header_text}",
                        metadata={"position": 0},
                    )
                )

            for row_idx, row in enumerate(table[1:], start=1):
                cells = [self._normalize_whitespace(str(cell)) for cell in row]
                pair_text = "; ".join(
                    f"{header}: {value}"
                    for header, value in zip(headers, cells)
                    if value
                )
                if pair_text:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"table_row_{row_idx}",
                            chunk_type="table_row",
                            text=pair_text,
                            metadata={"position": row_idx},
                        )
                    )

        for idx, paragraph in enumerate(example.get("post_text", []) or []):
            text = self._normalize_whitespace(str(paragraph))
            if text:
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"post_{idx}",
                        chunk_type="post_text",
                        text=text,
                        metadata={"position": idx},
                    )
                )

        logger.info(
            "document_chunked",
            example_id=example.get("id"),
            num_chunks=len(chunks),
        )
        return chunks

    def retrieve_for_example(
        self,
        question: str,
        example: Dict[str, Any],
        k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k evidence chunks from the current document.

        This is the retrieval path used for answer generation and evaluation.
        """
        chunks = self.build_chunks(example)
        if not chunks:
            return []

        k = k or config.agent.retrieval_top_k
        dense_weight = config.agent.dense_weight if dense_weight is None else dense_weight
        sparse_weight = config.agent.sparse_weight if sparse_weight is None else sparse_weight

        with LoggerContext(
            logger,
            "retrieve_for_example",
            question=question,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        ):
            chunk_texts = [chunk.text for chunk in chunks]
            self._load_embedding_model()

            if self.embedding_model is not None:
                dense_inputs = [question] + chunk_texts
                embeddings = self.embedding_model.encode(
                    dense_inputs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                query_embedding = embeddings[0]
                chunk_embeddings = embeddings[1:]
                dense_scores = np.dot(chunk_embeddings, query_embedding)
                dense_scores_norm = (dense_scores + 1.0) / 2.0
            else:
                dense_scores_norm = np.zeros(len(chunk_texts))

            if BM25Okapi is not None:
                tokenized_chunks = [self._tokenize(text) for text in chunk_texts]
                bm25 = BM25Okapi(tokenized_chunks)
                sparse_scores = bm25.get_scores(self._tokenize(question))
                sparse_max = float(np.max(sparse_scores)) if len(sparse_scores) else 0.0
                sparse_scores_norm = sparse_scores / sparse_max if sparse_max > 0 else sparse_scores
            else:
                sparse_scores_norm = np.array(
                    [
                        self._token_overlap_score(self._tokenize(question), self._tokenize(text))
                        for text in chunk_texts
                    ]
                )

            results: List[Dict[str, Any]] = []
            for idx, chunk in enumerate(chunks):
                hybrid_score = (dense_weight * float(dense_scores_norm[idx])) + (
                    sparse_weight * float(sparse_scores_norm[idx])
                )
                results.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "dense_score": float(dense_scores_norm[idx]),
                        "sparse_score": float(sparse_scores_norm[idx]),
                        "hybrid_score": hybrid_score,
                    }
                )

            ranked = sorted(results, key=lambda item: item["hybrid_score"], reverse=True)[:k]
            logger.info(
                "document_retrieval_completed",
                question=question,
                num_chunks=len(chunks),
                num_results=len(ranked),
                top_chunk_id=ranked[0]["chunk_id"] if ranked else None,
                top_hybrid_score=ranked[0]["hybrid_score"] if ranked else None,
            )
            return ranked

    @staticmethod
    def _token_overlap_score(query_tokens: Sequence[str], chunk_tokens: Sequence[str]) -> float:
        if not query_tokens or not chunk_tokens:
            return 0.0
        query_set = set(query_tokens)
        chunk_set = set(chunk_tokens)
        return len(query_set & chunk_set) / max(len(query_set), 1)

    def answer_in_top_k(self, example: Dict[str, Any], question: str, k: int = 5) -> bool:
        """Heuristic retrieval metric: whether a normalized gold answer appears in top-k chunks."""
        gold = self._normalize_whitespace(str(example.get("exe_ans") or example.get("answer") or ""))
        if not gold:
            return False

        normalized_gold = gold.lower().replace(",", "")
        retrieved = self.retrieve_for_example(question=question, example=example, k=k)
        for chunk in retrieved:
            chunk_text = chunk["text"].lower().replace(",", "")
            if normalized_gold in chunk_text:
                return True
        return False


if __name__ == "__main__":
    from src.data_loader import load_finqa_dataset

    dataset = load_finqa_dataset(split="validation")
    example = dataset[0]

    retriever = FinQARetriever()
    results = retriever.retrieve_for_example(example["question"], example, k=5)

    print(f"Question: {example['question']}\n")
    for idx, item in enumerate(results, start=1):
        print(
            f"{idx}. {item['chunk_id']} [{item['chunk_type']}] "
            f"hybrid={item['hybrid_score']:.3f}"
        )
        print(f"   {item['text'][:200]}")
