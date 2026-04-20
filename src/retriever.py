"""Hybrid retriever (FAISS + BM25) for FinQA dataset."""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import config
from src.data_loader import load_finqa_dataset, prepare_example_for_rag
from src.logger import LoggerContext, get_logger

logger = get_logger(__name__)


class FinQARetriever:
    """Hybrid retriever combining FAISS (dense) and BM25 (sparse) for FinQA dataset."""

    def __init__(self, index_path: str = "./data/faiss_index"):
        """
        Initialize the FinQA hybrid retriever.

        Args:
            index_path: Path to save/load FAISS index and metadata
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model_name = config.vector_store.embedding_model
        self.embedding_model = None
        self.faiss_index = None
        self.bm25_index = None
        self.documents = []
        self.tokenized_contexts = []
        self.dimension = None

        logger.info(
            "retriever_initialized",
            index_path=str(self.index_path),
            embedding_model=self.embedding_model_name,
            retrieval_type="hybrid_faiss_bm25",
        )

    def _load_embedding_model(self) -> None:
        """Load the sentence transformer embedding model."""
        if self.embedding_model is None:
            with LoggerContext(
                logger, "load_embedding_model", model=self.embedding_model_name
            ):
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(
                    "embedding_model_loaded",
                    model=self.embedding_model_name,
                    dimension=self.dimension,
                )

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 (simple whitespace + lowercase).

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split on whitespace, filter short tokens
        return [token.lower() for token in text.split() if len(token) > 1]

    def build_index(self) -> None:
        """Build both FAISS and BM25 indices from train set."""
        with LoggerContext(logger, "build_index"):
            # Load embedding model
            self._load_embedding_model()

            # Load dataset
            dataset = load_finqa_dataset(split="train")
            logger.info("dataset_loaded_for_indexing", num_examples=len(dataset))

            # Prepare all examples
            self.documents = []
            contexts = []
            self.tokenized_contexts = []

            logger.info("preparing_examples_for_embedding", total=len(dataset))

            for idx, example in enumerate(dataset):
                prepared = prepare_example_for_rag(example)

                # Store document with metadata
                doc = {
                    "question": prepared["question"],
                    "answer": prepared["answer"],
                    "program": prepared["program"],
                    "context": prepared["context"],
                    "table_str": prepared["table_str"],
                    "raw_table": prepared["raw_table"],
                    "doc_id": idx,
                }
                self.documents.append(doc)
                contexts.append(prepared["context"])

                # Tokenize for BM25
                self.tokenized_contexts.append(self._tokenize_for_bm25(prepared["context"]))

                # Log progress every 1000 documents
                if (idx + 1) % 1000 == 0:
                    logger.info("embedding_progress", processed=idx + 1, total=len(dataset))

            logger.info("examples_prepared", total=len(self.documents))

            # Build FAISS index
            with LoggerContext(logger, "embed_contexts", num_contexts=len(contexts)):
                embeddings = self.embedding_model.encode(
                    contexts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                logger.info(
                    "embeddings_created",
                    shape=embeddings.shape,
                    dtype=str(embeddings.dtype),
                )

            with LoggerContext(logger, "build_faiss_index", dimension=self.dimension):
                # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                self.faiss_index.add(embeddings)
                logger.info(
                    "faiss_index_built",
                    total_vectors=self.faiss_index.ntotal,
                    dimension=self.dimension,
                )

            # Build BM25 index
            with LoggerContext(logger, "build_bm25_index", num_documents=len(self.tokenized_contexts)):
                self.bm25_index = BM25Okapi(self.tokenized_contexts)
                logger.info(
                    "bm25_index_built",
                    num_documents=len(self.tokenized_contexts),
                )

            # Save the indices
            self.save_index()

    def save_index(self) -> None:
        """Save FAISS index, BM25 index, and metadata to disk."""
        if self.faiss_index is None or self.bm25_index is None:
            logger.warning("save_index_skipped", reason="indices_not_built")
            return

        with LoggerContext(logger, "save_index", path=str(self.index_path)):
            # Save FAISS index
            faiss_file = self.index_path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_file))
            logger.info("faiss_index_saved", file=str(faiss_file))

            # Save BM25 index (pickle the BM25Okapi object)
            bm25_file = self.index_path / "bm25.index"
            with open(bm25_file, "wb") as f:
                pickle.dump(self.bm25_index, f)
            logger.info("bm25_index_saved", file=str(bm25_file))

            # Save metadata (documents + tokenized contexts)
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "tokenized_contexts": self.tokenized_contexts,
                        "dimension": self.dimension,
                        "embedding_model": self.embedding_model_name,
                    },
                    f,
                )
            logger.info(
                "metadata_saved",
                file=str(metadata_file),
                num_documents=len(self.documents),
            )

    def load_index(self) -> bool:
        """
        Load FAISS index, BM25 index, and metadata from disk.

        Returns:
            True if indices loaded successfully, False otherwise
        """
        import shutil

        faiss_file = self.index_path / "faiss.index"
        bm25_file = self.index_path / "bm25.index"
        metadata_file = self.index_path / "metadata.pkl"

        if not faiss_file.exists() or not bm25_file.exists() or not metadata_file.exists():
            logger.warning(
                "load_index_failed",
                reason="files_not_found",
                faiss_exists=faiss_file.exists(),
                bm25_exists=bm25_file.exists(),
                metadata_exists=metadata_file.exists(),
            )
            return False

        with LoggerContext(logger, "load_index", path=str(self.index_path)):
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(faiss_file))
            logger.info(
                "faiss_index_loaded",
                file=str(faiss_file),
                total_vectors=self.faiss_index.ntotal,
            )

            # Load BM25 index
            with open(bm25_file, "rb") as f:
                self.bm25_index = pickle.load(f)
            logger.info("bm25_index_loaded", file=str(bm25_file))

            # Load metadata
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            self.documents = metadata["documents"]
            self.tokenized_contexts = metadata["tokenized_contexts"]
            self.dimension = metadata["dimension"]
            embedding_model_name = metadata["embedding_model"]

            logger.info(
                "metadata_loaded",
                file=str(metadata_file),
                num_documents=len(self.documents),
                dimension=self.dimension,
                embedding_model=embedding_model_name,
            )

            # Load embedding model if not already loaded
            if self.embedding_model is None:
                self._load_embedding_model()

            # Verify embedding model matches - if not, delete index and rebuild
            if embedding_model_name != self.embedding_model_name:
                logger.warning(
                    "embedding_model_mismatch_deleting_index",
                    index_model=embedding_model_name,
                    config_model=self.embedding_model_name,
                    action="deleting_old_index",
                )
                # Delete old index
                shutil.rmtree(self.index_path)
                self.index_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "old_index_deleted",
                    reason="model_mismatch",
                    old_model=embedding_model_name,
                    new_model=self.embedding_model_name,
                )
                return False

            return True

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents using FAISS only.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of documents with metadata and similarity scores
        """
        if self.faiss_index is None:
            logger.error("retrieve_failed", reason="no_index_loaded")
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        with LoggerContext(logger, "retrieve_faiss", query=query, k=k):
            # Load embedding model if needed
            if self.embedding_model is None:
                self._load_embedding_model()

            # Embed query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)

            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    result = self.documents[idx].copy()
                    result["faiss_score"] = float(score)
                    result["similarity_score"] = float(score)  # For backward compatibility
                    results.append(result)

            logger.info(
                "retrieval_completed",
                query=query,
                k=k,
                num_results=len(results),
                top_score=results[0]["similarity_score"] if results else None,
            )

            return results

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 5,
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents using hybrid FAISS + BM25 retrieval.

        Combines dense (FAISS) and sparse (BM25) retrieval using weighted score fusion.

        Args:
            query: Query string
            k: Number of documents to retrieve
            faiss_weight: Weight for FAISS scores (default 0.7)
            bm25_weight: Weight for BM25 scores (default 0.3)

        Returns:
            List of documents with metadata and hybrid scores
        """
        if self.faiss_index is None or self.bm25_index is None:
            logger.error("retrieve_hybrid_failed", reason="indices_not_loaded")
            raise RuntimeError("Indices not loaded. Call load_index() or build_index() first.")

        with LoggerContext(
            logger,
            "retrieve_hybrid",
            query=query,
            k=k,
            faiss_weight=faiss_weight,
            bm25_weight=bm25_weight,
        ):
            # Load embedding model if needed
            if self.embedding_model is None:
                self._load_embedding_model()

            # FAISS retrieval (get more candidates for reranking)
            k_candidates = min(k * 3, len(self.documents))  # Get 3x candidates

            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            faiss_scores, faiss_indices = self.faiss_index.search(query_embedding, k_candidates)
            faiss_scores = faiss_scores[0]
            faiss_indices = faiss_indices[0]

            # BM25 retrieval
            tokenized_query = self._tokenize_for_bm25(query)
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Normalize scores to [0, 1] range
            # FAISS scores (cosine similarity) are already in [-1, 1], shift to [0, 1]
            faiss_scores_norm = (faiss_scores + 1) / 2

            # BM25 scores: normalize by max score
            bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            bm25_scores_norm = bm25_scores / bm25_max

            # Create score dictionary for all documents
            hybrid_scores = {}

            # Add FAISS scores for candidates
            for idx, score_norm in zip(faiss_indices, faiss_scores_norm):
                if idx < len(self.documents):
                    hybrid_scores[int(idx)] = {
                        "faiss_score": float(score_norm),
                        "bm25_score": float(bm25_scores_norm[idx]),
                        "faiss_raw": float(faiss_scores[list(faiss_indices).index(idx)]),
                        "bm25_raw": float(bm25_scores[idx]),
                    }

            # Calculate hybrid scores
            for idx in hybrid_scores:
                faiss_s = hybrid_scores[idx]["faiss_score"]
                bm25_s = hybrid_scores[idx]["bm25_score"]
                hybrid_scores[idx]["hybrid_score"] = (
                    faiss_weight * faiss_s + bm25_weight * bm25_s
                )

            # Sort by hybrid score and get top-k
            sorted_indices = sorted(
                hybrid_scores.keys(),
                key=lambda idx: hybrid_scores[idx]["hybrid_score"],
                reverse=True,
            )[:k]

            # Prepare results
            results = []
            for idx in sorted_indices:
                result = self.documents[idx].copy()
                result["doc_id"] = idx
                result["hybrid_score"] = hybrid_scores[idx]["hybrid_score"]
                result["faiss_score_normalized"] = hybrid_scores[idx]["faiss_score"]
                result["bm25_score_normalized"] = hybrid_scores[idx]["bm25_score"]
                result["faiss_score_raw"] = hybrid_scores[idx]["faiss_raw"]
                result["bm25_score_raw"] = hybrid_scores[idx]["bm25_raw"]
                results.append(result)

            logger.info(
                "hybrid_retrieval_completed",
                query=query,
                k=k,
                num_results=len(results),
                top_hybrid_score=results[0]["hybrid_score"] if results else None,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
            )

            return results


if __name__ == "__main__":
    """
    Example usage: Build index and test hybrid retrieval.
    Run with: python -m src.retriever
    """
    # Initialize retriever
    retriever = FinQARetriever()

    # Try to load existing index
    if not retriever.load_index():
        print("\nNo existing index found. Building new hybrid index from train set...")
        retriever.build_index()
        print("\nHybrid index (FAISS + BM25) built and saved successfully!")
    else:
        print("\nHybrid index (FAISS + BM25) loaded successfully!")

    # Test hybrid retrieval
    print("\n" + "="*80)
    print("TESTING HYBRID RETRIEVAL (FAISS + BM25)")
    print("="*80 + "\n")

    test_query = "what is the interest expense in 2009?"
    print(f"Query: {test_query}\n")

    results = retriever.retrieve_hybrid(test_query, k=5, faiss_weight=0.7, bm25_weight=0.3)

    print(f"Retrieved {len(results)} documents:\n")

    for i, result in enumerate(results, 1):
        print(f"{'─'*80}")
        print(f"RESULT #{i}")
        print(f"{'─'*80}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"  FAISS (normalized): {result['faiss_score_normalized']:.4f} (raw: {result['faiss_score_raw']:.4f})")
        print(f"  BM25  (normalized): {result['bm25_score_normalized']:.4f} (raw: {result['bm25_score_raw']:.4f})")
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nProgram: {result['program']}")
        print(f"\nContext Preview (first 300 chars):")
        print(f"{result['context'][:300]}...")
        print()

    print("="*80)
    print("HYBRID RETRIEVAL TEST COMPLETE")
    print("="*80)
