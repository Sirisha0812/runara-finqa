"""
Demo: FinQA Agent Architecture (No vLLM Required)

This demonstrates how the agent processes questions without needing a running vLLM server.
"""

from src.retriever import FinQARetriever
from src.logger import get_logger
import json

logger = get_logger(__name__)

def demo_retrieval():
    """Show how hybrid retrieval works"""
    print("="*80)
    print("DEMO: FinQA Agent Architecture")
    print("="*80)
    print()

    # Initialize retriever
    print("1. INITIALIZING HYBRID RETRIEVER (FAISS + BM25)")
    print("-" * 80)
    retriever = FinQARetriever()
    retriever.load_index()
    print(f"✅ Loaded {len(retriever.documents)} financial documents")
    print()

    # Test question
    question = "what is the interest expense in 2009?"
    print(f"2. TEST QUESTION: \"{question}\"")
    print("-" * 80)
    print()

    # Retrieval
    print("3. HYBRID RETRIEVAL (FAISS 70% + BM25 30%)")
    print("-" * 80)
    results = retriever.retrieve_hybrid(query=question, k=4)

    for i, doc in enumerate(results, 1):
        print(f"\n📄 Document {i}:")
        print(f"   Hybrid Score: {doc['hybrid_score']:.4f}")
        print(f"   Question: {doc['question'][:100]}...")
        print(f"   Answer: {doc['answer']}")
        print(f"   Context Preview: {doc['context'][:200]}...")

    print()
    print("="*80)
    print("4. AGENT WORKFLOW (Without vLLM)")
    print("="*80)
    print()
    print("┌─────────────┐")
    print("│  RETRIEVE   │ ← ✅ WORKS (shown above)")
    print("└──────┬──────┘")
    print("       │")
    print("┌──────▼──────┐")
    print("│   REASON    │ ← ⚠️  NEEDS vLLM (LLM generates reasoning)")
    print("└──────┬──────┘")
    print("       │")
    print("┌──────▼──────┐")
    print("│ CALCULATOR  │ ← ✅ WORKS (sympy evaluation)")
    print("└──────┬──────┘")
    print("       │")
    print("┌──────▼──────┐")
    print("│  VERIFIER   │ ← ⚠️  NEEDS vLLM (LLM fact-checks)")
    print("└──────┬──────┘")
    print("       │")
    print("┌──────▼──────┐")
    print("│   ANSWER    │ ← ⚠️  NEEDS vLLM (LLM formats answer)")
    print("└─────────────┘")
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("✅ Retrieval works locally (no GPU needed)")
    print("⚠️  LLM nodes require vLLM server")
    print()
    print("To run full agent:")
    print("1. Start vLLM server on Google Colab (see COLAB_SETUP.md)")
    print("2. Update .env with Colab URL")
    print("3. Run: python test_real_vllm.py")
    print()
    print("See RESULTS.md for full GPU test results from previous run!")
    print("="*80)

if __name__ == "__main__":
    demo_retrieval()
