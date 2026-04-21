"""Test hybrid retrieval with company name boost"""

from src.retriever import FinQARetriever

def test_queries():
    # Initialize retriever
    retriever = FinQARetriever()

    # Load index
    if not retriever.load_index():
        print("Error: Index not found. Run 'python -m src.retriever' first to build index.")
        return

    print("\n" + "=" * 100)
    print("TESTING HYBRID RETRIEVAL WITH COMPANY NAME BOOST")
    print("=" * 100)

    # Test queries
    queries = [
        "what was the percentage cumulative total return for citi common stock for the five year period ended december 31 2017?",
        "what percentage of total oil and gas mmboe comes from canada?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 100}")
        print(f"QUERY {i}")
        print("=" * 100)
        print(f"Question: {query}")
        print()

        # Check if company name detected
        has_company = retriever._has_company_name(query)
        print(f"Company Name Detected: {has_company}")
        if has_company:
            print("  → Weights adjusted: FAISS=0.5, BM25=0.5")
        else:
            print("  → Default weights: FAISS=0.7, BM25=0.3")
        print()

        # Retrieve
        results = retriever.retrieve_hybrid(query, k=8)

        print(f"Retrieved {len(results)} documents")
        print(f"Top hybrid score: {results[0]['hybrid_score']:.4f}")
        print()

        # Show top 4 results
        print("TOP 4 RETRIEVED DOCUMENTS:")
        print("-" * 100)
        for j, doc in enumerate(results[:4], 1):
            print(f"\n{j}. Hybrid Score: {doc['hybrid_score']:.4f} "
                  f"(FAISS: {doc['faiss_score_normalized']:.3f}, BM25: {doc['bm25_score_normalized']:.3f})")
            print(f"   Question: {doc['question'][:120]}...")
            print(f"   Answer: {doc['answer']}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    test_queries()
