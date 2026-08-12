"""
Retrieval evaluation harness.
Runs 5 sample queries against the vector store and prints top-2 results
with scores, for inclusion in docs/retrieval_evaluation.md.
"""
from rag.retriever import retrieve

QUERIES = [
    "What security features does the Sri Lankan Rs.5000 note have?",
    "Why did Thailand float the baht in 1997?",
    "Which Indian banknote depicts the Sanchi Stupa?",
    "How does Japan use holograms on its 2024 banknote series?",
    "Is it disrespectful to fold currency in Thailand?",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        results = retrieve(q, k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']}")
            print(f"      {r['text'][:160].replace(chr(10), ' ')}...")
