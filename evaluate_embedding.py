import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from knowledge_base_construction import run_pipeline, query_rag

# ── Models to evaluate ────────────────────────────────────────────────────────
EMBED_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
]

# ── Eval queries — OOP slide content ─────────────────────────────────────────
EVAL_QUERIES = [
    {"query": "What is inheritance in object oriented programming?",
     "keyword": "inherit"},
    {"query": "Explain polymorphism with example",
     "keyword": "polymorphism"},
    {"query": "What is encapsulation?",
     "keyword": "encapsulat"},
    {"query": "What is abstraction in OOP?",
     "keyword": "abstract"},
    {"query": "Difference between class and object",
     "keyword": "object"},
    {"query": "What is method overriding?",
     "keyword": "overrid"},
    {"query": "Explain constructor",
     "keyword": "constructor"},
    {"query": "What is an interface?",
     "keyword": "interface"},
]
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}

def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def recall_at_k(results, keyword, k=5):
    for r in results[:k]:
        if keyword.lower() in r["text"].lower():
            return 1
    return 0


def reciprocal_rank(results, keyword):
    for rank, r in enumerate(results, start=1):
        if keyword.lower() in r["text"].lower():
            return 1 / rank
    return 0


def evaluate_model(model_name: str) -> dict:
    print(f"\n===== Evaluating: {model_name} =====")

    collection_name = f"eval_{model_name.replace('/', '_').replace('-', '_')}"

    embed_start = time.time()
    collection = run_pipeline(embed_model=model_name, collection_name=collection_name)
    embed_time = round(time.time() - embed_start, 2)

    get_model(model_name)  # warm up cache before query loop

    recalls, mrrs, query_times = [], [], []

    for item in EVAL_QUERIES:
        t0 = time.time()
        results = query_rag(item["query"], collection,
                            embed_model=model_name, top_k=5)
        query_times.append(time.time() - t0)

        r = recall_at_k(results, item["keyword"])
        m = reciprocal_rank(results, item["keyword"])
        recalls.append(r)
        mrrs.append(m)

        top = results[0]["text"][:120].replace("\n", " ") if results else "—"
        print(f"  [{'HIT ' if r else 'MISS'}] keyword='{item['keyword']}'  "
              f"top chunk: {top}...")

    n = len(EVAL_QUERIES)
    return {
        "model":              model_name,
        "Recall@5":           round(sum(recalls) / n, 4),
        "MRR":                round(sum(mrrs) / n, 4),
        "Embedding_Time_sec": embed_time,
        "Avg_Query_Time_sec": round(sum(query_times) / n, 4),
    }


def main():
    all_results = []

    for model in EMBED_MODELS:
        try:
            res = evaluate_model(model)
            all_results.append(res)
        except Exception as e:
            print(f"  [FAILED] {model}: {e}")

    if not all_results:
        print("No models evaluated successfully.")
        return

    df = pd.DataFrame(all_results).sort_values("MRR", ascending=False)

    print("\n===== FINAL RESULTS =====")
    print(df.to_string(index=False))

    df.to_csv("embedding_comparison.csv", index=False)
    print("\nSaved: embedding_comparison.csv")
    print(f"Recommended model: {df.iloc[0]['model']}")


if __name__ == "__main__":
    main()