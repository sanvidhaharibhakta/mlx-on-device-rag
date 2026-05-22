"""
Helpfulness eval: questions whose answers ARE in the corpus.
A correct grounded answer = pass. A refusal = over-refusal failure.
"""
from query import RAGPipeline

# Questions answerable from your papers — fill in based on what you ingested
IN_CONTEXT_QUESTIONS = [
    {
        "q": "What is multi-head attention?",
        "must_contain_any": ["multi-head", "attention", "heads", "parallel"],
        "source_hint": "1706.03762",
    },
    {
        "q": "What problem does retrieval-augmented generation solve?",
        "must_contain_any": ["retrieval", "knowledge", "hallucination", "external", "up-to-date", "factual"],
        "source_hint": "RAG paper",
    },
    {
        "q": "What is the architecture of the Transformer model?",
        "must_contain_any": ["encoder", "decoder", "attention", "self-attention"],
        "source_hint": "1706.03762",
    },
    {
        "q": "What is positional encoding?",
        "must_contain_any": ["position", "sinusoidal", "sequence", "order"],
        "source_hint": "1706.03762",
    },
    {
        "q": "What dataset is used in DeepFM?",
        "must_contain_any": ["criteo", "ctr", "click", "advertising", "company"],
        "source_hint": "1703.04247",
    },
]

REFUSAL_PHRASE = "cannot answer this from the provided documents"

def classify(answer: str, must_contain_any):
    a = answer.lower().strip()
    # Pure refusal = exactly the refusal phrase, possibly with whitespace
    if a == REFUSAL_PHRASE or a == REFUSAL_PHRASE + ".":
        return "REFUSED"
    # Otherwise check if it answers the question
    if any(kw in a for kw in must_contain_any):
        return "ANSWERED"
    return "OFF_TOPIC"

def run_eval(rag, label=""):
    print(f"\n{'='*60}\nHELPFULNESS EVAL: {label}\n{'='*60}")
    results = []
    for item in IN_CONTEXT_QUESTIONS:
        result = rag.answer(item["q"], verbose=False)
        verdict = classify(result["answer"], item["must_contain_any"])
        results.append({"q": item["q"], "verdict": verdict, "answer": result["answer"][:200]})
        symbol = {"ANSWERED": "✓", "REFUSED": "✗", "OFF_TOPIC": "?"}[verdict]
        print(f"  {symbol} {verdict}: {item['q']}")
        print(f"     → {result['answer'][:200]}")
    answered = sum(1 for r in results if r["verdict"] == "ANSWERED")
    refused = sum(1 for r in results if r["verdict"] == "REFUSED")
    off = sum(1 for r in results if r["verdict"] == "OFF_TOPIC")
    print(f"\nResults: {answered} answered / {refused} over-refused / {off} off-topic")
    print(f"Helpfulness pass rate: {answered/len(results)*100:.0f}%")
    return answered / len(results), results

if __name__ == "__main__":
    rag = RAGPipeline()
    pass_rate, results = run_eval(rag, label="Balanced prompt v3")