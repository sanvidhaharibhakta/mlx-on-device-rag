"""
Run honesty + helpfulness evals across all 3 quantization levels.
Same eval questions, same RAG pipeline — only the LLM changes.
"""
import gc
import json
from datetime import datetime
import mlx.core as mx

# We need to dynamically swap the LLM in the RAG pipeline,
# so we import the building blocks rather than the full class
from query import RAGPipeline, LLM_MODEL
from eval_honesty import OUT_OF_CONTEXT_QUESTIONS, classify as classify_honesty
from eval_helpfulness import IN_CONTEXT_QUESTIONS, classify as classify_helpfulness

MODELS = [
    ("4-bit", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
    ("8-bit", "mlx-community/Llama-3.2-1B-Instruct-8bit"),
    ("bf16",  "mlx-community/Llama-3.2-1B-Instruct-bf16"),
]

def run_evals_for_model(label, model_name):
    print(f"\n{'='*60}")
    print(f"  EVAL: {label} ({model_name})")
    print(f"{'='*60}")

    # Build the RAG pipeline, then swap in the model we want to test
    import query as q
    q.LLM_MODEL = model_name  # monkey-patch the module-level constant
    rag = RAGPipeline()

    # --- Honesty ---
    print(f"\n  Honesty eval ({len(OUT_OF_CONTEXT_QUESTIONS)} questions):")
    honesty_results = []
    for question in OUT_OF_CONTEXT_QUESTIONS:
        result = rag.answer(question, verbose=False)
        verdict = classify_honesty(result["answer"])
        honesty_results.append({
            "q": question, "verdict": verdict, "answer": result["answer"][:200],
        })
        symbol = {"REFUSED": "✓", "LEAKED": "✗", "AMBIGUOUS": "?"}[verdict]
        print(f"    {symbol} {verdict}: {question}")
    honesty_pass = sum(1 for r in honesty_results if r["verdict"] == "REFUSED")
    honesty_rate = honesty_pass / len(honesty_results)

    # --- Helpfulness ---
    print(f"\n  Helpfulness eval ({len(IN_CONTEXT_QUESTIONS)} questions):")
    help_results = []
    for item in IN_CONTEXT_QUESTIONS:
        result = rag.answer(item["q"], verbose=False)
        verdict = classify_helpfulness(result["answer"], item["must_contain_any"])
        help_results.append({
            "q": item["q"], "verdict": verdict, "answer": result["answer"][:200],
        })
        symbol = {"ANSWERED": "✓", "REFUSED": "✗", "OFF_TOPIC": "?"}[verdict]
        print(f"    {symbol} {verdict}: {item['q']}")
    help_pass = sum(1 for r in help_results if r["verdict"] == "ANSWERED")
    help_rate = help_pass / len(help_results)

    print(f"\n  → Honesty:     {honesty_pass}/{len(honesty_results)} ({honesty_rate*100:.0f}%)")
    print(f"  → Helpfulness: {help_pass}/{len(help_results)} ({help_rate*100:.0f}%)")

    # Cleanup before next model
    del rag
    gc.collect()
    mx.clear_cache()

    return {
        "label": label,
        "model": model_name,
        "honesty_rate": round(honesty_rate, 3),
        "honesty_pass": honesty_pass,
        "honesty_n": len(honesty_results),
        "help_rate": round(help_rate, 3),
        "help_pass": help_pass,
        "help_n": len(help_results),
        "honesty_details": honesty_results,
        "help_details": help_results,
    }

if __name__ == "__main__":
    all_results = [run_evals_for_model(label, m) for label, m in MODELS]

    print(f"\n\n{'='*60}\nQUALITY SUMMARY\n{'='*60}")
    print(f"{'Quant':<8} {'Honesty':<12} {'Helpfulness':<14}")
    for r in all_results:
        print(f"{r['label']:<8} "
              f"{r['honesty_pass']}/{r['honesty_n']} ({r['honesty_rate']*100:.0f}%){'':<2} "
              f"{r['help_pass']}/{r['help_n']} ({r['help_rate']*100:.0f}%)")

    fname = f"eval_quants_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {fname}")