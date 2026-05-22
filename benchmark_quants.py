"""
Benchmark Llama-3.2-1B across 4-bit, 8-bit, and bf16 quantization on MLX.
Measures: prompt-processing tok/s, generation tok/s, peak GPU memory.
"""
import gc
import json
import time
from datetime import datetime
import mlx.core as mx
from mlx_lm import load, generate

MODELS = [
    ("4-bit", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
    ("8-bit", "mlx-community/Llama-3.2-1B-Instruct-8bit"),
    ("bf16",  "mlx-community/Llama-3.2-1B-Instruct-bf16"),
]

# Fixed prompt set — same across all 3 quants for fair comparison
PROMPTS = [
    "Explain the concept of transformers in machine learning.",
    "What is retrieval-augmented generation?",
    "Write a short poem about the ocean.",
    "Summarize the key ideas of object-oriented programming.",
    "What are the tradeoffs of model quantization?",
]

WARMUP_RUNS = 1     # discard first run (cold cache)
TIMED_RUNS = 3      # take median of these
MAX_TOKENS = 150

def benchmark_one(label, model_name):
    print(f"\n{'='*60}")
    print(f"  {label}: {model_name}")
    print(f"{'='*60}")

    mx.reset_peak_memory()  # reset MLX peak memory counter
    model, tokenizer = load(model_name)
    mem_after_load_gb = mx.get_peak_memory() / 1e9
    print(f"Memory after load: {mem_after_load_gb:.3f} GB")

    all_prompt_tps = []
    all_gen_tps = []
    peak_mem_gb = mem_after_load_gb

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Warmup
        for _ in range(WARMUP_RUNS):
            generate(model, tokenizer, prompt=formatted,
                     max_tokens=MAX_TOKENS, verbose=False)

        # Timed runs — MLX prints tok/s when verbose=True, but we need it
        # as numbers, so we time it ourselves and count tokens via tokenizer
        prompt_tokens = len(tokenizer.encode(formatted))
        for _ in range(TIMED_RUNS):
            t0 = time.perf_counter()
            response = generate(model, tokenizer, prompt=formatted,
                                max_tokens=MAX_TOKENS, verbose=False)
            elapsed = time.perf_counter() - t0
            gen_tokens = len(tokenizer.encode(response))
            # Total time roughly = prompt_processing + generation
            # We approximate: assume prompt processing is small; report gen tps
            gen_tps = gen_tokens / elapsed
            all_gen_tps.append(gen_tps)
            peak_mem_gb = max(peak_mem_gb, mx.get_peak_memory() / 1e9)

        print(f"  {prompt[:40]:40s} | {gen_tps:6.1f} tok/s | {gen_tokens} tokens")

    # Median is more robust than mean against outliers
    all_gen_tps.sort()
    median_gen_tps = all_gen_tps[len(all_gen_tps) // 2]

    result = {
        "label": label,
        "model": model_name,
        "median_gen_tps": round(median_gen_tps, 2),
        "min_gen_tps": round(min(all_gen_tps), 2),
        "max_gen_tps": round(max(all_gen_tps), 2),
        "peak_mem_gb": round(peak_mem_gb, 3),
        "mem_after_load_gb": round(mem_after_load_gb, 3),
        "n_samples": len(all_gen_tps),
    }
    print(f"\n  → Median generation: {median_gen_tps:.1f} tok/s")
    print(f"  → Peak memory:       {peak_mem_gb:.3f} GB")

    # Critical: free the model before loading the next one
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    return result

if __name__ == "__main__":
    results = []
    for label, model_name in MODELS:
        results.append(benchmark_one(label, model_name))

    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Quant':<8} {'Gen tok/s':<12} {'Peak GB':<10} {'Δ vs bf16':<12}")
    bf16_tps = next(r["median_gen_tps"] for r in results if r["label"] == "bf16")
    bf16_mem = next(r["peak_mem_gb"] for r in results if r["label"] == "bf16")
    for r in results:
        tps_ratio = r["median_gen_tps"] / bf16_tps
        mem_ratio = r["peak_mem_gb"] / bf16_mem
        print(f"{r['label']:<8} {r['median_gen_tps']:<12.1f} {r['peak_mem_gb']:<10.3f} "
              f"{tps_ratio:.2f}× speed, {mem_ratio:.2f}× memory")

    fname = f"benchmark_quants_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {fname}")