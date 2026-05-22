import psutil
import os
import json
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

PROMPTS = [
    "Explain the concept of transformers in machine learning.",
    "What is retrieval-augmented generation?",
    "Write a short poem about the ocean.",
    "Summarize the key ideas of object-oriented programming.",
    "What are the tradeoffs of model quantization?",
]

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def benchmark(model_name, prompts, max_tokens=150, runs_per_prompt=3):
    print(f"\n=== {model_name} ===")
    mem_before = get_memory_mb()
    model, tokenizer = load(model_name)
    mem_after_load = get_memory_mb()
    print(f"Model load memory delta: {mem_after_load - mem_before:.0f} MB\n")

    all_results = []
    for prompt in prompts:
        # Apply chat template since it's an Instruct model
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # Run each prompt multiple times, take median to reduce noise
        runs = []
        for _ in range(runs_per_prompt):
            response = generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                verbose=True,  # prints tok/s after each generation
            )
            runs.append(response)
        all_results.append({"prompt": prompt[:50], "sample_response": runs[-1][:200]})
        print()

    peak_mem = get_memory_mb()
    print(f"Peak process memory: {peak_mem:.0f} MB")
    return {"model": model_name, "peak_mem_mb": peak_mem, "results": all_results}

if __name__ == "__main__":
    result = benchmark(MODEL, PROMPTS)
    # Save raw results
    fname = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {fname}")