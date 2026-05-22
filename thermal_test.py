"""
Sustained-load thermal test. Generate continuously and log tok/s every iteration.
On a fanless MacBook Air, throughput should drop over time as the SoC throttles.
"""
import time
import json
from datetime import datetime
import mlx.core as mx
from mlx_lm import load, generate

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DURATION_MIN = 20            # how long to run
MAX_TOKENS_PER_GEN = 200     # per request
PROMPT = "Explain the architecture and training process of large language models in detail."

def run():
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, tokenize=False,
    )

    end_time = time.time() + DURATION_MIN * 60
    log = []
    iteration = 0
    print(f"Running sustained load for {DURATION_MIN} minutes...")
    print(f"{'time_s':>8} {'iter':>4} {'tok/s':>8}")

    start = time.time()
    while time.time() < end_time:
        t0 = time.perf_counter()
        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=MAX_TOKENS_PER_GEN,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        n_tokens = len(tokenizer.encode(response))
        tps = n_tokens / elapsed
        elapsed_total = time.time() - start
        log.append({"t_seconds": round(elapsed_total, 1),
                    "iter": iteration, "tok_per_sec": round(tps, 2)})
        print(f"{elapsed_total:8.1f} {iteration:4d} {tps:8.1f}")
        iteration += 1

    fname = f"thermal_test_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump({"model": MODEL, "duration_min": DURATION_MIN,
                   "prompt": PROMPT, "log": log}, f, indent=2)
    print(f"\nSaved → {fname}")

    # Quick summary
    first_5 = [r["tok_per_sec"] for r in log[:5]]
    last_5 = [r["tok_per_sec"] for r in log[-5:]]
    print(f"\nFirst 5 iters avg: {sum(first_5)/len(first_5):.1f} tok/s")
    print(f"Last 5 iters avg:  {sum(last_5)/len(last_5):.1f} tok/s")
    print(f"Sustained throughput: {sum(last_5)/sum(first_5)*100:.0f}% of peak")

if __name__ == "__main__":
    run()