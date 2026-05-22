=== mlx-community/Llama-3.2-1B-Instruct-4bit ===
Fetching 6 files: 100%|████████████████████████| 6/6 [00:00<00:00, 74017.13it/s]
Download complete: : 0.00B [00:00, ?B/s]                  | 0/6 [00:00<?, ?it/s]
Memory after load: 251 MB
  116.4 tok/s | 151 tokens | 1.30s
  118.3 tok/s | 151 tokens | 1.28s
  114.9 tok/s | 85 tokens | 0.74s
  122.4 tok/s | 151 tokens | 1.23s
  122.6 tok/s | 151 tokens | 1.23s

Avg tokens/sec: 118.9
Peak memory:    363 MB

## 2026-05-22 — Baseline: Llama-3.2-1B-Instruct-4bit on MLX
Hardware: MacBook Air, Apple Silicon (run `sysctl -n machdep.cpu.brand_string` to fill in)
Model size on disk: 713 MB

Cold start (first generation):
- Prompt processing: 86 tok/s
- Generation: 137 tok/s
- Peak memory: 0.80 GB

Warm (subsequent generations):
- Prompt processing: 844 tok/s
- Generation: 131 tok/s
- Peak memory: 0.80 GBa 

## Ingestion benchmark (2026-05-22)
Embedder: all-MiniLM-L6-v2 (384-dim, ~80MB)
Corpus: 7 ML papers
Chunks: 1,294 (avg ~185/paper)
Embedding time: 7.2 seconds (~180 chunks/sec)
Index size: 2.5 MB

## 2026-05-22 — Honesty eval (8 out-of-context questions)
Baseline prompt:  25% refusal rate (2/8) — model leaked parametric knowledge
Strict prompt v2: 100% refusal rate (8/8)
Δ: +75 percentage points

Changes:
- Reframed system identity ("strict document Q&A assistant, no other knowledge")
- Specified exact refusal string instead of abstract instruction
- Repeated constraint after the question (recency bias in small models)

## 2026-05-22 — Final v3 (balanced prompt + retrieval thresholding)
Approach: Two-axis eval — honesty (out-of-context refusal) + helpfulness (in-context answering)

Honesty eval (8 questions, all out-of-corpus):
  v1 (lenient prompt):              25%   — heavy parametric leakage
  v2 (strict prompt):               100%  — but over-refused everything
  v3 (balanced + threshold ≥0.45):  100%  — early refusal when top score < threshold

Helpfulness eval (5 in-corpus questions):
  v2 (strict prompt):               ~0%   — refused even well-grounded questions
  v3 (balanced + threshold):        100%  — all 5 answered with grounded content

Key insight: Prompt engineering alone plateaued ~60% honesty on small models.
Retrieval-score thresholding (defense in depth) reached 100% on both axes
by short-circuiting weak retrievals before they reach the LLM.

## 2026-05-22 — Quantization sweep (Llama-3.2-1B-Instruct on MLX)
Hardware: MacBook Air, Apple Silicon (no fan)
Method: 5 prompts × 3 timed runs per model, max_tokens=150, median reported

Quant | Gen tok/s | Peak Mem (GB) | Speed Δ | Memory Δ
4-bit |   120.3   |     0.80      |  3.13×  |  0.32×
8-bit |    70.3   |     1.42      |  1.83×  |  0.56×
bf16  |    38.5   |     2.54      |  1.00×  |  1.00×

Key finding: ~linear scaling between weight size and throughput,
consistent with memory-bandwidth-bound inference on Apple Silicon.
Quality impact: TBD (see eval below).

## 2026-05-22 — Thermal characterization (4-bit, 19 min sustained)
Hardware: MacBook Air, Apple Silicon, no fan, no charger
Test: continuous generation, ~200 tokens per call, no cooldown

Peak (min 0):         123 tok/s (median, low-variance)
Steady median:        ~115 tok/s (93% of peak), sustained
Tail variance:        worst-case dips to 75 tok/s in later minutes
Throttling pattern:   median holds, variance widens
                      (dynamic V/F scaling preserves mean throughput)

Deployment implication:
  Single-shot benchmarks overstate user-perceived throughput
  for sustained workloads. Real on-device chat experience
  should budget against the p95 tail (~95 tok/s), not the peak.