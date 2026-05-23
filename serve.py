"""
Service layer: reads JSON questions from stdin, writes JSON answers to stdout.
Designed to be launched as a subprocess by the Swift app.

Protocol:
  Input  (one per line):  {"question": "..."}
  Output (one per line):  {"type": "retrieved", "chunks": [...]}
                          {"type": "token", "text": "..."}
                          {"type": "done", "stats": {...}}
                          {"type": "error", "message": "..."}
"""
import json
import sys
import time
from query import RAGPipeline

def emit(obj):
    """Write a JSON line to stdout and flush immediately so Swift sees it now."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    # Log readiness to stderr so it doesn't pollute the data channel
    print("[serve.py] loading pipeline...", file=sys.stderr, flush=True)
    rag = RAGPipeline()
    print("[serve.py] ready", file=sys.stderr, flush=True)
    emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            question = req.get("question", "").strip()
            if not question:
                emit({"type": "error", "message": "empty question"})
                continue

            # Retrieve first, emit the chunks
            t0 = time.perf_counter()
            retrieved = rag.retrieve(question)
            top_score = retrieved[0]["score"]
            emit({
                "type": "retrieved",
                "chunks": [
                    {"source": r["source"], "chunk_id": r["chunk_id"],
                     "score": round(r["score"], 3),
                     "preview": r["chunk"][:200]}
                    for r in retrieved
                ],
                "top_score": round(top_score, 3),
                "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1),
            })

            # For v1 we use the existing answer() method — non-streaming for simplicity.
            # We'll add real token streaming later if time allows.
            result = rag.answer(question, verbose=False)

            # Stream the answer in word chunks for a "live" feel even without true streaming
            words = result["answer"].split(" ")
            for w in words:
                emit({"type": "token", "text": w + " "})
                time.sleep(0.01)  # tiny delay so UI shows progressive rendering

            emit({
                "type": "done",
                "stats": {
                    "retrieval_ms": result.get("retrieval_ms", 0),
                    "generation_s": result.get("generation_s", 0),
                    "early_refusal": result.get("early_refusal", False),
                },
            })
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"bad json: {e}"})
        except Exception as e:
            emit({"type": "error", "message": str(e)})

if __name__ == "__main__":
    main()