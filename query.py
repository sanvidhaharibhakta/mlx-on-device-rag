import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

INDEX_PATH = Path("index.npz")
META_PATH = Path("index_meta.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
TOP_K = 4  # number of chunks to retrieve
RETRIEVAL_THRESHOLD = 0.45  # below this, we refuse without calling the LLM

PROMPT_TEMPLATE = """You are a document Q&A assistant. Use ONLY the provided context to answer.

Rules:
- If the context contains the answer, give a clear, complete answer and stop.
- If the context does NOT contain the answer, respond with this exact sentence and nothing else: I cannot answer this from the provided documents.
- Do not combine an answer with a refusal. Pick one.
- Do not use outside knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

class RAGPipeline:
    def __init__(self):
        print("Loading embedder...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("Loading index...")
        data = np.load(INDEX_PATH, allow_pickle=True)
        self.embeddings = data["embeddings"]  # (N, 384), already L2-normalized
        self.chunks = data["chunks"]
        with open(META_PATH) as f:
            self.meta = json.load(f)
        print(f"  → {len(self.chunks)} chunks loaded")

        print(f"Loading LLM: {LLM_MODEL}")
        self.model, self.tokenizer = load(LLM_MODEL)
        print("Ready.\n")

    def retrieve(self, question, k=TOP_K):
        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        # Cosine similarity = dot product since both are L2-normalized
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            {
                "chunk": self.chunks[i],
                "score": float(scores[i]),
                "source": self.meta[i]["source"],
                "chunk_id": self.meta[i]["chunk_id"],
            }
            for i in top_idx
        ]

    

    def answer(self, question, verbose=True):
        t0 = time.perf_counter()
        retrieved = self.retrieve(question)
        t_retrieve = time.perf_counter() - t0

        top_score = retrieved[0]["score"]

    # Early refusal: if the best retrieval is weak, don't even call the LLM
        if top_score < RETRIEVAL_THRESHOLD:
            if verbose:
                print(f"⏱  Retrieval: {t_retrieve*1000:.1f}ms")
                print(f"   Top score {top_score:.3f} < threshold {RETRIEVAL_THRESHOLD} — early refusal")
                print(f"\n💬 Answer:\n==========")
                print("I cannot answer this from the provided documents.")
                print("==========")
                # Defense-in-depth: strip trailing/leading refusal artifacts if the model
        # produced a real answer AND tacked on the refusal phrase
                REFUSAL = "I cannot answer this from the provided documents."
                cleaned = response.strip()
        # If the response contains a real answer plus the refusal phrase,
        # keep whichever is dominant
                if REFUSAL in cleaned and len(cleaned) > len(REFUSAL) + 20:
            # There's substantive content beyond the refusal phrase — remove the refusal
                    cleaned = cleaned.replace(REFUSAL, "").strip()
                    response = cleaned
            return {
                "answer": "I cannot answer this from the provided documents.",
                "retrieved": retrieved,
                "retrieval_ms": t_retrieve * 1000,
                "generation_s": 0.0,
                "early_refusal": True,
            }

        context = "\n\n---\n\n".join(
            f"[Source: {r['source']}, chunk {r['chunk_id']}, score {r['score']:.3f}]\n{r['chunk']}"
            for r in retrieved
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if verbose:
            print(f"⏱  Retrieval: {t_retrieve*1000:.1f}ms (top score {top_score:.3f})")
            print(f"   Retrieved chunks:")
            for r in retrieved:
                print(f"     - {r['source']} (chunk {r['chunk_id']}, score {r['score']:.3f})")
            print(f"\n💬 Answer:")

        t0 = time.perf_counter()
        response = generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=300,
            verbose=verbose,
        )
        t_generate = time.perf_counter() - t0

        return {
            "answer": response,
            "retrieved": retrieved,
            "retrieval_ms": t_retrieve * 1000,
            "generation_s": t_generate,
            "early_refusal": False,
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    print("Ask questions about your documents. Ctrl-C to exit.\n")
    while True:
        try:
            q = input("Q: ").strip()
            if not q:
                continue
            rag.answer(q)
            print("\n" + "─" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break