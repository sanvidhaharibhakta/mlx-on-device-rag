import os
import json
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("docs")
INDEX_PATH = Path("index.npz")
META_PATH = Path("index_meta.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~80MB, fast
CHUNK_SIZE = 500   # characters, not tokens — simpler for v1
CHUNK_OVERLAP = 100

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def build_index():
    print(f"Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    all_meta = []
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        print(f"  Processing {pdf_path.name}")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": pdf_path.name, "chunk_id": i})
        print(f"    → {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)

    np.savez(INDEX_PATH, embeddings=embeddings, chunks=np.array(all_chunks, dtype=object))
    with open(META_PATH, "w") as f:
        json.dump(all_meta, f)

    print(f"\n✓ Saved {len(all_chunks)} chunks → {INDEX_PATH}")
    print(f"  Index size: {INDEX_PATH.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    build_index()