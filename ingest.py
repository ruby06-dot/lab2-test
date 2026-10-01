"""
ingest.py - one-off script that builds the vector database.

Reads every document in lab2-starter-main, extracts metadata, splits the
text into chunks, embeds the chunks and stores them in a local ChromaDB
(the chroma_db folder).

Run it from the repo root with:  python ingest.py
"""

# Streamlit Cloud needs a newer sqlite3; this is a no-op if pysqlite3 is absent
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# ---------- Settings ----------
DATA_DIR = Path("lab2-starter-main")   # unzipped evidence folder
DB_PATH = "chroma_db"                  # where the database is saved
COLLECTION_NAME = "canvassian"
BATCH_SIZE = 100                       # chunks written per batch

# OpenAI embedding model (Home.py must use the same model)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large",
)

# Split on sentence boundaries: ~512 tokens per chunk, 50-token overlap
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def make_title(path: Path) -> str:
    """Turn a file name into a readable title (drop trailing number and extension)."""
    parts = path.stem.split("_")
    if parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def extract_metadata(path: Path, text: str) -> dict:
    """doc_type comes from the folder name; emails use their Subject line as title."""
    doc_type = path.parent.name  # board_papers / contracts / emails
    metadata = {
        "doc_type": doc_type,
        "source": f"{doc_type}/{path.name}",
        "title": make_title(path),
    }
    if doc_type == "emails":
        lines = text.strip().splitlines()
        if lines and lines[0].lower().startswith("subject:"):
            metadata["title"] = lines[0][len("subject:"):].strip()
    return metadata


def main():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete any previous collection so re-running does not create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    # Step 1: read files, extract metadata, chunk
    ids, documents, metadatas = [], [], []
    files = sorted(DATA_DIR.glob("*/*.txt"))
    print(f"Found {len(files)} files")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        metadata = extract_metadata(path, text)
        # Prefix every chunk with its type and title so it keeps its context
        header = f"[{metadata['doc_type']}] {metadata['title']}\n\n"
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{metadata['source']}#{i}")
            documents.append(header + chunk)
            metadatas.append({**metadata, "chunk": i})

    # Step 2: embed and write to the database in batches
    print(f"{len(documents)} chunks in total, writing to the database")
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  written {min(end, len(documents))} / {len(documents)}")

    # Step 3: quick sanity check - search contracts for change of control terms
    results = collection.query(
        query_texts=["change of control termination clause"],
        n_results=5,
        where={"doc_type": "contracts"},
    )
    print("\nTest query: change of control (contracts only)")
    for meta in results["metadatas"][0]:
        print("  ", meta["source"])


if __name__ == "__main__":
    main()