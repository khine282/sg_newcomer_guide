import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ingest.load_pdf import DATA_DIR

# Same path as ingest.split_documents.OUTPUT_FILE. Not imported from there,
# so the web app doesn't pull in the video/web loaders and their packages.
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
PERSIST_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "sg_newcomer_guide"

# Batches only so we can print progress.
BATCH_SIZE = 50

# Local multilingual model, so English and Burmese questions land close
# together. We used Gemini's embeddings before, but the free tier only allows
# 1000 per day - not enough to try several index types.
# Same bge-m3 model we ran in Ollama, but loaded with sentence-transformers so
# it also runs on the deployed app, where there is no Ollama.
# On the CPU so the GPU stays free for the eval judge; embedding a question
# still takes well under a second once the model is loaded.
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def load_chunks(path=CHUNKS_FILE):
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


def get_vectorstore(collection_name=COLLECTION_NAME):
    """Used by later phases to open the saved vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIR),
    )


def build_vectorstore(chunks, collection_name=COLLECTION_NAME, resume=False):
    vectordb = get_vectorstore(collection_name)
    if resume:
        # Only embed chunks that aren't stored yet, so an interrupted build
        # can continue where it stopped.
        done = set(vectordb.get(include=[])["ids"])
        chunks = [c for c in chunks if str(c.metadata["chunk_id"]) not in done]
        print(f"  resuming: {len(done)} already stored, {len(chunks)} to add")
    else:
        # Start fresh each run, otherwise re-running adds every chunk again.
        vectordb.reset_collection()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectordb.add_documents(batch, ids=[str(c.metadata["chunk_id"]) for c in batch])
        print(f"  added chunks {start}-{start + len(batch) - 1}")
    return vectordb


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks...")
    vectordb = build_vectorstore(chunks)
    print(f"Vector store has {vectordb._collection.count()} chunks in {PERSIST_DIR}\n")

    questions = [
        "What are the CPF contribution rates?",
        "စင်ကာပူမှာ ဖုန်းဆင်းကတ် ဘယ်မှာဝယ်ရမလဲ",  # Where do I buy a SIM card in Singapore?
    ]
    for q in questions:
        print(f"Q: {q}")
        for doc in vectordb.similarity_search(q, k=3):
            m = doc.metadata
            print(f"  - [{m['source_type']}/{m['topic']}] {m['source'].split(chr(92))[-1]}"
                  f" | {doc.page_content[:80]!r}")
        print()
