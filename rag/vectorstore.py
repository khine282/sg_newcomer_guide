import json
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingest.load_pdf import DATA_DIR
from ingest.split_documents import OUTPUT_FILE as CHUNKS_FILE

load_dotenv()

PERSIST_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "sg_newcomer_guide"

# Gemini free tier allows 100 embedding requests per minute (each chunk
# counts as one), so we send small batches spread out: 20 every 15s = 80/min.
BATCH_SIZE = 20
WAIT_SECONDS = 15

# Multilingual model, so English and Burmese questions land close together.
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def load_chunks(path=CHUNKS_FILE):
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


def get_vectorstore():
    """Used by later phases to open the saved vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIR),
    )


def build_vectorstore(chunks):
    vectordb = get_vectorstore()
    # Start fresh each run, otherwise re-running adds every chunk again.
    vectordb.reset_collection()

    for start in range(0, len(chunks), BATCH_SIZE):
        if start > 0:
            time.sleep(WAIT_SECONDS)
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
