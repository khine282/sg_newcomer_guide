import json
from collections import Counter
from copy import deepcopy

from langchain_core.documents import Document

from ingest.load_notion import load_notion
from ingest.load_pdf import DATA_DIR, load_pdfs
from ingest.load_video import load_videos
from ingest.load_web import load_web

OUTPUT_FILE = DATA_DIR / "processed" / "documents.jsonl"
REQUIRED_FIELDS = ["topic", "agency", "source_type", "source_url", "last_checked"]

# Deliberate "messy data" (like the duplicated Lecture01 PDF in Course 2).
# Used later to show similarity search returning duplicates, and MMR fixing it.
DUPLICATE_FILE = "cpf_contributions_guide.pdf"


def add_duplicate(docs):
    dupes = [deepcopy(d) for d in docs
             if d.metadata.get("source", "").endswith(DUPLICATE_FILE)]
    print(f"Added {len(dupes)} duplicate pages from {DUPLICATE_FILE}")
    return docs + dupes


def check_metadata(docs):
    problems = 0
    for d in docs:
        missing = [f for f in REQUIRED_FIELDS if f not in d.metadata]
        if missing:
            problems += 1
            print(f"Missing {missing} in {d.metadata.get('source')}")
    print(f"Metadata check: {problems} documents with problems")


def save_documents(docs, path=OUTPUT_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(
                {"page_content": d.page_content, "metadata": d.metadata},
                ensure_ascii=False,
            ) + "\n")
    print(f"Saved {len(docs)} documents to {path}")


def load_documents(path=OUTPUT_FILE):
    """Used by later phases to read the saved documents."""
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]

MIN_CHARS = 100
def remove_empty(docs):
    kept = [d for d in docs if len(d.page_content.strip()) >= MIN_CHARS]
    print(f"Removed {len(docs) - len(kept)} near-empty documents (< {MIN_CHARS} chars)")
    return kept

def load_all():
    docs = []
    docs += load_pdfs()
    docs += load_web()
    docs += load_videos()
    docs += load_notion()
    docs = remove_empty(docs)
    docs = add_duplicate(docs)
    return docs


if __name__ == "__main__":
    docs = load_all()
    print()
    check_metadata(docs)
    print("\nBy source type:", dict(Counter(d.metadata["source_type"] for d in docs)))
    print("By agency:", dict(Counter(d.metadata["agency"] for d in docs)))
    print(f"Total documents: {len(docs)}\n")
    save_documents(docs)