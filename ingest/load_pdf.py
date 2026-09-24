from pathlib import Path

import yaml
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES_FILE = DATA_DIR / "sources.yaml"
PDF_DIR = DATA_DIR / "raw" / "pdfs"


def load_sources():
    with open(SOURCES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pdfs():
    sources = load_sources()
    docs = []

    for src in sources.get("pdfs", []):
        path = PDF_DIR / src["file"]
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        pages = PyPDFLoader(str(path)).load()

        for page in pages:
            page.metadata.update({
                "topic": src["topic"],
                "agency": src["agency"],
                "source_type": "pdf",
                "source_url": src["url"],
                "last_checked": str(src["last_checked"]),
            })

        docs.extend(pages)
        print(f"Loaded {len(pages)} pages from {src['file']}")

    return docs


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"\nTotal pages: {len(docs)}")
    print("\nMetadata of first page:")
    print(docs[0].metadata)
    print("\nFirst 500 characters:")
    print(docs[0].page_content[:500])