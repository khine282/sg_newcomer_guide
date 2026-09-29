from langchain_community.document_loaders import NotionDirectoryLoader

from ingest.load_pdf import DATA_DIR

NOTION_DIR = DATA_DIR / "notion_export"


def load_notion():
    if not NOTION_DIR.exists():
        print(f"Missing folder: {NOTION_DIR}")
        return []

    docs = NotionDirectoryLoader(str(NOTION_DIR)).load()

    for doc in docs:
        doc.metadata.update({
            "topic": "tips",
            "agency": "personal",
            "source_type": "notion",
            "source_url": "",
            "last_checked": "2026-09",
        })
        print(f"Loaded {doc.metadata['source']} ({len(doc.page_content)} chars)")

    return docs


if __name__ == "__main__":
    docs = load_notion()
    print(f"\nTotal Notion pages: {len(docs)}")
    if docs:
        print("\nMetadata of first page:")
        print(docs[0].metadata)
        print("\nFirst 500 characters:")
        print(docs[0].page_content[:500])