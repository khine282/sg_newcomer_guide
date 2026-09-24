import re
import time

from dotenv import load_dotenv

load_dotenv()  # must run before importing WebBaseLoader (reads USER_AGENT)

from langchain_community.document_loaders import WebBaseLoader

from ingest.load_pdf import load_sources

MIN_CHARS = 500  # pages shorter than this probably failed to load


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # collapse blank lines
    return text.strip()


def load_web():
    sources = load_sources()
    docs = []

    for src in sources.get("web", []):
        url = src["url"]
        try:
            page_docs = WebBaseLoader(url).load()
        except Exception as e:
            print(f"FAILED: {url} ({e})")
            continue

        for doc in page_docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({
                "topic": src["topic"],
                "agency": src["agency"],
                "source_type": "web",
                "source_url": url,
                "last_checked": str(src["last_checked"]),
            })

            length = len(doc.page_content)
            warning = "  <-- TOO SHORT, check this page" if length < MIN_CHARS else ""
            print(f"Loaded {url} ({length} chars){warning}")

        docs.extend(page_docs)
        time.sleep(1)  # be polite to the servers

    return docs


if __name__ == "__main__":
    docs = load_web()
    print(f"\nTotal web pages: {len(docs)}")
    print("\nMetadata of first page:")
    print(docs[0].metadata)
    print("\nFirst 800 characters:")
    print(docs[0].page_content[:800])