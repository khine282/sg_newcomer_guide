import re
import time

from dotenv import load_dotenv

load_dotenv()  # must run before importing WebBaseLoader (reads USER_AGENT)

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from ingest.load_pdf import load_sources

MIN_CHARS = 500  # pages shorter than this probably failed to load
JUNK_TAGS = ["nav", "header", "footer", "script", "style",
             "aside", "form", "noscript", "button"]


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # collapse blank lines
    return text.strip()


def extract_main_text(soup):
    for tag in soup(JUNK_TAGS):
        tag.decompose()                          # delete menus, footers, etc.
    main = soup.find("main") or soup.body or soup
    return main.get_text(separator="\n")


def load_web():
    sources = load_sources()
    docs = []

    for src in sources.get("web", []):
        url = src["url"]
        try:
            soup = WebBaseLoader(url).scrape()   # raw HTML as BeautifulSoup
        except Exception as e:
            print(f"FAILED: {url} ({e})")
            continue

        title = soup.title.get_text(strip=True) if soup.title else ""
        text = clean_text(extract_main_text(soup))

        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "title": title,
                "topic": src["topic"],
                "agency": src["agency"],
                "source_type": "web",
                "source_url": url,
                "last_checked": str(src["last_checked"]),
            },
        )

        length = len(text)
        warning = "  <-- TOO SHORT, check this page" if length < MIN_CHARS else ""
        print(f"Loaded {url} ({length} chars){warning}")

        docs.append(doc)
        time.sleep(1)  # be polite to the servers

    return docs


if __name__ == "__main__":
    docs = load_web()
    print(f"\nTotal web pages: {len(docs)}")
    for doc in docs:
        print("\n" + "=" * 80)
        print(doc.metadata["title"])
        print("-" * 80)
        print(doc.page_content[:400])