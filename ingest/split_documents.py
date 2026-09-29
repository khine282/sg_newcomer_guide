import json
from collections import Counter

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ingest.load_all import load_documents
from ingest.load_pdf import DATA_DIR

OUTPUT_FILE = DATA_DIR / "processed" / "chunks.jsonl"

# Same starting values as Course 2 (Lesson: Document Splitting).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Notion pages are Markdown, so we split them by headers first
# (like the Notion example in Course 2) and keep the header as metadata.
HEADERS_TO_SPLIT_ON = [("#", "header_1"), ("##", "header_2")]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
    is_separator_regex=True,
)
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)


def split_markdown(doc):
    """Split one Markdown document by headers, keeping the original metadata."""
    sections = markdown_splitter.split_text(doc.page_content)
    for section in sections:
        section.metadata = {**doc.metadata, **section.metadata}
    return text_splitter.split_documents(sections)


def split_documents(docs):
    chunks = []
    for doc in docs:
        if doc.metadata["source_type"] == "notion":
            chunks += split_markdown(doc)
        else:
            chunks += text_splitter.split_documents([doc])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def save_chunks(chunks, path=OUTPUT_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False,
            ) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    lengths = [len(c.page_content) for c in chunks]
    print(f"Documents: {len(docs)} -> Chunks: {len(chunks)}")
    print(f"Chunk length: min {min(lengths)}, max {max(lengths)}, "
          f"avg {sum(lengths) // len(lengths)}")
    print("Chunks by source type:",
          dict(Counter(c.metadata["source_type"] for c in chunks)))

    notion = [c for c in chunks if c.metadata["source_type"] == "notion"]
    if notion:
        print("\nExample Notion chunk metadata:")
        print(notion[1].metadata if len(notion) > 1 else notion[0].metadata)

    print()
    save_chunks(chunks)
