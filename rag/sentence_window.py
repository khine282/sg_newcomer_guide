"""Sentence-window retrieval (Course 3, Lesson 3).

Instead of 1000-character chunks, we embed single sentences, so a search
matches the one sentence that answers the question. Each sentence keeps its
surrounding sentences (the "window") in metadata, and after the search we give
the LLM the window instead of the lone sentence, so it still has context.
"""
import json
import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ingest.load_all import OUTPUT_FILE as DOCUMENTS_FILE
from rag.vectorstore import build_vectorstore, get_vectorstore

COLLECTION_NAME = "sg_newcomer_guide_sentences"

# Sentences on each side of the matched one (the course uses 3).
WINDOW_SIZE = 3

# PDFs have many tiny fragments (headings, bullet points, page numbers).
# Joining them with the next sentence gives better embeddings, and fewer of
# them: Gemini's free tier only allows about 1000 embedding requests per day.
MIN_SENTENCE_LENGTH = 40

# Split after . ! ? or the Burmese full stop, or at a blank line.
SENTENCE_END = re.compile(r"(?<=[.!?။])\s+|\n\s*\n")


def split_sentences(text):
    sentences = []
    pending = ""
    for part in SENTENCE_END.split(text):
        pending = f"{pending} {' '.join(part.split())}".strip()
        if len(pending) >= MIN_SENTENCE_LENGTH:
            sentences.append(pending)
            pending = ""
    if pending:
        sentences.append(pending)
    return sentences


def make_sentence_docs(docs):
    sentence_docs = []
    seen = set()
    for doc in docs:
        # Skip the deliberately duplicated CPF pages: same text, same embedding.
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)

        sentences = split_sentences(doc.page_content)
        for i, sentence in enumerate(sentences):
            window = sentences[max(0, i - WINDOW_SIZE):i + WINDOW_SIZE + 1]
            sentence_docs.append(Document(
                page_content=sentence,
                metadata={**doc.metadata, "window": " ".join(window),
                          "chunk_id": len(sentence_docs)},
            ))
    return sentence_docs


def load_documents(path=DOCUMENTS_FILE):
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


def replace_with_windows(docs, k):
    """Swap each matched sentence for its window, skipping overlapping windows."""
    results = []
    for doc in docs:
        m = doc.metadata
        # Neighbouring sentences have almost the same window - keep only one.
        if any(r.metadata["source"] == m["source"]
               and abs(r.metadata["chunk_id"] - m["chunk_id"]) <= WINDOW_SIZE
               for r in results):
            continue
        results.append(Document(page_content=m["window"],
                                metadata={**m, "sentence": doc.page_content}))
        if len(results) == k:
            break
    return results


class SentenceWindowRetriever(BaseRetriever):
    """Search sentences, then return the windows around the best k of them.

    A real BaseRetriever (not `search | RunnableLambda`), because
    create_retrieval_chain only passes the question text to BaseRetrievers;
    anything else gets the whole {"input": ...} dict.
    """
    k: int = 4
    fetch_k: int = 12

    def _get_relevant_documents(self, query, *, run_manager):
        docs = get_vectorstore(COLLECTION_NAME).similarity_search(query, k=self.fetch_k)
        return replace_with_windows(docs, self.k)


def get_sentence_window_retriever(k=4, fetch_k=12):
    return SentenceWindowRetriever(k=k, fetch_k=fetch_k)


if __name__ == "__main__":
    sentence_docs = make_sentence_docs(load_documents())
    lengths = [len(d.page_content) for d in sentence_docs]
    print(f"{len(sentence_docs)} sentences, avg length {sum(lengths) // len(lengths)}")
    print(f"Embedding them into '{COLLECTION_NAME}'...")
    # resume=True: embedding ~900 sentences on the CPU takes a while, so a build
    # that gets interrupted can continue instead of starting over.
    build_vectorstore(sentence_docs, COLLECTION_NAME, resume=True)

    retriever = get_sentence_window_retriever()
    for q in ["How much annual leave do employees get?",
              "CPF ဆိုတာ ဘာလဲ"]:  # What is CPF?
        print(f"\nQ: {q}")
        for d in retriever.invoke(q):
            print(f"  [#{d.metadata['chunk_id']} {d.metadata['topic']}] "
                  f"matched: {d.metadata['sentence'][:80]!r}")
