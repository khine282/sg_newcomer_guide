from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator

from rag.llm import llm
from rag.vectorstore import get_vectorstore

vectordb = get_vectorstore()

# Descriptions (with the allowed values) help the LLM build correct filters,
# including for questions asked in Burmese.
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="topic",
        description="The topic of the chunk. One of "
                    "['work', 'cpf', 'transport', 'tips', 'housing', 'healthcare']",
        type="string",
    ),
    AttributeInfo(
        name="agency",
        description="The organisation that published the source. One of "
                    "['MOM', 'CPF Board', 'LTA', 'personal', 'CEA', 'SingHealth Polyclinics']",
        type="string",
    ),
    AttributeInfo(
        name="source_type",
        description="Where the chunk came from. One of ['pdf', 'web', 'video', 'notion']",
        type="string",
    ),
]
DOCUMENT_CONTENT_DESCRIPTION = "Practical information about living and working in Singapore"


def get_mmr_retriever(k=4, fetch_k=20):
    """MMR: fetch the top fetch_k matches, then pick k that are relevant AND different."""
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})


def get_self_query_retriever(k=4):
    """The LLM turns the question into a search query plus a metadata filter."""
    return SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        DOCUMENT_CONTENT_DESCRIPTION,
        METADATA_FIELD_INFO,
        # Passed explicitly: auto-detection breaks with the current
        # langchain-classic / langchain-community versions.
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": k},
    )


def get_compression_retriever(base_retriever=None):
    """The LLM keeps only the parts of each chunk that answer the question."""
    return ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=base_retriever or get_mmr_retriever(),
    )


def show(title, docs):
    print(f"\n--- {title} ({len(docs)} results)")
    for d in docs:
        m = d.metadata
        text = " ".join(d.page_content.split())
        print(f"  [{m['topic']}/{m['source_type']} #{m.get('chunk_id')}] {text[:100]}")


if __name__ == "__main__":
    q = "What are the CPF contribution rates?"
    print(f"Q: {q}")
    show("Similarity search", vectordb.similarity_search(q, k=4))
    show("MMR", get_mmr_retriever().invoke(q))

    q = "How do I travel around Singapore cheaply?"
    print(f"\nQ: {q}")
    show("Similarity search", vectordb.similarity_search(q, k=4))
    show("Metadata filter: topic=transport",
         vectordb.similarity_search(q, k=4, filter={"topic": "transport"}))

    self_query = get_self_query_retriever()
    for q in [
        "What do the official MOM documents say about working hours?",
        "ဆေးခန်းမှာ ဆရာဝန်ပြဖို့ ဘာလုပ်ရမလဲ",  # How do I see a doctor at a clinic?
    ]:
        print(f"\nQ: {q}")
        print("  LLM built:", self_query.query_constructor.invoke({"query": q}))
        show("Self-query", self_query.invoke(q))

    q = "How much annual leave do employees get?"
    print(f"\nQ: {q}")
    show("Contextual compression", get_compression_retriever().invoke(q))
