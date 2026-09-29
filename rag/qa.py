from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from rag.llm import llm
from rag.retrieval import get_mmr_retriever

# Each chunk is shown to the LLM with where it came from, so it can cite it.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {agency} ({source_type}) {source_url}]\n{page_content}"
)

SYSTEM_PROMPT = """You help newcomers with practical questions about living in Singapore.
Use ONLY the context below to answer. If the context does not contain the answer,
say you don't know and suggest the official agency website. Do not make up numbers.

Answer in the same language as the question (English or Burmese).
Keep the answer short and practical. At the end, list the sources you used
(agency and URL). Tips from "personal" sources are personal experience, not official.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


def get_qa_chain(retriever=None):
    """Retrieve chunks, 'stuff' them all into one prompt, and let the LLM answer."""
    answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=DOCUMENT_PROMPT)
    return create_retrieval_chain(retriever or get_mmr_retriever(), answer_chain)


def show_answer(result):
    print(result["answer"])
    print("\n  Retrieved chunks:",
          ", ".join(f"#{d.metadata['chunk_id']} ({d.metadata['topic']})" for d in result["context"]))


if __name__ == "__main__":
    qa = get_qa_chain()
    for q in [
        "How much annual leave do employees get?",
        "စင်ကာပူမှာ ဖုန်းဆင်းကတ် ဘယ်မှာဝယ်ရမလဲ",  # Where do I buy a SIM card?
        "How do I apply for Singapore citizenship?",  # not in our documents
    ]:
        print(f"\n=== Q: {q}\n")
        show_answer(qa.invoke({"input": q}))
