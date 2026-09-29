from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch

from rag.llm import llm
from rag.qa import DOCUMENT_PROMPT, SYSTEM_PROMPT
from rag.retrieval import get_mmr_retriever

# Follow-up questions like "what about for part-timers?" can't be searched
# on their own, so the LLM first rewrites them into a standalone question.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest question, rewrite the question "
               "so it can be understood without the chat history. Keep the same language. "
               "Do NOT answer it, only rewrite it if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def get_history_aware_retriever():
    """Like create_history_aware_retriever, but converts the rewritten question
    to a plain str. With the current langchain-google-genai the text comes back
    as a TextAccessor object, which the embedding API rejects (500 error)."""
    base = get_mmr_retriever()
    rewrite = CONDENSE_PROMPT | llm | (lambda msg: str(msg.text))
    return RunnableBranch(
        (lambda x: not x.get("chat_history"), (lambda x: x["input"]) | base),
        rewrite | base,
    )


def get_chat_chain():
    retriever = get_history_aware_retriever()
    answer_chain = create_stuff_documents_chain(llm, chat_prompt, document_prompt=DOCUMENT_PROMPT)
    return create_retrieval_chain(retriever, answer_chain)


class Chat:
    """Keeps the conversation history (the 'memory') between questions."""

    def __init__(self, max_turns=5):
        self.chain = get_chat_chain()
        self.history = []
        self.max_turns = max_turns

    def ask(self, question):
        result = self.chain.invoke({"input": question, "chat_history": self.history})
        self.history += [HumanMessage(question), AIMessage(result["answer"])]
        # Only keep the last few turns so the prompt doesn't grow forever.
        self.history = self.history[-2 * self.max_turns:]
        return result


if __name__ == "__main__":
    chat = Chat()
    print("SG Newcomer Guide (type 'quit' to exit)")
    print("Learning project - always confirm on the official websites.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print(f"\nBot: {chat.ask(question)['answer']}\n")
