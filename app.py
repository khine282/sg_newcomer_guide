"""Web chat for the SG Newcomer Guide.

Run with:  streamlit run app.py

Uses the same Chat class as `python -m rag.chat`, just with a web page
instead of the terminal, and shows which chunks each answer came from.
"""
import streamlit as st

from rag.chat import Chat

st.set_page_config(page_title="SG Newcomer Guide", page_icon="🇸🇬")
st.title("🇸🇬 SG Newcomer Guide")
st.caption("Ask about work, CPF, housing, transport or healthcare in Singapore - "
           "in English or Burmese (မြန်မာ).")

with st.sidebar:
    st.markdown("**Learning project.** Answers come only from a small set of official "
                "documents. Always confirm on the official websites.")
    st.markdown("**Try asking:**\n"
                "- How much annual leave do employees get?\n"
                "- What if I have worked less than a year?\n"
                "- စင်ကာပူမှာ ဖုန်းဆင်းကတ် ဘယ်မှာဝယ်ရမလဲ\n"
                "- How do I apply for Singapore citizenship?")
    if st.button("New conversation"):
        st.session_state.clear()

# Streamlit re-runs this whole file on every message, so the chat (and its
# memory) is kept in session_state instead of being created again each time.
if "chat" not in st.session_state:
    st.session_state.chat = Chat()
    st.session_state.messages = []


def show_sources(docs):
    with st.expander(f"Retrieved chunks ({len(docs)})"):
        for d in docs:
            m = d.metadata
            text = " ".join(d.page_content.split())
            st.markdown(f"**{m['agency']}** · {m['source_type']} · {m['topic']}  \n"
                        f"{m['source_url']}  \n> {text[:300]}...")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            result = st.session_state.chat.ask(question)
        st.markdown(result["answer"])
        show_sources(result["context"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["context"]})
