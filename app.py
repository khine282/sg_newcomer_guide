"""Web chat for the SG Newcomer Guide.

Run with:  streamlit run app.py

Uses the same Chat class as `python -m rag.chat`, just with a web page
instead of the terminal, and shows which chunks each answer came from.

The public demo runs on the Gemini free tier (about 20 requests per day per
model), so questions are limited per visitor and per day, and each question
is logged without its text.
"""
import datetime
import logging
import os
import re
import threading
import time

import streamlit as st

from rag.chat import Chat

# Each question uses 1-2 Gemini requests (rewrite + answer).
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "8"))
MAX_QUESTIONS_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sg_guide")


@st.cache_resource
def daily_counter():
    """Shared by all visitors (one per app process), reset every day."""
    return {"date": None, "count": 0, "lock": threading.Lock()}


def take_daily_slot():
    c = daily_counter()
    with c["lock"]:
        today = datetime.date.today()
        if c["date"] != today:
            c["date"], c["count"] = today, 0
        if c["count"] >= MAX_QUESTIONS_PER_DAY:
            return False
        c["count"] += 1
        return True


def language(text):
    # Myanmar Unicode block
    return "my" if re.search(r"[က-႟]", text) else "en"

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
    st.session_state.asked = 0


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
        if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
            log.info("limit=session lang=%s", language(question))
            answer, sources = (f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per "
                               "visitor. Please run it locally to ask more."), []
        elif not take_daily_slot():
            log.info("limit=daily lang=%s", language(question))
            answer, sources = ("The demo has reached its daily question limit (free API "
                               "tier). Please try again tomorrow."), []
        else:
            st.session_state.asked += 1
            start = time.perf_counter()
            try:
                with st.spinner("Searching the documents..."):
                    result = st.session_state.chat.ask(question)
                answer, sources = result["answer"], result["context"]
                log.info("ok lang=%s chars=%d docs=%d agencies=%s secs=%.1f",
                         language(question), len(question), len(sources),
                         sorted({d.metadata["agency"] for d in sources}),
                         time.perf_counter() - start)
            except Exception as e:
                # Usually every Gemini model in the fallback list is busy or
                # out of quota.
                log.exception("error lang=%s type=%s secs=%.1f", language(question),
                              type(e).__name__, time.perf_counter() - start)
                answer, sources = ("Sorry, the language model is unavailable right now "
                                   "(free API tier). Please try again later."), []
        st.markdown(answer)
        if sources:
            show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources})
