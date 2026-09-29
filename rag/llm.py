from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Free-tier Gemini models often return 503 ("high demand"), sometimes for
# minutes. Instead of waiting, try the next model in the list.
MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]


def make_model(name):
    # temperature=0 so answers are repeatable while we test.
    return ChatGoogleGenerativeAI(model=name, temperature=0, max_retries=1)


llm = make_model(MODELS[0]).with_fallbacks([make_model(m) for m in MODELS[1:]])
