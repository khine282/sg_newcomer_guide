from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# gemini-3.8-flash / 3.7-flash kept returning 503 (high demand) on the
# free tier, so we use the older, less busy 3.5-flash.
# temperature=0 so answers are repeatable while we test.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
