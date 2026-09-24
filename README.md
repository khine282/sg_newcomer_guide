# SG Newcomer Guide

A multilingual (English/Burmese) RAG chatbot answering practical
"living in Singapore" questions from official sources.

Learning project built after DeepLearning.AI's
"Building and Evaluating Advanced RAG" and
"LangChain Chat with Your Data" courses.

⚠️ Learning project. Always confirm information on the official websites.


## Progress

### Phase 1: Document loading
- Sources: __ PDF pages, __ web pages, __ videos, __ Notion page(s)
- Loaders used: PyPDFLoader, WebBaseLoader, GenericLoader + local Whisper, NotionDirectoryLoader
- Every document has metadata: topic, agency, source type, source URL, last checked date
- Problems I hit:
  - Some government pages (MOM, HDB) load content with JavaScript, so the loader got 0 characters. I removed them and used PDFs or other official pages instead
  - Web pages included menus and footers; I removed nav/header/footer tags before extracting text
  - A 33 second promo video gave almost no transcript (Whisper only transcribes speech); I replaced it with a longer explainer
  - Some PDF pages were nearly empty; I filter out documents under 100 characters
- Data is imbalanced: most documents come from MOM PDFs
- What I learned: (your own words)