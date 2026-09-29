"""RAG triad evaluation (Course 3), using an LLM as the judge.

- Answer relevance:  does the answer address the question?
- Context relevance: is each retrieved chunk relevant to the question?
- Groundedness:      is every claim in the answer supported by the retrieved chunks?

Each score is 0.0-1.0. The course uses TruLens for this; we write the three
judges ourselves so we can see the exact prompts.

The judge is a local model (Ollama), because the Gemini free tier only allows
20 requests per day per model - not enough for ~36 grading calls per run.
The chatbot itself still uses Gemini (12 calls per run).
"""
import json
import time
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_MODEL = "qwen3:8b"
# num_gpu=0 runs on the CPU: the GPU fails with a CUDA error on the current
# NVIDIA driver (546.18). After a driver update, remove it to use the GPU.
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0, reasoning=False, num_gpu=0)

# Pause between questions so the chatbot stays under Gemini's per-minute limit.
DELAY_SECONDS = 10


class Score(BaseModel):
    reason: str = Field(description="One or two sentences explaining the score")
    score: float = Field(description="Score from 0.0 (worst) to 1.0 (best)")


def make_judge(instructions):
    prompt = ChatPromptTemplate.from_messages([
        ("system", instructions + "\nAlways give a score between 0.0 and 1.0."),
        ("human", "{input}"),
    ])
    return prompt | judge_llm.with_structured_output(Score)


def grade(judge, text):
    # Small local models sometimes ignore the scale, so keep scores in 0-1.
    return min(max(judge.invoke({"input": text}).score, 0.0), 1.0)


answer_relevance_judge = make_judge(
    "You grade a chatbot. Given a QUESTION and an ANSWER, score how well the answer "
    "addresses the question (1.0 = fully and directly answers it). The question and "
    "answer may be in English or Burmese. An honest 'I don't know' to a question "
    "that cannot be answered is fine and should score at least 0.5."
)
context_relevance_judge = make_judge(
    "You grade a search system. Given a QUESTION and several retrieved CONTEXT chunks, "
    "score what fraction of the chunks are relevant for answering the question "
    "(1.0 = all directly useful, 0.0 = all unrelated). The question may be in Burmese."
)
groundedness_judge = make_judge(
    "You check a chatbot for hallucinations. Given CONTEXT and an ANSWER, score what "
    "fraction of the factual claims in the answer are supported by the context "
    "(1.0 = everything supported). Ignore source lists, disclaimers and "
    "'I don't know' statements - they are not claims."
)


def evaluate_one(question, result):
    answer = result["answer"]
    context = "\n---\n".join(
        f"[chunk {i + 1}] {d.page_content}" for i, d in enumerate(result["context"]))
    return {
        "question": question,
        "answer": answer,
        "chunk_ids": [d.metadata.get("chunk_id") for d in result["context"]],
        "answer_relevance": grade(answer_relevance_judge,
                                  f"QUESTION: {question}\n\nANSWER: {answer}"),
        "context_relevance": grade(context_relevance_judge,
                                   f"QUESTION: {question}\n\nCONTEXT:\n{context}"),
        "groundedness": grade(groundedness_judge,
                              f"CONTEXT:\n{context}\n\nANSWER: {answer}"),
    }


def evaluate(chain, name):
    """Run every test question through `chain`, score it, and save the results."""
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    rows = []
    for i, q in enumerate(questions):
        if i > 0:
            time.sleep(DELAY_SECONDS)
        result = chain.invoke({"input": q["question"]})
        row = {"topic": q["topic"], **evaluate_one(q["question"], result)}
        rows.append(row)
        print(f"[{i + 1}/{len(questions)}] ans {row['answer_relevance']:.2f} | "
              f"ctx {row['context_relevance']:.2f} | grd {row['groundedness']:.2f} | "
              f"{q['question'][:50]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n{name} averages:")
    for metric in ["answer_relevance", "context_relevance", "groundedness"]:
        print(f"  {metric:18} {sum(r[metric] for r in rows) / len(rows):.2f}")
    print(f"Saved to {path}")
    return rows


if __name__ == "__main__":
    from rag.qa import get_qa_chain

    evaluate(get_qa_chain(), "baseline")
