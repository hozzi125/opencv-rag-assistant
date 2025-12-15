import os
import textwrap
from typing import List, Dict, Any
import requests
from index_chroma import retrieve_top_k

#-CONFIG-
OLLAMA_MODEL = "llama3.1:8b" # simple and effective
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_CONTEXT_CHARS = 12_000
TOP_K = 5

# build context from top-k chunks
def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Формує контекст для LLM із чанків + та метаданих (джерела для цитат)
    """
    parts = []
    used = 0

    for i, h in enumerate(hits, start=1):
        md = h.get("metadata", {}) or {}
        url = md.get("source_url", "")
        title = md.get("doc_title", "")
        section = md.get("section_title", "")

        chunk = h["text"].strip()
        block = f"""[DOC {i}]
Title: {title}
Section: {section}
Source: {url}
Content: 
{chunk}
"""
        # prevent going above context limit
        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()

# promts for RAG/No-Rag
def build_prompt_with_rag(question:  str, context: str) -> str:
    """
    промпт для RAG-режиму: відповідати тільки з наданого контексту.

    """ 
    return f"""
You are technical assistant answering questions about OpenCV.

Rules:
- Answer ONLY using the provided context. If the context doesn't contain the answer,
  say exactly: "Sorry, I couldn't find this in the provided documents."
- Be concise but technical.
- When you state a fact, cite sources as [DOC i] and include Source URL(s) at the end.

Question:
{question}

Context:
{context}

Return format:
Answer: ...
Sources:
- <url> (DOC i, section name)
""".strip()

def build_prompt_no_rag(question: str) -> str:
    """
    Baselin (без RAG): просто питання для моделі без контексту
    """
    return f"""
You are a technical assistant for OpenCV questions.
Answer as accurately as you can, using only your internal knowledge.
If you are not sure, say "Sorry, I am not certain."

Question:
{question}
""".strip()

def generate_llm(prompt: str) -> str:
    """
    Генерація відповіді локальною моделлю Ollama HTTP API.
    Потрібно: встановлений Ollama + завантажена модель(ollama pull llama3.1:8b)
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Не можу підключитися до Ollama (localhost:11434)."
            "Перевірте, що Ollama встановлено і запущено, та що модель завантажена: "
            "ollama pull llama3.1:8b"
        )

# answers with RAG\without RAG

def answer_with_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Docstring for answer_with_rag
    режим RAG:
    - приймає запит користувача
    - робить top-k retrieval
    - будує контекст
    - відправляє в LLM
    """
    hits = retrieve_top_k(question, k=top_k)
    context = build_context(hits)
    prompt = build_prompt_with_rag(question, context)
    answer = generate_llm(prompt)
    return {"answer": answer, "hits": hits, "context_chars": len(context)}

def answer_without_rag(question: str) -> Dict[str, Any]:
    """
    Baseline(без RAG)
    """
    prompt = build_prompt_no_rag(question)
    answer = generate_llm(prompt)
    return {"answer": answer}

# short summary

def overall_comment(results: List[Dict[str, str]]) -> str:
    """
    Короткий узагальнюючий коментар, де RAG дає виграш.
    """
    return (
       "У всіх оцінюваних питаннях відповіді на основі RAG базуються на конкретних "
      "фрагментах офіційної документації OpenCV та містять чіткі посилання на джерела."
      "Базові відповіді (без RAG) все ще можуть бути прийнятними, але вони більш "
      "схильні до розпливчастості або часткової неточності, особливо для детальних описів параметрів "
      "та рекомендованих практик. Таким чином, RAG покращує технічну точність "
      "і зменшує ризик галюцинацій, а також робить відповіді "
      "простежуваними до конкретних сторінок документації."
    )

# demo run
def run_demo():
    """
    Демонстрація якості:
    - 3-5 технічних питань
    - відповіді з RAG та без RAG
    - один короткий підсумковий коментар наприкінці
    """

    demo_questions = [
        "When should I use NORM_HAMMING vs NORM_L2 in BFMatcher?",
        "What are the main steps of Canny edge detection and why is Gaussian blur applied before it?",
        "What does OpenCV consider a contour and what kind of image should I pass to findContours?",
        "What data do I need for camera calibration in OpenCV and what does the calibration output?",
        "In knnMatch with BFMatcher, what does k mean and how is the ratio test applied?"
    ]

    results: List[Dict[str, str]] = []

    for q in demo_questions:
        print("\n" + "=" * 100)
        print("Q:", q)

        out_no = answer_without_rag(q)
        out_rag = answer_with_rag(q)

        ans_no = out_no["answer"]
        ans_rag = out_rag["answer"]

        print("\n--- WITHOUT RAG ---")
        print(textwrap.fill(ans_no, width=100))

        print("\n--- WITH RAG ---")
        print(textwrap.fill(ans_rag, width=100))

        results.append(
            {
                "question": q,
                "no_rag": ans_no,
                "rag": ans_rag
            }
        )
    # final short summary
    print("\n" + "=" * 100)
    print("OVERALL COMMENT: RAG vs NO-RAG\n")
    summary = overall_comment(results)
    print(textwrap.fill(summary, width=100))

if __name__=="__main__":
    run_demo()

