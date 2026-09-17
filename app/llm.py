"""
Sends the retrieved PDF context and the user's question to Gemini.

Why this file exists:
This is the "generation" half of RAG (Retrieval-Augmented Generation).
Retrieval (Qdrant) finds relevant chunks; this file's job is to hand
those chunks to an LLM along with strict instructions to answer only
from them, so the model doesn't invent facts that aren't in the PDF.
"""

import google.generativeai as genai

from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-1.5-flash")

NOT_FOUND_MESSAGE = "Not found in this document."

PROMPT_TEMPLATE = """You are answering questions using ONLY the context below, \
which was extracted from a PDF document. Do not use any outside knowledge.

If the answer is not contained in the context, respond with exactly:
"{not_found}"

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Asks Gemini to answer a question using only the given chunks as context.

    Args:
        question: the user's question.
        chunks: retrieved chunks, each {"page_number": int, "text": str, ...}.
            Order matters a little (most relevant first), but the model
            sees all of them together, not one at a time.

    Returns:
        The model's answer as plain text.
    """
    context = "\n\n".join(
        f"[Page {chunk['page_number']}]\n{chunk['text']}" for chunk in chunks
    )

    prompt = PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE, context=context, question=question
    )

    response = _model.generate_content(prompt)
    return response.text.strip()
