# Import async context manager helper for FastAPI lifespan hooks
from contextlib import asynccontextmanager
# Import FastAPI application class
from fastapi import FastAPI
# Import Pydantic base model for request validation
from pydantic import BaseModel
# Import local RAG engine
from rag import RAGEngine
# Import async helper that talks to Ollama
from llm_client import ask_llm

# Create one shared RAG engine at import time (loads docs once)
rag = RAGEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup hook placeholder (index already built in RAGEngine.__init__)
    yield
    # Shutdown hook placeholder


# Create the FastAPI app with the lifespan handler attached
app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    # Incoming user / vision message text
    message: str


@app.get("/health")
def health():
    # Simple liveness endpoint for checks and debugging
    return {
        # Generic OK flag
        "status": "ok",
        # Service name for clarity
        "service": "ai-service",
        # Whether RAG has an in-memory index
        "rag_ready": rag.index is not None or bool(rag.doc_by_stem),
        # How many knowledge documents are loaded by filename stem
        "docs": len(rag.doc_by_stem),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    # Normalize the incoming message
    message = (req.message or "").strip()
    # Reject empty messages early
    if not message:
        # Return a stable JSON error payload
        return {"response": "Empty message.", "used_rag": False}

    # Try to retrieve local knowledge for this message / label
    context = rag.retrieve(message)

    # If knowledge was found, force the LLM to use only that context
    if context:
        # Strict grounded prompt: no outside knowledge
        prompt = (
            "You are a concise assistant for an enterprise object-description system.\n"
            "Use ONLY the context below. Do NOT use outside or prior knowledge.\n"
            "If the context is incomplete, still answer only from what is written there.\n"
            "Answer in 1-2 short sentences.\n\n"
            f"Context:\n{context}\n\n"
            f"Item or question: {message}\n"
            "Answer:"
        )
    else:
        # No local knowledge: allow a brief general fallback answer
        prompt = (
            "Answer briefly in 1-2 short sentences.\n"
            "If you are unsure, say you do not have specific information.\n\n"
            f"Question: {message}\n"
            "Answer:"
        )

    # Ask the local LLM for the final wording
    answer = await ask_llm(prompt)
    # Fallback text if the model returns nothing
    if not answer:
        # Prefer showing raw context rather than an empty overlay string
        answer = context if context else "No response from model."

    # Return both the answer and whether RAG context was used
    return {
        # Final natural-language answer
        "response": answer,
        # True only when local knowledge was injected into the prompt
        "used_rag": context is not None,
    }
