import os
import time
import logging
from contextlib import asynccontextmanager
from threading import Lock

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field

from src.search import CHROMA_PATH, COLLECTION_NAME, GROQ_MODEL, build_context, generate_answer, load_embedding_model, retrieve_documents
from src.dashboard import configure_dashboard, dashboard_router, log_event

resources = {}
inference_lock = Lock()
configure_dashboard(resources, inference_lock)

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing from the environment.")
    print("Initializing Groq client...")
    groq_client = Groq(api_key=api_key, timeout=30, max_retries=0)
    print(f"Opening ChromaDB at {CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = db.get_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError("The ChromaDB collection contains no documents.")
    print(f"Loading embedding model: {collection.count()} indexed chunks")
    embedding_model, device = load_embedding_model()
    resources.update(groq_client=groq_client, collection=collection, embedding_model=embedding_model, device=device)
    log_event(logging.INFO, "api_initialized", model=GROQ_MODEL, device=device, indexed_chunks=collection.count())
    print(f"API initialized. Model={GROQ_MODEL}, device={device}")
    yield
    resources.clear()
    log_event(logging.INFO, "api_shutdown")

app = FastAPI(title="JS Bank RAG API", description="Question-answering API backed by the bank knowledge base.", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(dashboard_router)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
class SourceItem(BaseModel):
    label: str
    source: str
    page: int | str | None = None
    chunk_index: int | str | None = None
class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    response_time_seconds: float

@app.get("/health")
def health():
    if not resources:
        raise HTTPException(status_code=503, detail="API is initializing.")
    return {"status":"ok", "indexed_chunks":resources["collection"].count(), "embedding_device":resources["device"], "answer_model":GROQ_MODEL}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty or whitespace.")
    if not resources:
        raise HTTPException(status_code=503, detail="API is not ready.")
    started = time.perf_counter()
    request_id = os.urandom(8).hex()
    log_event(logging.INFO, "chat_started", request_id=request_id)
    try:
        with inference_lock:
            retrieved = retrieve_documents(question, resources["embedding_model"], resources["collection"])
            context, sources = build_context(retrieved)
            if not context.strip():
                raise HTTPException(status_code=503, detail="No usable reference documents were retrieved.")
            answer = generate_answer(question, context, resources["groq_client"])
        elapsed = round(time.perf_counter() - started, 3)
        log_event(logging.INFO, "chat_completed", request_id=request_id, response_time_seconds=elapsed, source_count=len(sources))
        return ChatResponse(answer=answer, sources=[SourceItem(label=s["label"], source=str(s["source"]), page=s.get("page"), chunk_index=s.get("chunk_index")) for s in sources], response_time_seconds=elapsed)
    except HTTPException:
        log_event(logging.WARNING, "chat_http_error", request_id=request_id, status_code=503)
        raise
    except RateLimitError:
        log_event(logging.WARNING, "chat_rate_limited", request_id=request_id)
        raise HTTPException(status_code=429, detail="The AI service rate limit was reached. Please try again shortly.")
    except APIConnectionError:
        log_event(logging.ERROR, "chat_connection_error", request_id=request_id)
        raise HTTPException(status_code=502, detail="Could not connect to the AI service. Please try again.")
    except APIStatusError as exc:
        log_event(logging.ERROR, "chat_provider_error", request_id=request_id, provider_status=exc.status_code)
        raise HTTPException(status_code=502, detail="The AI service returned an error. Please try again.")
    except Exception as exc:
        log_event(logging.ERROR, "chat_internal_error", request_id=request_id, error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="An internal error occurred while answering the question.")
