"""
FastAPI backend exposing the RAG pipeline over HTTP for the React frontend.

Run with: uvicorn main:app --reload --port 8000
(run this from inside the src/ folder)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generate import answer_question
from vector_store import add_pdf_to_store, delete_document
from retrieve import list_available_sources
import os
import secrets
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG Q&A API")

DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD")
AUTH_REQUIRED = os.environ.get("REQUIRE_DEMO_AUTH", "").lower() == "true"
if AUTH_REQUIRED and not DEMO_PASSWORD:
    raise RuntimeError("DEMO_PASSWORD must be set when REQUIRE_DEMO_AUTH=true.")
AUTH_REQUIRED = AUTH_REQUIRED or bool(DEMO_PASSWORD)
basic_auth = HTTPBasic(auto_error=False)


def require_demo_auth(
    credentials: HTTPBasicCredentials | None = Depends(basic_auth),
) -> None:
    if not AUTH_REQUIRED:
        return

    username_valid = credentials is not None and secrets.compare_digest(
        credentials.username.encode("utf-8"), DEMO_USERNAME.encode("utf-8")
    )
    password_valid = credentials is not None and secrets.compare_digest(
        credentials.password.encode("utf-8"), (DEMO_PASSWORD or "").encode("utf-8")
    )
    if not username_valid or not password_valid:
        raise HTTPException(
            status_code=401,
            detail="Demo credentials are missing or invalid.",
            headers={"WWW-Authenticate": "Basic"},
        )


# Allow the React frontend to call this API.
# Locally this defaults to the Vite dev server; in production, set
# FRONTEND_URL as an environment variable to your deployed frontend's URL.
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


class QuestionRequest(BaseModel):
    question: str
    source: str | None = None  # optional: restrict search to one document


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
)


@app.post(
    "/upload",
    response_model=UploadResponse,
    dependencies=[Depends(require_demo_auth)],
)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(save_path, "wb") as f:
            f.write(contents)

        # Chunk + embed + add ONLY this file's new chunks (incremental logic)
        chunks_added = add_pdf_to_store(save_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    if chunks_added == 0:
        message = "This document was already indexed - nothing new to add."
    else:
        message = "Indexed successfully. You can now ask questions about it."

    return UploadResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message=message
    )


@app.post(
    "/ask",
    response_model=AnswerResponse,
    dependencies=[Depends(require_demo_auth)],
)
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = answer_question(question, source=request.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Covers: no collection indexed yet, LLM API errors, network issues, etc.
        raise HTTPException(
            status_code=502,
            detail=f"Something went wrong generating an answer: {e}"
        )

    return AnswerResponse(answer=answer, sources=sources)


@app.get("/documents", dependencies=[Depends(require_demo_auth)])
def get_documents():
    """List currently indexed document filenames - powers the frontend filter dropdown."""
    try:
        return {"documents": list_available_sources()}
    except Exception:
        # No collection exists yet (nothing uploaded) - not a real error
        return {"documents": []}


@app.delete(
    "/documents/{filename}",
    dependencies=[Depends(require_demo_auth)],
)
def remove_document(filename: str):
    """Remove a document's chunks from the vector store and delete its file."""
    chunks_removed = delete_document(filename)
    if chunks_removed == 0:
        raise HTTPException(status_code=404, detail=f"'{filename}' was not found in the index.")
    return {"filename": filename, "chunks_removed": chunks_removed}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend():
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Frontend is not built.")
    return FileResponse(index_file)