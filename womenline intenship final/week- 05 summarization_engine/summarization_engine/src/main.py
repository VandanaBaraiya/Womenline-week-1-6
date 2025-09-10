from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import JournalRequest, SummaryResponse
from .summarize import summarize_text, get_summarizer

app = FastAPI(title="AI Summarization Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    # Warm up model at startup (optional but reduces first-request latency)
    try:
        get_summarizer()
    except Exception as e:
        # Don't crash if model can't load at startup (CI env, etc.)
        print(f"[WARN] Could not preload summarizer: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize-journal", response_model=SummaryResponse)
def summarize(req: JournalRequest):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short to summarize.")
    try:
        summary = summarize_text(req.text)
        if not summary:
            raise ValueError("Empty summary")
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
