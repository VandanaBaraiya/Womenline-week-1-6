from pydantic import BaseModel, Field

class JournalRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw journal text to summarize")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="1–3 sentence concise summary")
