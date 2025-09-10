from typing import Optional
import os

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .config import MODEL_NAME, MAX_SUMMARY_SENTENCES, DEVICE
from .utils import clean_text, limit_sentences

_SUMMARIZER = None

def _detect_device():
    if DEVICE:
        return DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_summarizer(model_name: Optional[str] = None):
    global _SUMMARIZER
    if _SUMMARIZER is not None and model_name in (None, MODEL_NAME):
        return _SUMMARIZER
    name = model_name or MODEL_NAME
    device = 0 if _detect_device() == "cuda" else -1

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    _SUMMARIZER = summarizer
    return _SUMMARIZER

def summarize_text(text: str, max_sentences: Optional[int] = None) -> str:
    text = clean_text(text)
    if not text or len(text) < 10:
        return ""

    summarizer = get_summarizer()

    # You can tweak these per model if needed.
    # For BART/PEGASUS: reasonable defaults for concise output
    result = summarizer(
        text,
        max_length=160,
        min_length=30,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    raw = result[0]["summary_text"].strip()
    return limit_sentences(raw, max_sentences or MAX_SUMMARY_SENTENCES)
