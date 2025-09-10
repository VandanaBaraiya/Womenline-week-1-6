import re
from src.summarize import summarize_text, get_summarizer

def count_sentences(text: str) -> int:
    return len(re.findall(r'[.!?]', text))

def test_summarize_basic():
    _ = get_summarizer()  # load once
    text = (
        "Today was a busy day. I woke up early, finished my assignments, and "
        "went to the gym. In the evening, I cooked pasta and watched a movie. "
        "Overall, I felt productive and happy."
    )
    summary = summarize_text(text, max_sentences=3)
    assert isinstance(summary, str)
    assert 1 <= count_sentences(summary) <= 3

def test_short_input():
    text = "Too short"
    summary = summarize_text(text)
    assert summary == ""
