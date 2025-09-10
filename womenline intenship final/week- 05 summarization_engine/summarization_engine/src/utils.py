import re

_SENTENCE_END_RE = re.compile(r'([.!?])\s+')

def clean_text(text: str) -> str:
    # Basic cleanup; adjust to your needs
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def limit_sentences(text: str, max_sentences: int = 3) -> str:
    # Naive splitter to cap the number of sentences
    if not text:
        return text
    parts = _SENTENCE_END_RE.split(text)
    # parts looks like: ['Sentence', '.', ' next', '.', ' rest', '!', ' ...']
    # Reassemble while counting enders
    out = []
    enders = 0
    for chunk in parts:
        out.append(chunk)
        if chunk in {'.', '!', '?'}:
            enders += 1
            if enders >= max_sentences:
                break
    result = ''.join(out).strip()
    # Ensure it ends with a sentence terminator
    if result and result[-1] not in '.!?':
        result += '.'
    return result
