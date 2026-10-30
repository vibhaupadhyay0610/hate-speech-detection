import re

def clean_text(text: str) -> str:
    """Noise reduction: lowercase, strip URLs/mentions/hashtags/punctuation, collapse spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
