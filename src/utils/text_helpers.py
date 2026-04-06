import re
from typing import List, Tuple

def _score_text(t: str) -> Tuple[int, int]:
    t = (t or "").strip()
    lines = [x for x in t.splitlines() if x.strip()]
    return (len(t), len(lines))

def pre_normalize_ocr(text: str) -> str:
    t = (text or "").strip()
    t = (
        t.replace("\u2010", "-")
         .replace("\u2011", "-")
         .replace("\u2012", "-")
         .replace("\u2013", "–")
         .replace("\u2014", "–")
    )
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    fixes = [
        (r"\bCOz\b", "CO2"),
        (r"\bSIO2\b", "SiO2"),
        (r"\bC02\b", "CO2"),
        (r"\bSİO2\b", "SiO2"),
    ]
    for p, rpl in fixes:
        t = re.sub(p, rpl, t)
    return t

def chunk_text(text: str, max_chars: int = 9000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts, buf, buf_len = [], [], 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        add = len(p) + 2
        if buf and buf_len + add > max_chars:
            parts.append("\n\n".join(buf).strip())
            buf, buf_len = [p], len(p)
        else:
            buf.append(p)
            buf_len += add
    if buf:
        parts.append("\n\n".join(buf).strip())
    return parts
