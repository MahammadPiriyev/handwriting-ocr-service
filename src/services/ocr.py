import base64
import json
import time
from typing import List

import fitz  # PyMuPDF
import requests
from fastapi import UploadFile

from src.core.config import (
    VISION_RETRIES,
    VISION_TIMEOUT,
    MIN_OCR_CHARS,
    MIN_OCR_LINES,
    MAX_OCR_CHARS_PER_CHUNK,
    GOOGLE_VISION_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    API_VERSION,
    DEPLOYMENT_NAME,
    OCR_SYSTEM_PROMPT
)
from src.utils.text_helpers import _score_text, pre_normalize_ocr, chunk_text


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def _vision_call(feature_type: str, image_bytes: bytes) -> str:
    api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [
            {
                "image": {"content": encode_image_bytes(image_bytes)},
                "features": [{"type": feature_type}],
                "imageContext": {"languageHints": ["az", "ru"]},
            }
        ]
    }

    r = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=VISION_TIMEOUT,
    )

    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"Vision API xətası. Status={r.status_code}\n{data}")

    resp0 = (data.get("responses") or [{}])[0]
    if "error" in resp0:
        raise RuntimeError(f"Vision response error: {resp0['error']}")

    if "fullTextAnnotation" in resp0 and "text" in resp0["fullTextAnnotation"]:
        return resp0["fullTextAnnotation"]["text"] or ""

    ta = resp0.get("textAnnotations") or []
    if ta and "description" in ta[0]:
        return ta[0]["description"] or ""

    return ""

def run_vision_ocr_best(image_bytes: bytes) -> str:
    if not GOOGLE_VISION_API_KEY:
        raise RuntimeError("GOOGLE_VISION_API_KEY env-də set olunmayıb.")

    best_text = ""
    best_score = (0, 0)
    feature_order = ["DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"]

    for attempt in range(1, VISION_RETRIES + 1):
        for ft in feature_order:
            try:
                txt = _vision_call(ft, image_bytes)
                score = _score_text(txt)

                if score > best_score:
                    best_score = score
                    best_text = txt

                if score[0] >= MIN_OCR_CHARS and score[1] >= MIN_OCR_LINES:
                    return txt

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                pass
            except Exception:
                pass

        time.sleep(2 ** (attempt - 1))

    return best_text

def _azure_chat_url() -> str:
    """Azure OpenAI chat completions URL builder."""
    base = (AZURE_OPENAI_ENDPOINT or "").rstrip("/")
    if not base:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT boşdur.")

    if "/openai/" in base:
        base = base.split("/openai/")[0]

    if not DEPLOYMENT_NAME:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT (DEPLOYMENT_NAME) boşdur.")

    if not API_VERSION:
        raise RuntimeError("AZURE_OPENAI_API_VERSION (API_VERSION) boşdur.")

    return f"{base}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"

def azure_chat_completions(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 2500,
    timeout: int = 60,
) -> str:
    url = _azure_chat_url()

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(
            f"Azure chat failed.\n"
            f"URL: {url}\n"
            f"Status: {r.status_code}\n"
            f"Body: {r.text}"
        )

    data = r.json()
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

def azure_fix_text(ocr_text: str, pass_hint: str = "") -> str:
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY env-də set olunmayıb.")
    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT env-də set olunmayıb.")

    user_prompt = f"""
Aşağıdakı mətn OCR nəticəsidir. Qaydaları tətbiq et və yalnız təmizlənmiş yekun mətni qaytar.
{pass_hint}

--- OCR MƏTNİ BAŞLAYIR ---
{ocr_text}
--- OCR MƏTNİ BİTİR ---
""".strip()

    return azure_chat_completions(
        messages=[
            {"role": "system", "content": OCR_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2500,
        timeout=60,
    )

def clean_with_azure_pipeline(raw_ocr: str) -> str:
    if not (raw_ocr or "").strip():
        return ""

    ready = pre_normalize_ocr(raw_ocr)
    parts = chunk_text(ready, MAX_OCR_CHARS_PER_CHUNK)

    cleaned_parts = []
    for i, part in enumerate(parts, 1):
        cleaned_parts.append(azure_fix_text(part, pass_hint=f"(Hissə {i}/{len(parts)})"))

    cleaned = "\n\n".join([p for p in cleaned_parts if p]).strip()
    return cleaned

def render_pdf_pages_to_images(pdf_bytes: bytes, dpi: int) -> List[dict]:
    """Render each PDF page to a separate image.

    Returns a list of dicts: {"bytes": <image bytes>, "mime": "image/png"}.
    Images are encoded as PNG (lossless) so downstream pipelines can rely on
    the mime type and raw bytes for further processing.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        out: List[dict] = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            out.append({"bytes": img_bytes})
        return out
    finally:
        doc.close()

