from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.api.models import ChatMessage
from src.core.config import DPI
from src.services.ocr import (
    render_pdf_pages_to_images,
    run_vision_ocr_best,
    clean_with_azure_pipeline
)
from src.services.ocr2 import run_ocr2_pipeline
from src.services.storage import save_ocr_document, get_all_documents
from src.services.llm import find_relevant_sources, generate_chat_response
from fastapi.responses import RedirectResponse


app = FastAPI(title="SOCAR OCR + LLM API", version="1.0.0")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=302)

@app.post("/ocr/v1")
async def ocr_endpoint(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts a PDF file upload and returns the extracted Markdown text for each page.
    """
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    
    is_pdf_file = filename.endswith(".pdf") or content_type == "application/pdf"
    
    # Simple check for supported images if not PDF
    # The pipeline logic handles bytes, but simple MIME check helps.
    if not is_pdf_file and "image/" not in content_type:
         # Fallback check by extension if content-type is octet-stream
         known_exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
         if not any(filename.endswith(ext) for ext in known_exts):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF or Image files are supported."
            )

    raw = await file.read()
    
    try:
        if is_pdf_file:
            pages = render_pdf_pages_to_images(raw, DPI)
        else:
            pages = [{"bytes": raw, "mime": content_type if "image/" in content_type else "image/png"}]

        results: List[Dict[str, Any]] = []

        for idx, page in enumerate(pages, start=1):
            raw_txt = run_vision_ocr_best(page["bytes"])
            cleaned = clean_with_azure_pipeline(raw_txt) if (raw_txt or "").strip() else ""
            results.append({"page_number": idx, "MD_text": cleaned})
        
        save_ocr_document(file.filename, results)
        
        return results
        
    except Exception as e:
        print(f"OCR failed with error: {repr(e)}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {repr(e)}")


@app.post("/ocr/v2")
async def ocr2_endpoint(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts a PDF or Image file upload.
    Uses OpenRouter (Ministral) + Azure (Llama) pipeline.
    Returns the extracted Markdown text for each page.
    """
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    
    is_pdf_file = filename.endswith(".pdf") or content_type == "application/pdf"
    
    # Simple check for supported images if not PDF
    # The pipeline logic handles bytes, but simple MIME check helps.
    if not is_pdf_file and "image/" not in content_type:
         # Fallback check by extension if content-type is octet-stream
         known_exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
         if not any(filename.endswith(ext) for ext in known_exts):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF or Image files are supported."
            )

    raw = await file.read()
    results: List[Dict[str, Any]] = []

    try:
        if is_pdf_file:
            pages = render_pdf_pages_to_images(raw, DPI)
            for idx, page in enumerate(pages, start=1):
                # Run OCR2 pipeline on each page image (mime is provided per page)
                cleaned = run_ocr2_pipeline(page["bytes"], mime_type=page.get("mime", "image/png"))
                results.append({"page_number": idx, "MD_text": cleaned})
        else:
            # Single image
            mime = content_type if "image/" in content_type else "image/png"
            cleaned = run_ocr2_pipeline(raw, mime_type=mime)
            results.append({"page_number": 1, "MD_text": cleaned})

        # Save result to storage (reusing existing storage logic)
        save_ocr_document(file.filename, results)
        
        return results
        
    except Exception as e:
        print(f"OCR2 failed with error: {repr(e)}")
        raise HTTPException(status_code=500, detail=f"OCR2 failed: {repr(e)}")


@app.post("/llm")
async def llm_endpoint(messages: List[ChatMessage]) -> Dict[str, Any]:
    if not messages:
        raise HTTPException(status_code=400, detail="Empty chat history")

    last_user_message = next(
        (msg.content for msg in reversed(messages) if msg.role == "user"),
        None
    )

    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    all_sources = []
    try:
        all_docs = get_all_documents()
        for doc_name, ocr_results in all_docs:
            sources = find_relevant_sources(
                query=last_user_message,
                pdf_name=doc_name,
                pages=ocr_results
            )
            all_sources.extend(sources)

    except Exception as e:
        # Assuming get_all_documents might raise an exception
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    answer = generate_chat_response(messages, all_sources)

    return {
        "sources": all_sources,
        "answer": answer
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "PDF OCR + LLM API",
        "version": "1.0.0"
    }
