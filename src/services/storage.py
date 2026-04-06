import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

try:
    import psycopg2
except ImportError:
    psycopg2 = None

from src.core.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS,
    OCR_DOCS_DIR,
)

def get_db_connection():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 module not found. Database features are unavailable.")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def save_ocr_document(pdf_name: str, ocr_results: list) -> str:
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO history (id, doc_id, doc_name, ocr_results, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                doc_id,
                pdf_name,
                json.dumps(ocr_results),
                created_at
            )
        )

        conn.commit()
        cursor.close()
        conn.close()

        return doc_id

    except Exception as e:
        print("Error saving to database:", e)
        return ""

def load_ocr_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Load OCR document by ID"""
    doc_path = OCR_DOCS_DIR / f"{doc_id}.json"
    if not doc_path.exists():
        return None
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_documents() -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Fetches all documents from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT doc_name, ocr_results FROM history")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows

    except Exception as e:
        print(f"Database error while fetching all documents: {e}")
        # In a real app, you'd raise a custom exception
        raise
