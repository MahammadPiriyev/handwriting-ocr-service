from typing import List, Dict, Any

import requests
import time

from src.core.config import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL
)


def find_relevant_sources(query: str, pdf_name: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find relevant text snippets from document pages based on query"""
    sources = []
    query_lower = query.lower()
    
    for page in pages:
        page_num = page.get("page_number", 0)
        text = page.get("MD_text", "")
        text_lower = text.lower()
        
        # Check for exact match or keyword match
        match_index = -1
        
        if query_lower in text_lower:
            match_index = text_lower.find(query_lower)
        else:
            # Fallback: find first occurrence of any significant word
            for word in query_lower.split():
                if len(word) > 3 and word in text_lower:
                    match_index = text_lower.find(word)
                    break
        
        if match_index != -1:
            # Found a match. Extract context around it.
            # We want roughly 1000 chars total, centered on match if possible
            # But if text is short (< 2000 chars), just return whole text to be safe
            if len(text) < 2000:
                snippet = text
            else:
                start_idx = max(0, match_index - 500)
                end_idx = min(len(text), match_index + 1000)
                snippet = text[start_idx:end_idx]
                if start_idx > 0:
                    snippet = "..." + snippet
                if end_idx < len(text):
                    snippet = snippet + "..."

            sources.append({
                "pdf_name": pdf_name,
                "page_number": page_num,
                "content": snippet
            })
    
    return sources[:3]  # Return max 3 sources

def generate_chat_response(messages: List[Dict[str, str]], sources: List[Dict[str, Any]]) -> str:
    """Generate LLM response based on conversation history and sources"""
    if not messages:
        return "Zəhmət olmasa sual verin."

    # Prepare context from sources
    if not sources:
        context_text = "Mənbələrdə bu sualla bağlı məlumat tapılmadı."
    else:
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Mənbə {i} (Səhifə {source['page_number']}):\n{source['content']}\n")
        context_text = "\n".join(context_parts)
    
    # System prompt for LLM
    system_prompt = """Sən Azərbaycan dilində cavab verən bir köməkçi assistantsan. 
Aşağıdakı mənbələrə və istifadəçi ilə olan söhbət tarixçəsinə əsaslanaraq cavab ver.

Mənbələr:
{context}

Qaydalar:
1. Cavabını əsasən yuxarıdakı mənbələr əsasında ver.
2. Əgər mənbələrdə cavab yoxdursa, ancaq söhbət tarixçəsində varsa (məsələn "mənim adım nədir?"), tarixçəyə əsasən cavab ver.
3. Cavabları Azərbaycan dilində ver.
4. Faktları dəqiq təqdim et.""".format(context=context_text)
    
    # Call OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHAT_API_KEY}"
    }
    
    # Construct messages payload: [System, ...History]
    # Ensure messages are in the format {"role": "...", "content": "..."}
    payload_messages = [{"role": "system", "content": system_prompt}]
    
    # LIMIT HISTORY to last 10 messages to avoid large payloads / token limits
    recent_messages = messages[-10:]
    
    # Validate and append user messages
    for msg in recent_messages:
        # Pydantic model to dict if needed, or dict cleanup
        if hasattr(msg, "dict"):
            m = msg.dict()
        else:
            m = dict(msg)
            
        # Ensure only allowed roles (user/assistant) are passed if needed, 
        # generally OAI accepts user/assistant/system.
        payload_messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })

    payload = {
        "model": CHAT_MODEL,
        "messages": payload_messages,
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{CHAT_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                # Rate limit hit
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                else:
                    return "Sistem çox yüklənib (Rate Limit), zəhmət olmasa bir az sonra yenidən cəhd edin."
            else:
                return f"LLM API xətası: {response.status_code}"
                
        except Exception as e:
            return f"LLM çağırışı zamanı xəta: {str(e)}"
    
    return "Xəta baş verdi."