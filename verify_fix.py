
from typing import List, Dict, Any
from src.services.llm import find_relevant_sources

def test_find_relevant_sources():
    # Setup dummy data
    long_text = "start " + ("padding " * 200) + "TARGET_KEYWORD answer is here " + ("padding " * 200) + " end"
    assert len(long_text) > 1000, "Setup error: text too short"
    
    pages = [{
        "page_number": 1,
        "MD_text": long_text
    }]
    
    # Test
    sources = find_relevant_sources("TARGET_KEYWORD", "test.pdf", pages)
    
    # Verify
    if not sources:
        print("FAIL: No sources found")
        return
        
    content = sources[0]["content"]
    print(f"Content length: {len(content)}")
    
    if "TARGET_KEYWORD" in content:
        print("PASS: Keyword found in content")
    else:
        print("FAIL: Keyword NOT found in content")
        print(f"Content start: {content[:100]}...")

    if "answer is here" in content:
        print("PASS: Context around keyword found")
    else:
        print("FAIL: Context around keyword NOT found")

if __name__ == "__main__":
    test_find_relevant_sources()
