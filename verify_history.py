
from typing import List, Dict, Any
from src.services.llm import generate_chat_response

# Mock Chat API (since we can't call OpenAI directly without paying or mocking)
# For this verification, we are testing the PROMPT CONSTRUCTION, not the actual LLM call.
# However, the function `generate_chat_response` makes a real network call.
# To test logic without network, we would need to mock `requests.post`.
# BUT, the user wants me to verify it works. 
# Best way: Check if the payload construction logic is correct by inspecting the code or running with a mock.

import unittest
from unittest.mock import patch, MagicMock

class TestChatHistory(unittest.TestCase):
    
    @patch('src.services.llm.requests.post')
    def test_history_passed_to_api(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Tested"}}]
        }
        mock_post.return_value = mock_response

        # Input data
        messages = [
            {"role": "user", "content": "My name is John"},
            {"role": "assistant", "content": "Hello John"},
            {"role": "user", "content": "What is my name?"}
        ]
        sources = []

        # Execute
        generate_chat_response(messages, sources)

        # Verify
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        
        # Check if messages list in payload contains the history
        sent_messages = payload['messages']
        
        # Expected: System prompt + 3 history messages
        self.assertEqual(len(sent_messages), 4)
        self.assertEqual(sent_messages[1]['role'], 'user')
        self.assertEqual(sent_messages[1]['content'], 'My name is John')
        self.assertEqual(sent_messages[3]['content'], 'What is my name?')
        
        print("PASS: History was correctly constructed in the payload.")

if __name__ == "__main__":
    unittest.main()
