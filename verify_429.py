
import unittest
from unittest.mock import patch, MagicMock
import time
from src.services.llm import generate_chat_response

class TestRateLimit(unittest.TestCase):
    
    @patch('src.services.llm.time.sleep')
    @patch('src.services.llm.requests.post')
    def test_retry_logic(self, mock_post, mock_sleep):
        # Setup mock behavior: 2 failures (429), then success (200)
        fail_response = MagicMock()
        fail_response.status_code = 429
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}]
        }
        
        # Side effect: return fail twice, then success
        mock_post.side_effect = [fail_response, fail_response, success_response]

        # Execute
        result = generate_chat_response([{"role": "user", "content": "hi"}], [])

        # Verify
        self.assertEqual(result, "Success after retry")
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        
        # Check backoff times (1s, then 2s)
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)
        
        print("PASS: Retry logic works with exponential backoff")

    @patch('src.services.llm.requests.post')
    def test_history_truncation(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        # Create 20 dummy messages
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        
        generate_chat_response(messages, [])
        
        # Check payload
        args, kwargs = mock_post.call_args
        sent_messages = kwargs['json']['messages']
        
        # Expect 1 system + 10 history = 11
        self.assertEqual(len(sent_messages), 11)
        self.assertEqual(sent_messages[1]['content'], 'msg 10') # First of the last 10
        self.assertEqual(sent_messages[-1]['content'], 'msg 19') # Last one
        
        print("PASS: History truncated to last 10 messages")

if __name__ == "__main__":
    unittest.main()
