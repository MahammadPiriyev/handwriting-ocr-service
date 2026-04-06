import requests
import os
import time
import base64

def create_test_image(filename="test_image.png"):
    # Write a tiny 1x1 PNG (base64) so this script doesn't require opencv
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    data = base64.b64decode(png_b64)
    with open(filename, "wb") as f:
        f.write(data)
    return filename

def test_ocr2_endpoint():
    url = "http://localhost:8000/ocr/v2"
    file_path = create_test_image()
    
    print(f"Testing {url} with {file_path}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'image/png')}
            # Retry connection a few times in case server is starting
            for i in range(5):
                try:
                    response = requests.post(url, files=files, timeout=120)
                    break
                except requests.exceptions.ConnectionError:
                    print("Connection refused, retrying in 2s...")
                    time.sleep(2)
            else:
                print("Failed to connect to server after retries.")
                return

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(response.json())
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    test_ocr2_endpoint()
