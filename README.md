# handwriting-ocr-service

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/MahammadPiriyev/T800-socar-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance FastAPI service for document OCR, LLM-based text correction, and Retrieval-Augmented Generation (RAG) over a PostgreSQL knowledge base.

## Table of Contents

- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the API Server](#running-the-api-server)
  - [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## System Architecture

The service is composed of modular components orchestrated by a FastAPI backend.

- **API Layer (FastAPI)**: Manages asynchronous HTTP requests for document processing and chat.

  - `POST /ocr/v1`: Google Vision -> Azure OpenAI cleanup pipeline.
  - `POST /ocr/v2`: OpenRouter (Mistral) -> Azure (Llama) cleanup pipeline.
  - `POST /llm`: RAG-based chat endpoint.
  - `GET /health`: Health check.

- **Processing Pipelines**:

  - **Ingestion**: Handles `application/pdf` and `image/*` content types. PDFs are rendered into PNG images per page via PyMuPDF.
  - **OCR & Correction**:
    - Raw text is extracted by an external OCR provider (Google Vision or OpenRouter).
    - The extracted text is corrected by an LLM (Azure OpenAI) using a detailed system prompt (`OCR_SYSTEM_PROMPT`) specialized for Azerbaijani/Russian technical terms.
  - **Storage**: Corrected text and document metadata are persisted in a PostgreSQL `history` table.

- **Retrieval-Augmented Generation (RAG)**:
  - The `/llm` endpoint receives a user query.
  - It fetches all documents from the PostgreSQL database.
  - `rank-bm25` is used to find the most relevant text chunks from the entire document corpus.
  - These chunks are injected as context into a final LLM prompt to generate a grounded answer.

## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.8+
- PostgreSQL server
- API keys for:
  - Google Cloud Vision
  - Azure OpenAI
  - OpenRouter.ai

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/MahammadPiriyev/T800-socar-ai.git
    cd T800-socar-ai
    ```

2.  **Create and activate a virtual environment:**

    ```sh
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

The application is configured using environment variables.

1.  Copy the sample environment file:

    ```sh
    cp .env.sample .env
    ```

2.  Edit the `.env` file with your credentials.

| Variable                  | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| **PostgreSQL**            |                                                         |
| `DB_HOST`                 | Database server host.                                   |
| `DB_PORT`                 | Database server port.                                   |
| `DB_NAME`                 | Database name.                                          |
| `DB_USER`                 | Database username.                                      |
| `DB_PASS`                 | Database password.                                      |
| **OCR Pipeline v1**       |                                                         |
| `GOOGLE_VISION_API_KEY`   | API key for Google Cloud Vision.                        |
| `AZURE_OPENAI_API_KEY`    | Key for Azure OpenAI (used for text cleanup).           |
| `AZURE_OPENAI_ENDPOINT`   | Endpoint for your Azure OpenAI resource.                |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name for the cleanup model (e.g., GPT-4).    |
| **OCR Pipeline v2**       |                                                         |
| `OPENROUTER_API_KEY`      | API key for OpenRouter.ai.                              |
| `OPENROUTER_MODEL`        | Model to use for OCR (e.g., `mistralai/ministral-..`).  |
| `AZURE_OCR2_API_KEY`      | Key for the secondary Azure service (if different).     |
| `AZURE_OCR2_ENDPOINT`     | Endpoint for the secondary Azure service.               |
| `AZURE_OCR2_DEPLOYMENT`   | Deployment name for the v2 cleanup model (e.g., Llama). |
| **RAG Chat Endpoint**     |                                                         |
| `CHAT_API_KEY`            | API key for the final chat generation model.            |
| `CHAT_BASE_URL`           | Base URL for the chat model's API.                      |
| `CHAT_MODEL`              | Model name for chat generation (e.g., `gpt-oss-120b`).  |

> **Note**: An `AuthenticationError` in the logs or API response indicates an invalid API key or endpoint. Verify these values in your cloud provider's dashboard.

## Usage

### Running the API Server

Start the server using Uvicorn:

```sh
uvicorn src.api.endpoints:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### API Reference

#### `POST /ocr/v1` and `POST /ocr/v2`

Processes a document and stores the corrected text.

- **Request**: `multipart/form-data`

  - `file`: The PDF or image file to process.

  ```sh
  curl -X POST -F "file=@/path/to/document.pdf" http://127.0.0.1:8000/ocr/v1
  ```

- **Success Response (200 OK)**:

  ```json
  [
    {
      "page_number": 1,
      "MD_text": "This is the cleaned and corrected text from the first page..."
    }
  ]
  ```

#### `POST /llm`

Initiates a RAG chat session based on all processed documents.

- **Request**: `application/json`

  ```sh
  curl -X POST -H "Content-Type: application/json" \
    -d '[{"role": "user", "content": "Summarize the findings on the Məhsuldar Qat formation."}]' \
    http://127.0.0.1:8000/llm
  ```

- **Success Response (200 OK)**:

  ```json
  {
    "sources": [
      {
        "pdf_name": "document.pdf",
        "page_number": 1,
        "text_chunk": "Relevant text chunk from the document..."
      }
    ],
    "answer": "Based on the provided documents, the Məhsuldar Qat formation shows..."
  }
  ```

## Project Structure

```
T800-socar-ai/
├── data/
│   └── ocr_cache/        # Caches raw responses from external OCR services
├── src/
│   ├── api/              # FastAPI application, endpoints, and Pydantic models
│   ├── core/             # Configuration loading and system prompts
│   └── services/         # Business logic for OCR, storage, and LLM/RAG
├── .env.sample           # Environment variable template
├── requirements.txt      # Project dependencies
├── verify_ocr2.py        # Standalone test script for the v2 OCR endpoint
└── README.md
```

## Project Structure

```
T800-socar-ai/
├── data/
│   └── ocr_cache/        # Caches responses from OCR services
├── src/
│   ├── api/              # FastAPI application, endpoints, and models
│   ├── core/             # Configuration and core settings
│   └── services/         # Business logic for OCR, storage, and LLM
├── .env.sample           # Sample environment variables
├── .gitignore
├── requirements.txt      # Project dependencies
├── verify_ocr2.py        # Test script for the v2 OCR endpoint
└── README.md
```

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and open a pull request with your changes.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`).
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`).
4.  Push to the Branch (`git push origin feature/NewFeature`).
5.  Open a Pull Request.

## License

Distributed under the MIT License. See the `LICENSE` file for more information.
