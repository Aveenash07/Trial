# Trial

# RAG QA Bot with LLM Analysis

A FastAPI-based Retrieval-Augmented Generation (RAG) Question-Answering bot with integrated LLM text analysis capabilities.

## Features

- **Document Upload**: Upload and store text documents for knowledge base creation
- **Question Answering**: Query uploaded documents using natural language
- **LLM Text Analysis**: Analyze text for summaries, key entities, and sentiment
- **RESTful API**: Clean, well-documented API endpoints
- **Error Handling**: Comprehensive error handling and logging
- **Health Monitoring**: Built-in health check endpoint

## Requirements

- Python 3.10+
- FastAPI
- Pydantic
- Additional dependencies (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd folder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Ensure you have the required modules:
   - `llm_text_analysis.py` - Contains the `analyze_text_with_llm` function
   - `pinecone_processes.py` - Contains the RAGBot implementation (or use the placeholder)

2. Configure your LLM API keys and settings in the respective modules.

## Usage

### Starting the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

## API Endpoints

### 1. Upload Document
**POST** `/upload/`

Upload a text document to the knowledge base.

**Parameters:**
- `file`: Upload file (must be `.txt` format, UTF-8 encoded)

**Example:**
```bash
curl -X POST "http://localhost:8000/upload/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.txt"
```

**Response:**
```json
{
  "message": "Successfully stored document with 245 words."
}
```

### 2. Ask Question
**POST** `/ask/`

Query the knowledge base with a natural language question.

**Request Body:**
```json
{
  "question": "What is the main topic of the uploaded document?"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ask/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?"}'
```

**Response:**
```json
{
  "answer": "Based on the uploaded document, the main topic is..."
}
```

### 3. Analyze Text
**POST** `/analyze-text/`

Perform LLM-powered analysis on provided text.

**Request Body:**
```json
{
  "text": "Your text to analyze here. Must be at least 50 words and between 100-10000 characters."
}
```

**Validation Rules:**
- Minimum 100 characters
- Maximum 10,000 characters  
- At least 50 words
- UTF-8 encoded text

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze-text/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text content here..."}'
```

**Response:**
```json
{
  "summary": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
  "entities": [
    {"name": "Entity 1", "role": "Description of role"},
    {"name": "Entity 2", "role": "Description of role"},
    {"name": "Entity 3", "role": "Description of role"}
  ],
  "sentiment": "positive"
}
```

### 4. Health Check
**GET** `/health`

Check the service status.

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG QA Bot with LLM Analysis"
}
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format, empty files, validation errors
- **500 Internal Server Error**: Processing failures, LLM API errors

All errors include descriptive messages and are logged for debugging.

## Logging

The application uses Python's built-in logging module with INFO level logging. Logs include:
- Document upload status
- Question processing
- Text analysis operations
- Error details with stack traces

## File Processing

### Supported Formats
- **Text files only** (`.txt` extension)
- **UTF-8 encoding** required
- **Non-empty files** only

### Text Sanitization
Uploaded files are automatically sanitized to remove problematic control characters while preserving:
- Newlines (`\n`)
- Tabs (`\t`) 
- Carriage returns (`\r`)

## Development

### Project Structure
```
├── main.py                    # FastAPI application
├── llm_text_analysis.py      # LLM analysis functions
├── pinecone_processes.py     # RAG bot implementation
└── README.md                 # This file
```

### Running in Development Mode
```bash
uvicorn main:app --reload
```

