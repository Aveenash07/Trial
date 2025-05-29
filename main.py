from fastapi import FastAPI, UploadFile, File, HTTPException, Request

# Assuming pinecone_processes.py contains RAGBot as described
# from pinecone_processes import RAGBot
from llm_text_analysis import (
    analyze_text_with_llm,
)  # Assuming this is in llm_text_analysis.py
from pydantic import BaseModel, Field, validator
from typing import (
    List,
    Literal,
)  # Literal is not used in the snippet, but kept if intended for future
import logging
import re
import unicodedata  # unicodedata is not directly used in this file snippet, but often useful


# Placeholder for RAGBot if pinecone_processes.py is not fully defined for this example
class RAGBot:
    def store_document(self, text: str) -> str:
        logger.info(f"Storing document (length: {len(text)})")
        # Dummy implementation
        return f"Successfully stored document with {len(text.split())} words."

    def answer_question(self, question: str) -> str:
        logger.info(f"Answering question: {question}")
        # Dummy implementation
        return f"This is a placeholder answer to: {question}"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG QA Bot")
rag_bot = RAGBot()  # Initialize your RAGBot


class Query(BaseModel):
    question: str


class LLMAnalysisRequest(BaseModel):
    text: str = Field(
        ..., min_length=100, max_length=10000, description="Text to analyze"
    )

    @validator("text")
    def validate_text_length(cls, v):
        word_count = len(v.split())
        if word_count < 50:
            raise ValueError(
                "Text should contain at least 50 words for meaningful analysis"
            )
        return v


@app.post("/upload/", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename or not isinstance(file.filename, str):
            logger.error("No filename provided or filename is invalid")
            raise HTTPException(status_code=400, detail="A valid filename is required")

        if not file.filename.lower().endswith(".txt"):
            logger.error(
                f"Invalid file type: {file.filename}. Only .txt files are allowed."
            )
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")

        text_bytes = await file.read()
        if not text_bytes:
            logger.error("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="File is empty")

        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode file as UTF-8: {e}")
            raise HTTPException(
                status_code=400, detail="File must be a valid UTF-8 encoded text file"
            )

        # Sanitize text to remove problematic control characters before storing or processing
        # This removes C0 and C1 control characters except for tab, newline, carriage return
        cleaned_text = "".join(
            ch
            for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
        )

        result = rag_bot.store_document(cleaned_text)
        return {"message": result}
    except HTTPException:  # Re-raise HTTPExceptions to avoid masking them as 500
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during file upload: {str(e)}",
        )


@app.post("/ask/", response_model=dict)
async def ask_question(query: Query):
    try:
        answer = rag_bot.answer_question(query.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-text/", response_model=dict)
async def analyze_text_with_llm_endpoint(
    request_data: LLMAnalysisRequest,
):  # Changed variable name for clarity
    """
    Analyze text using OpenAI LLM to provide:
    - Summary in bullet points
    - Three key entities and their roles
    - Sentiment analysis
    Returns JSON with keys: "summary", "entities", "sentiment"
    """
    try:
        logger.info(
            f"Analyzing text with OpenAI GPT-3.5-turbo for /analyze-text/ endpoint"
        )
        logger.info(f"Text length: {len(request_data.text.split())} words")

        # Perform the analysis
        result = analyze_text_with_llm(request_data.text)
        logger.info("Text analysis completed successfully")
        return result
    except (
        ValueError
    ) as e:  # Specific errors that might indicate bad input or parsing issues
        logger.error(
            f"Validation or processing error in text analysis: {e}", exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in text analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed due to an unexpected error: {str(e)}",
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG QA Bot with LLM Analysis"}
