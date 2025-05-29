from openai import OpenAI
import json
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTextAnalyzer:
    def __init__(self):
        """
        Initialize the LLM Text Analyzer.
        Set your OpenAI API key in environment variable OPENAI_API_KEY
        """
        self.api_key = os.getenv("OPEN_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
    
    def _create_analysis_prompt(self, text: str) -> str:
        """Create a comprehensive prompt for text analysis"""
        prompt = f"""
Please analyze the following text and provide a response in JSON format with exactly these keys: "summary", "entities", and "sentiment".

Instructions:
1. Summary: Provide 3-5 bullet points summarizing the key points of the text
2. Entities: Identify exactly 3 key entities (people, organizations, concepts, etc.) and describe their roles
3. Sentiment: Analyze the overall sentiment as "positive", "negative", or "neutral"

Text to analyze:
{text}

Please respond with a valid JSON object in this exact format:
{{
    "summary": [
        "First key point",
        "Second key point",
        "Third key point"
    ],
    "entities": [
        {{"name": "Entity Name 1", "role": "Description of role"}},
        {{"name": "Entity Name 2", "role": "Description of role"}},
        {{"name": "Entity Name 3", "role": "Description of role"}}
    ],
    "sentiment": "positive/negative/neutral"
}}
"""
        return prompt
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        raw_response_content = None
        try:
            prompt = self._create_analysis_prompt(text)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text and responds with properly formatted JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            if not response.choices or not response.choices[0].message:
                logger.error("OpenAI response is missing expected structure (choices or message).")
                raise ValueError("Invalid response structure from OpenAI.")

            raw_response_content = response.choices[0].message.content
            if raw_response_content is None:
                logger.error("OpenAI response content is None.")
                raise ValueError("OpenAI response content is None.")

            processed_content = raw_response_content.strip()
            if processed_content.startswith("```json"):
                processed_content = processed_content[len("```json"):].strip()
            if processed_content.startswith("```"):
                 processed_content = processed_content[len("```"):].strip()
            if processed_content.endswith("```"):
                processed_content = processed_content[:-len("```")].strip()
            
            result = json.loads(processed_content)

            required_keys = ["summary", "entities", "sentiment"]
            if not all(key in result for key in required_keys):
                missing_keys = [key for key in required_keys if key not in result]
                logger.warning(f"OpenAI response JSON missing required keys: {missing_keys}. Got keys: {list(result.keys())}")
                raise ValueError(f"Response JSON from OpenAI missing required keys: {', '.join(missing_keys)}.")
            
            if not isinstance(result.get("summary"), list):
                raise ValueError("OpenAI response 'summary' is not a list.")
            if not isinstance(result.get("entities"), list):
                 raise ValueError("OpenAI response 'entities' is not a list.")
            if result.get("sentiment") not in ["positive", "negative", "neutral"]:
                 logger.warning(f"OpenAI response 'sentiment' had unexpected value: {result.get('sentiment')}")
                 # Allow unexpected sentiment values but log, or raise ValueError if strictness is required.

            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from OpenAI: {e}")
            logger.error(f"OpenAI response content that failed parsing (first 500 chars): {str(raw_response_content)[:500]}")
            raise ValueError(f"Failed to parse LLM response as JSON. Content started with: {str(raw_response_content)[:200]}...")
        except Exception as e:
            logger.error(f"Error during OpenAI analysis: {e}", exc_info=True)
            # Re-raise as a more generic runtime error if it's not a ValueError already
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"An unexpected error occurred during OpenAI analysis: {str(e)}")

def analyze_text_with_llm(text: str) -> Dict[str, Any]:
    analyzer = LLMTextAnalyzer()
    return analyzer.analyze_text(text)