# llm_integration.py
# Description: Module for integrating Large Language Models (LLMs) into the Lawmakers Suite 2048-AES. Supports Hugging Face models for local inference and OpenAI API for cloud-based queries. Designed for legal text analysis, such as contract review or case law summarization. Uses environment variables for API keys to ensure secure configuration.

from transformers import pipeline
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

def query_huggingface(prompt: str) -> str:
    """
    Query a Hugging Face model for text classification or generation.
    Args:
        prompt (str): Input text for the LLM (e.g., "Is this contract legally binding?").
    Returns:
        str: Model response (e.g., classification label or generated text).
    """
    try:
        classifier = pipeline("text-classification", model="distilbert-base-uncased")
        result = classifier(prompt)
        return result[0]["label"]
    except Exception as e:
        return f"Error querying Hugging Face model: {str(e)}"

def query_openai(prompt: str) -> str:
    """
    Query OpenAI API for advanced text generation.
    Args:
        prompt (str): Input text for the LLM.
    Returns:
        str: Generated text from OpenAI.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key not configured"
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "text-davinci-003", "prompt": prompt, "max_tokens": 100}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except requests.RequestException as e:
        return f"Error querying OpenAI: {str(e)}"

if __name__ == "__main__":
    # Example usage
    prompt = "Summarize the key points of a contract dispute case."
    hf_result = query_huggingface(prompt)
    openai_result = query_openai(prompt)
    print(f"Hugging Face Response: {hf_result}")
    print(f"OpenAI Response: {openai_result}")