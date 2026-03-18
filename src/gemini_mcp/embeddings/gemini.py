import os
import logging
from typing import List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# The brand new embedding model from Google
EMBEDDING_MODEL = "gemini-embedding-2-preview"

class GeminiEmbeddingClient:
    def __init__(self, api_key: str = None):
        """
        Initializes the Gemini client. If api_key is not provided, 
        it attempts to read from the GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        # Initialize the new google-genai SDK client
        self.client = genai.Client(api_key=self.api_key)
        
    def embed_items(self, items: List[Any]) -> List[List[float]]:
        """
        Sends a batch of items (strings, or Image bytes represented as types.Part) 
        to Gemini Tracking 2 to get vector embeddings. 
        Returns a list of float arrays (the vectors).
        """
        if not items:
            return []
            
        try:
            # We use embed_content with the required model.
            embeddings = []
            
            # Since the API sometimes struggles with deeply nested mixed batches or large parallel payloads, 
            # we iterate through the batch. The `embed_content` takes `contents`.
            # We can pass them as a list to the SDK
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=items
            )
            
            for emb in response.embeddings:
                embeddings.append(emb.values)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error calling Gemini Embedding 2: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single semantic search query string.
        """
        res = self.embed_texts([query])
        if res:
            return res[0]
        return []
