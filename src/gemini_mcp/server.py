import asyncio
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from gemini_mcp.db.store import ChromaStore
from gemini_mcp.embeddings.gemini import GeminiEmbeddingClient
from gemini_mcp.parsers.scanner import scan_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp = FastMCP("Gemini Document Search")

# Initialize global clients (lazy-loaded so they only instantiate on actual usage)
_db_store: Optional[ChromaStore] = None
_embedding_client: Optional[GeminiEmbeddingClient] = None

def get_db() -> ChromaStore:
    global _db_store
    if _db_store is None:
        _db_store = ChromaStore()
    return _db_store

def get_embedder() -> GeminiEmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = GeminiEmbeddingClient()
    return _embedding_client


@mcp.tool()
async def index_directory(directory_path: str) -> str:
    """
    Scans a local directory, extracts text from files (PDF, DOCX, TXT, MD), 
    generates semantic embeddings using Gemini 2 and stores them for searching.
    """
    try:
        db = get_db()
        embedder = get_embedder()
        
        chunks = []
        texts_to_embed = []
        
        logger.info(f"Starting scan of {directory_path}")
        
        # Batch processing to avoid massive API payloads
        BATCH_SIZE = 50
        total_indexed = 0
        
        for item in scan_directory(directory_path):
            chunks.append(item)
            texts_to_embed.append(item["text"])
            
            if len(chunks) >= BATCH_SIZE:
                embeddings = embedder.embed_texts(texts_to_embed)
                db.add_chunks(chunks, embeddings)
                total_indexed += len(chunks)
                chunks.clear()
                texts_to_embed.clear()
                
        # Process remaining chunks
        if chunks:
            embeddings = embedder.embed_texts(texts_to_embed)
            db.add_chunks(chunks, embeddings)
            total_indexed += len(chunks)
            
        return f"Successfully indexed {total_indexed} segments from {directory_path}."
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index {directory_path}: {str(e)}"

@mcp.tool()
async def search_my_documents(query: str, limit: int = 5) -> str:
    """
    Performs a semantic search over your previously indexed local documents 
    using the Gemini 2 Embedding model.
    """
    try:
        db = get_db()
        embedder = get_embedder()
        
        query_vec = embedder.embed_query(query)
        if not query_vec:
            return "Failed to generate embedding for query."
            
        matches = db.query(query_vec, n_results=limit)
        
        if not matches:
            return "No relevant documents found. Have you indexed your directories yet?"
            
        # Format the results cleanly for the LLM
        formatted = f"Found {len(matches)} relevant excerpts:\n\n"
        for i, match in enumerate(matches, 1):
            source = match["metadata"].get("source", "Unknown file")
            score = match.get("distance", 0.0)
            text = match["text"]
            
            formatted += f"--- Result {i} (from: {source}) (distance: {score:.3f}) ---\n"
            formatted += f"{text}\n\n"
            
        return formatted
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def list_indexed_directories() -> str:
    """
    Lists all the file paths that have been indexed in the database.
    """
    try:
        db = get_db()
        sources = db.list_indexed_sources()
        if not sources:
            return "The database is currently empty."
            
        return "The following files are indexed:\n- " + "\n- ".join(sources)
    except Exception as e:
        return f"Error connecting to database: {str(e)}"

def main():
    """Entry point for the MCP server when run as a script."""
    mcp.run()

if __name__ == "__main__":
    main()
