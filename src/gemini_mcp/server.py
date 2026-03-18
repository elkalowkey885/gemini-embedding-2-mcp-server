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
mcp = FastMCP("Gemini Embedding 2 MCP")

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


from google.genai import types

@mcp.tool()
async def index_directory(directory_path: str) -> str:
    """
    Scans a local directory, extracts text from files (PDF, DOCX, TXT, MD) AND 
    raw image bytes (.jpg, .png, .webp), generates semantic embeddings using 
    Gemini 2 and stores them for searching.
    """
    try:
        db = get_db()
        embedder = get_embedder()
        
        chunks = []
        items_to_embed = []
        
        logger.info(f"Starting scan of {directory_path}")
        
        # Batch processing to avoid massive API payloads
        BATCH_SIZE = 50
        total_indexed = 0
        
        for item in scan_directory(directory_path):
            chunks.append(item)
            
            if item.get("is_image", False):
                source = item["metadata"]["source"]
                ext = source.split(".")[-1].lower()
                mime = f"image/{ext}" if ext in ["png", "webp"] else "image/jpeg"
                items_to_embed.append(types.Part.from_bytes(data=item["raw_data"], mime_type=mime))
            else:
                items_to_embed.append(item["raw_data"])
            
            if len(chunks) >= BATCH_SIZE:
                embeddings = embedder.embed_items(items_to_embed)
                db.add_chunks(chunks, embeddings)
                total_indexed += len(chunks)
                chunks.clear()
                items_to_embed.clear()
                
        # Process remaining chunks
        if chunks:
            embeddings = embedder.embed_items(items_to_embed)
            db.add_chunks(chunks, embeddings)
            total_indexed += len(chunks)
            
        return f"Successfully indexed {total_indexed} segments (text & images) from {directory_path}."
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index {directory_path}: {str(e)}"

@mcp.tool()
async def search_my_documents(query: str, limit: int = 5) -> str:
    """
    Performs a semantic search over your previously indexed local documents 
    AND images using the Gemini 2 Embedding model.
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
        formatted = f"Found {len(matches)} relevant excerpts/images:\n\n"
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

@mcp.tool()
async def remove_directory_from_index(directory_path: str) -> str:
    """
    Removes all documents and images belonging to a specific directory path from the index.
    """
    try:
        db = get_db()
        deleted_count = db.delete_directory(directory_path)
        return f"Successfully removed {deleted_count} chunks/images originating from {directory_path}."
    except Exception as e:
        return f"Error removing directory: {str(e)}"

@mcp.tool()
async def sync_indexed_directories() -> str:
    """
    Auto-updates existing folders. It finds all unique parent directories of currently 
    indexed files and re-indexes them to capture new or modified files.
    """
    import os
    try:
        db = get_db()
        sources = db.list_indexed_sources()
        if not sources:
            return "Database is empty. Nothing to sync."
            
        # Find unique parent directories
        directories = set()
        for source in sources:
            parent = os.path.dirname(source)
            directories.add(parent)
            
        # To avoid redundant scans, only keep top-level directories
        top_level_dirs = set()
        for d in sorted(list(directories)):
            if not any(d.startswith(top + os.sep) for top in top_level_dirs):
                top_level_dirs.add(d)
                
        results = []
        for d in top_level_dirs:
            res = await index_directory(d)
            results.append(res)
            
        return "Sync Summary:\n" + "\n".join(results)
    except Exception as e:
        return f"Error during sync: {str(e)}"

def main():
    """Entry point for the MCP server when run as a script."""
    mcp.run()

if __name__ == "__main__":
    main()
