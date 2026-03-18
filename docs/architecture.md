# Architecture

The Gemini MCP Embedding Server is built with simplicity, local privacy, and performance in mind.

## Core Flow
1. **Document Ingestion (`parsers/scanner.py`)**:
   Recursively reads through a directory, skipping huge binary files and hidden folders. It extracts raw text using `PyMuPDF` for PDFs and `python-docx` for Word documents.
   
2. **Chunking & Embedding (`embeddings/gemini.py`)**:
   Documents are chunked into 1000-character segments with 200 characters of overlap. These chunks are sent in batches to the `gemini-embedding-2-preview` model API via the `google-genai` SDK.

3. **Vector Storage (`db/store.py`)**:
   The vectors (float arrays) and the original text are stored in a local `chromadb` instance. The database lives securely in your `~/.gemini_mcp_db` folder, ensuring your file contents never leave your machine except for the minimal API embedding call.

4. **MCP Server (`server.py`)**:
   Uses the `fastmcp` framework to seamlessly expose `index_directory`, `search_my_documents`, and `list_indexed_directories` directly to Claude Desktop or other MCP clients.
