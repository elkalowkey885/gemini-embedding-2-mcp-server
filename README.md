# Gemini Embedding 2 MCP Server

A powerful Model Context Protocol (MCP) server that connects your local documents to Claude (or any MCP-compatible AI) using Google's state-of-the-art **Gemini Embedding 2 Preview** model.

## Features
- **Local Vector Database**: Uses ChromaDB entirely locally to index your document embeddings.
- **Multimodal Embedding Power**: Powered by `gemini-embedding-2-preview`, Google's latest embedding model.
- **Native Image Search**: Scans and mathematically embeds `.jpg`, `.png`, and `.webp` graphics. Search context visually!
- **Dynamic Indexing**: Read, parse, and embed PDF, TXT, MD, DOCX files directly from Claude's interface.
- **Semantic Search**: Native MCP tool for lighting-fast semantic retrieval over your documents.

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for lightning-fast Python dependency management.

```bash
# Clone the repository
git clone https://github.com/AlaeddineMessadi/gemini-mcp-embedding-server.git
cd gemini-mcp-embedding-server

# Install dependencies
uv sync
```

## Setup & Configuration

1. **Get a Gemini API Key**: Grab one from Google AI Studio.
2. **Configure Claude Desktop**:
   Open your Claude Desktop config file (usually `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS) and add:

```json
{
  "mcpServers": {
    "gemini-embedding-2-mcp": {
      "command": "/path/to/gemini-mcp-embedding-server/.venv/bin/gemini-embedding-2-mcp",
      "args": [],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Available MCP Tools

Once connected, Claude will have access to:
- `index_directory(path: str)`: Scan and structurally embed a local folder (reads text AND images).
- `search_my_documents(query: str, limit: int)`: Perform semantic search over your indexed documents and pictures.
- `list_indexed_directories()`: See what files have been embedded.
- `sync_indexed_directories()`: Automatically re-indexes known folders to capture newly added files.
- `remove_directory_from_index(path: str)`: Remove a folder's vectors from your ChromaDB instance.

## License
MIT
