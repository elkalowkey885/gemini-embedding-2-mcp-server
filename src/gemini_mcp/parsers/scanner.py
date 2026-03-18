import os
import pathlib
import logging
from typing import List, Generator, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks for better semantic retrieval."""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        
    return chunks

def extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extracts raw text from a given file based on its extension."""
    ext = file_path.suffix.lower()
    
    try:
        if ext in ['.txt', '.md', '.csv']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        elif ext == '.pdf' and fitz is not None:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
            
        elif ext == '.docx' and docx is not None:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
            
        else:
            logger.debug(f"Unsupported or missing library for file type: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def scan_directory(directory_path: str, chunk_size: int = 1000, overlap: int = 200) -> Generator[Dict[str, Any], None, None]:
    """
    Recursively scans a directory, extracts text from supported files, and yields chunks.
    Yields dicts with: {'text': str, 'metadata': {'source': str, 'chunk_index': int}}
    """
    base_dir = pathlib.Path(directory_path)
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return
        
    for root, _, files in os.walk(base_dir):
        # Skip hidden directories like .git
        if '/.' in root.replace('\\', '/'):
            continue
            
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = pathlib.Path(root) / file
            
            # Simple size check to avoid absolutely massive binaries disguised as text
            try:
                if file_path.stat().st_size > 50 * 1024 * 1024:  # 50 MB limit
                    continue
            except Exception:
                continue

            text = extract_text_from_file(file_path)
            if not text.strip():
                continue
                
            chunks = chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    yield {
                        "text": chunk,
                        "metadata": {
                            "source": str(file_path.absolute()),
                            "chunk_index": i
                        }
                    }
