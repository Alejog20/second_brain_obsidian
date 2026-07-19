from typing import Dict, List, Any
import datetime
from .llm_router import get_model_config

class Digestor:
    """
    Handles the digestion of daily notes into atomic notes.
    Requirement 2: daily note -> atomic notes.
    """
    def __init__(self):
        pass

    def process_daily_note(self, daily_note_path: str) -> List[Dict[str, Any]]:
        """
        Reads a daily note and segments it into atomic units.
        
        Args:
            daily_note_path: Path to the daily note (e.g., '01-Daily/2026-07-16.md')
            
        Returns:
            A list of dictionaries, each representing a processed chunk.
        """
        # 1. Load raw text from path (Implementation detail for reading file)
        raw_content = self._read_file(daily_note_path)
        
        # 2. Segment content based on headers or LLM-assisted logic
        chunks = self._segment_content(raw_content)
        
        results = []
        for chunk in chunks:
            # 3. Embed and check against vector store (Logic placeholder)
            # If high similarity found -> merge/append
            # If no match -> create new note
            processed_chunk = self._process_chunk(chunk, daily_note_path)
            results.append(processed_chunk)
            
        return results

    def _read_file(self, path: str) -> str:
        """Helper to read the file content."""
        # This will eventually integrate with a proper file utility
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _segment_content(self, text: str) -> List[str]:
        """Splits the daily note into chunks based on headers or other separators."""
        # Implementation logic for splitting content
        # Often done by looking at lines starting with '#'
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        for line in lines:
            if line.startswith('#') and len(line) > 1:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _process_chunk(self, chunk: str, source_date: str) -> Dict[str, Any]:
        """Determines if a chunk should become a new note or be merged."""
        # Logic to check similarity in vector store would go here.
        # If no match: create file with frontmatter 
        # 'source:: [[MM-DD-YYYY]]' and 'created:: MM-DD-YYYY'
        return {"content": chunk, "source": source_date}

digestor = Digestor()

def digest_daily_note(path: str) -> List[Dict[str, Any]]:
    return digestor.process_daily_note(path)