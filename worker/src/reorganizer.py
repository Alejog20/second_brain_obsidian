from typing import Dict, List, Optional, Any
from .llm_router import get_model_config

class Reorganizer:
    """
    Handles the reorganization of notes including title checks, 
    grammar/clarity improvements, taxonomy placement, and link suggestions.
    """
    def __init__(self):
        pass

    def process_note(self, note_content: str, metadata: Dict) -> Dict[str, Any]:
        """
        Runs the reorganization pipeline for a single note.
        
        Args:
            note_content: The raw text of the note.
            metadata: Metadata about the note (e.g., current title, path).
            
        Returns:
            A dictionary containing the proposed changes and any flags for manual review.
        """
        results = {
            "title": metadata.get("title", "Untitled"),
            "content": note_content,
            "suggestions": [],
            "flags": []
        }

        # 1. Title Check & Improvement
        # Uses 'title_and_tagging' model configuration via the router.
        title_cfg = get_model_config("title_and_tagging")
        results["title"] = self._improve_title(note_content, results["title"], title_cfg)

        # 2. Grammar & Clarity Pass
        # Uses 'bulk_grammar_pass' model configuration.
        grammar_cfg = get_model_config("bulk_grammar_pass")
        result_text, is_clear = self._improve_grammar(note_content, grammar_cfg)
        results["content"] = result_text
        if not is_clear:
            results["flags"].append("low_clarity_warning")

        # 3. Taxonomy & Link Suggestions (logic placeholders for integration with vector store/embeddings)
        # These are currently flagged as 'suggestion' items.
        self._suggest_links(note_content, results)

        return results

    def _improve_title(self, content: str, current_title: str, config: Dict) -> str:
        """Check if title is missing or misleading and suggest a better one."""
        # Implementation would call the LLM using configuration from `config.yaml`
        return current_title

    def _improve_grammar(self, content: str, config: Dict) -> (str, bool):
        """Fix grammar/clarity errors without changing user's voice."""
        # Returns (updated_content, is_clearly_understood)
        return content, True

    def _suggest_links(self, content: str, results: Dict):
        """Suggest [[wikilinks]] based on internal similarity."""
        pass

reorganizer = Reorganizer()

def reorganize_note(content: str, metadata: Dict) -> Dict[str, Any]:
    return reorganizer.process_note(content, metadata)