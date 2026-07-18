import yaml
from typing import Dict, Any

class LLMRouter:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.models = self.config.get("models", {})

    def get_model_config(self, task_key: str) -> Dict[str, Any]:
        """
        Retrieves the provider and model for a given task key from configuration.
        
        Args:
            task_key: The key identifying the task (e.g., 'bulk_grammar_pass', 'title_and_tagging')
            
        Returns:
            A dictionary containing the provider and model name.
        """
        if task_key not in self.models:
            # Fallback or error handling if task key is missing
            raise ValueError(f"Task key '{task_key}' not found in configuration.")
            
        return self.models[task_key]

# Singleton instance for easy access across the worker
router = LLMRouter()

def get_model_config(task_key: str) -> Dict[str, Any]:
    return router.get_model_config(task_key)