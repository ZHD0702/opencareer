import yaml
from string import Template
from pathlib import Path

class PromptLoader:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self._load_prompts()
    
    def _load_prompts(self):
        with open(self.base_path / "system.yaml", "r", encoding="utf-8") as f:
            self.system_config = yaml.safe_load(f)
        
        with open(self.base_path / "personas/default.yaml", "r", encoding="utf-8") as f:
            self.persona = yaml.safe_load(f)
    
    def get_system_prompt(self) -> str:
        template = Template(self.system_config["template"])
        return template.substitute({
            "persona_name": self.persona["name"],
            "tone": self.persona["tone"],
            "style": self.persona["style"],
        })

# 全局单例
prompt_loader = PromptLoader()