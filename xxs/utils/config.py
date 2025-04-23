import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """ class for loading / managing yaml config files (basically prettify the code) """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """ load the config from the yaml file """

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """ get a config value by key """

        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """ enable dict-like access to config values """
        
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """ check if a config key exists """

        return key in self.config
    
    def reload(self) -> None:
        """ reload the config from file """

        self.load()