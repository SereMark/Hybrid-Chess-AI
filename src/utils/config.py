import os
import yaml

class Config:
    _instance = None
    
    def __new__(cls, config_path=None, mode="test"):
        if cls._instance is None and config_path is not None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init(config_path, mode)
        return cls._instance
    
    def _init(self, config_path, mode):
        self.path = config_path
        self.config = self._load()
        self.mode = mode
    
    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with open(self.path, 'r') as f:
            return yaml.safe_load(f)
    
    def set_mode(self, mode):
        if mode not in ["test", "prod"]:
            raise ValueError("Mode must be 'test' or 'prod'")
        self.mode = mode
    
    def get(self, key, default=None):
        keys = key.split('.')
        
        if len(keys) > 1 and keys[0] in self.config:
            section = self.config[keys[0]]
            if isinstance(section, dict):
                if self.mode == "prod" and "prod" in section and keys[1] in section["prod"]:
                    return section["prod"][keys[1]]
                
                if keys[1] in section:
                    if len(keys) == 2:
                        return section[keys[1]]
                    return self._get_nested(section, keys[1:], default)
        
        return self._get_nested(self.config, keys, default)
    
    def _get_nested(self, section, keys, default):
        current = section
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

def get_config(config_path=None, mode=None):
    instance = Config(config_path, mode if mode else "test")
    if mode is not None and config_path is None:
        instance.set_mode(mode)
    return instance