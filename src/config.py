"""Config loader with YAML support"""
from pathlib import Path
import yaml

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
