import json
import os
import logging

# --- CUSTOMIZATION POINT: Configure logging for configuration management ---
# Replace 'CHIMERA_ConfigManager' with your custom logger name and adjust level or output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CHIMERA_ConfigManager")

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()  # --- CUSTOMIZATION POINT: Initialize with default config ---

    def load_config(self) -> Dict:
        # --- CUSTOMIZATION POINT: Customize config loading logic ---
        # Add validation or environment variable overrides; supports Dune 3.20.0 % forms
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {"api_port": 8000, "heads": 4}

    def save_config(self, config: Dict):
        # --- CUSTOMIZATION POINT: Customize config saving logic ---
        # Add backup or encryption; supports OCaml Dune 3.20.0 watch mode
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        logger.info(f"Saved config to {self.config_file}")

# --- CUSTOMIZATION POINT: Instantiate and export config manager ---
# Integrate with your application
config_manager = ConfigManager()