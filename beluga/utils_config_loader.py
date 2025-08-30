# config_loader.py
# Description: Utility to load BELUGA configuration from YAML files.
# Ensures consistent setup for database, quantum, and sensor parameters.
# Usage: Call load_config to retrieve configuration dictionary.

import yaml

def load_config(config_path: str = "beluga_config.yaml") -> dict:
    """
    Loads BELUGA configuration from a YAML file.
    Input: Path to configuration file.
    Output: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage:
# config = load_config()
# print(config["quantum"]["device"])