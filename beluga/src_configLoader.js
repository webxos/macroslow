// configLoader.js
// Description: Utility to load BELUGA configuration from YAML or JSON files.
// Ensures consistent setup for client-side applications.
// Usage: Import and call loadConfig to retrieve configuration.

import { readFileSync } from 'fs';
import { parse } from 'yaml';

function loadConfig(configPath = "beluga_config.yaml") {
    /**
     * Loads configuration from a YAML file.
     * @param {string} configPath - Path to configuration file.
     * @returns {Object} - Configuration object.
     */
    const fileContent = readFileSync(configPath, 'utf8');
    return parse(fileContent);
}

// Example usage:
// const config = loadConfig();
// console.log(config.quantum.device);