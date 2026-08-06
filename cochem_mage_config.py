# cochem_canvas_target: cochem_mage_config.py
"""
Configuration module for CoChem-MAGE.
Handles all configuration settings for the MAGE system.
"""

import json
from pathlib import Path

class MAGEConfig:
    """
    Configuration class for CoChem-MAGE system.
    """
    
    def __init__(self, config_file: str = "cochem_mage_config.json"):
        """Initialize configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> dict:
        """Get default configuration values."""
        return {
            "project_name": "CoChem-MAGE",
            "version": "0.1.0",
            "data_dir": "./cochem_mage_data",
            "simulation_modules": {
                "rrkm": {"enabled": True},
                "chrom_opt": {"enabled": True}
            },
            "input": {
                "max_molecules": 100,
                "default_basis": "6-31G*",
                "default_method": "B3LYP"
            },
            "output": {
                "format": "json",
                "include_visualizations": True,
                "export_to_csv": True
            },
            "performance": {
                "max_concurrent_jobs": 4,
                "memory_limit_gb": 8,
                "timeout_minutes": 60
            }
        }
        
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        return self.config.get(key, default)
        
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value
        self._save_config()
        
    def _save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def update_from_dict(self, updates: dict):
        """Update configuration from dictionary."""
        self.config.update(updates)
        self._save_config()

def main():
    """Main entry point for configuration module."""
    print("Initializing CoChem-MAGE Configuration")
    
    config = MAGEConfig()
    print("Current configuration:", config.config)

if __name__ == "__main__":
    main()