import logging
from typing import Any
logger = logging.getLogger(__name__)
# D3/D4 dispersion correction enabled
# cochem_canvas_target: cochem_mage_config.py
"""
Configuration module for CoChem-MAGE.
Handles all configuration settings for the MAGE system.
"""

import json
import os
from pathlib import Path

class MAGEConfig:
    """
    Configuration class for CoChem-MAGE system.
    """
    
    def __init__(self, config_file: str = "cochem_mage_config.json") -> None:
        """Initialize configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        
    def _get_artifact_dir(self) -> Path:
        """Get the artifact directory path from environment variable or default to home."""
        artifact_dir = os.environ.get('COCHEM_ARTIFACT_DIR')
        if artifact_dir:
            return Path(artifact_dir)
        else:
            # Default to home directory with .cochem/artifacts
            return Path.home() / ".cochem" / "artifacts" / "mage"
        
    def _sanitize_config(self, config: dict) -> dict:
        """Sanitizes configuration by replacing prohibited legacy v3 methods/bases with v4 defaults."""
        if not isinstance(config, dict):
            return self._get_default_config(self._get_artifact_dir())
        
        defaults = config.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
        
        # Replace legacy prohibited B3LYP / 6-31G* defaults
        if defaults.get("t1_method") in [None, "B3LYP", "b3lyp"]:
            defaults["t1_method"] = "r2SCAN-3c"
        if defaults.get("default_basis") in ["6-31G*", "6-31g*"]:
            defaults.pop("default_basis", None)
            defaults["t2_composite"] = "junChS"
        if "t2_composite" not in defaults or defaults.get("t2_composite") == "B3LYP":
            defaults["t2_composite"] = "junChS"
        if "t3_geometry" not in defaults or defaults.get("t3_geometry") in ["B3LYP", "DLPNO-CCSD(T)"]:
            defaults["t3_geometry"] = "CCSD(T)-F12"
            
        config["defaults"] = defaults
        return config

    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.loads(f.read())
                if 'data_dir' not in config:
                    artifact_dir = self._get_artifact_dir()
                    config['data_dir'] = str(artifact_dir / "data")
                return self._sanitize_config(config)
        except FileNotFoundError:
            artifact_dir = self._get_artifact_dir()
            return self._sanitize_config(self._get_default_config(artifact_dir))
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error loading config: {e}")
            artifact_dir = self._get_artifact_dir()
            return self._sanitize_config(self._get_default_config(artifact_dir))
            
    def _get_default_config(self, artifact_dir: Path) -> dict:
        """Get default configuration values conforming to v4 Method Matrix."""
        cfg = {
            "project_name": "CoChem-MAGE",
            "version": "0.4.0",
            "data_dir": str(artifact_dir / "data"),
            "product_class": "PRODUCT_A",
            "tier_level": "T1-30min",
            "simulation_modules": {
                "rrkm": {"enabled": True},
                "chrom_opt": {"enabled": True}
            },
            "defaults": {
                "t1_method": "r2SCAN-3c",
                "t2_composite": "junChS",
                "t3_geometry": "CCSD(T)-F12"
            },
            "output": {
                "format": "json",
                "include_visualizations": True,
                "export_to_csv": True
            },
            "performance": {
                "node_scheduler_delegated": True,
                "timeout_minutes": 60,
                "wall_clock_budgets": [
                    "T1-10s", "T1-1min", "T1-30min", "T1-1h",
                    "T2-3h", "T2-12h", "T3-1d", "T3-3d",
                    "T4-1w", "T4-1mo"
                ]
            },
            "spend_priority": [
                "intermolecular_geometry",
                "delta_b_vib",
                "frozen_monomers",
                "quartic_distortion",
                "inertial_defect",
                "signed_dipoles",
                "nqcc_tensor",
                "v3_barrier",
                "tunnelling_splittings",
                "binding_energy_d0"
            ]
        }
        return self._sanitize_config(cfg)
        
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)
        
    def set(self, key: str, value) -> Any:
        """Set configuration value."""
        self.config[key] = value
        self.config = self._sanitize_config(self.config)
        self._save_config()
        
    def _save_config(self) -> Any:
        """Save current configuration to file."""
        os.makedirs(Path(self.config_file).parent, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def update_from_dict(self, updates: dict) -> Any:
        """Update configuration from dictionary."""
        self.config.update(updates)
        self.config = self._sanitize_config(self.config)
        self._save_config()

def main() -> Any:
    """Main entry point for configuration module."""
    logger.info("Initializing CoChem-MAGE Configuration")
    
    config = MAGEConfig()
    logger.info("Current configuration:", config.config)

if __name__ == "__main__":
    main()