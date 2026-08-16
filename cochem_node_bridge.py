#!/usr/bin/env python3
"""
CoChem-NODE: HPC & SLURM Bridge
Module: cochem_node_bridge.py
Purpose: Establishes a Parsl-driven asynchronous heterogeneous architecture 
         for optimized scheduling across HPC clusters (Slurm).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher

# Configure CoChem Standard Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - CoChem-NODE - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NodeBridge")

class NodeBridge:
    def __init__(self, config_path: str = "cochem_system_config.json") -> None:
        """
        Initializes the HPC bridge using Parsl.
        """
        self.config_path = Path(config_path)
        self.hpc_config = self._load_registry()
        self.connected = False

    def _load_registry(self) -> Dict[str, Any]:
        """Loads the HPC configuration block from the authoritative registry."""
        if not self.config_path.exists():
            logger.warning(f"Authoritative registry {self.config_path} not found. Operating in localized mode.")
            return {}
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.loads(f.read())
            return config.get("hpc_environments", {})
        except json.JSONDecodeError as e:
            logger.error(f"Registry corruption detected: {e}")
            return {}

    def connect(self, target_cluster: str = "primary") -> bool:
        """
        Loads Parsl config and starts the asynchronous execution environment.
        """
        if target_cluster not in self.hpc_config:
            logger.error(f"Cluster profile '{target_cluster}' not found in registry.")
            return False

        cluster_data = self.hpc_config[target_cluster]
        hostname = cluster_data.get("host")
        username = cluster_data.get("user")

        if not hostname or not username:
            logger.error("Incomplete HPC registry block. Hostname and User are required.")
            return False

        logger.info(f"Initializing Parsl Slurm/HighThroughputExecutor for {username}@{hostname}...")
        
        parsl_config = Config(
            executors=[
                HighThroughputExecutor(
                    label="slurm_hpc",
                    provider=SlurmProvider(
                        partition="compute",
                        nodes_per_block=1,
                        init_blocks=1,
                        max_blocks=4,
                        launcher=SrunLauncher(),
                        worker_init="module load miniconda; conda activate cochem"
                    ),
                )
            ],
            strategy='simple'
        )

        try:
            parsl.load(parsl_config)
            self.connected = True
            logger.info("CoChem-NODE Parsl Bridge securely established.")
            return True
        except Exception as e:
            logger.critical(f"Bridge connection severed during Parsl load: {str(e)}")
            return False

    def close(self) -> None:
        """Gracefully tears down the Parsl execution environment."""
        if self.connected:
            parsl.clear()
            self.connected = False
            logger.info("CoChem-NODE Parsl Bridge torn down cleanly.")

# Pre-flight unit test execution
if __name__ == "__main__":
    logger.info("Running CoChem-NODE Pre-Flight Interface Test...")
    bridge = NodeBridge()
    if not bridge.hpc_config:
        logger.info("[NOTICE] No HPC targets currently mapped in local cochem_system_config.json.")
        logger.info("[NOTICE] NodeBridge is ready to accept routing requests once provisioned.")
    else:
        logger.info("[SUCCESS] Discovered HPC configurations.")
