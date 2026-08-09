#!/usr/bin/env python3
"""
CoChem-NODE: HPC & SLURM Bridge
Module: cochem_node_bridge.py
Purpose: Establishes a fault-tolerant, key-authenticated SSH/SFTP transport 
         layer between the local pipeline and external HPC clusters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import paramiko

# Configure CoChem Standard Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - CoChem-NODE - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NodeBridge")

class NodeBridge:
    def __init__(self, config_path: str = "cochem_system_config.json"):
        """
        Initializes the HPC bridge by parsing the global registry.
        """
        self.config_path = Path(config_path)
        self.ssh_client = paramiko.SSHClient()
        # Enforce strict security policy by loading system host keys and rejecting unverified hosts (MAGE-16)
        self.ssh_client.load_system_host_keys()
        self.ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.sftp_client: Optional[paramiko.SFTPClient] = None
        self.connected = False
        
        self.hpc_config = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Loads the HPC configuration block from the authoritative registry."""
        if not self.config_path.exists():
            logger.warning(f"Authoritative registry {self.config_path} not found. Operating in localized mode.")
            return {}
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config.get("hpc_environments", {})
        except json.JSONDecodeError as e:
            logger.error(f"Registry corruption detected: {e}")
            return {}

    def _find_default_ssh_key(self) -> Optional[str]:
        """
        Discovers modern SSH private keys checking ed25519 and ecdsa before legacy id_rsa (MAGE-17).
        """
        ssh_dir = Path.home() / ".ssh"
        key_candidates = [
            ssh_dir / "id_ed25519",
            ssh_dir / "id_ecdsa",
            ssh_dir / "id_rsa"
        ]
        for key in key_candidates:
            if key.exists():
                return str(key)
        return str(ssh_dir / "id_ed25519")

    def connect(self, target_cluster: str = "primary") -> bool:
        """
        Establishes the SSH and SFTP connections securely using key-pairs.
        """
        if target_cluster not in self.hpc_config:
            logger.error(f"Cluster profile '{target_cluster}' not found in registry.")
            return False

        cluster_data = self.hpc_config[target_cluster]
        hostname = cluster_data.get("host")
        username = cluster_data.get("user")
        key_path = cluster_data.get("ssh_key_path", self._find_default_ssh_key())

        if not hostname or not username:
            logger.error("Incomplete HPC registry block. Hostname and User are required.")
            return False

        try:
            logger.info(f"Establishing bridge to {username}@{hostname}...")
            self.ssh_client.connect(
                hostname=hostname,
                username=username,
                key_filename=key_path,
                timeout=10,
                look_for_keys=True
            )
            self.sftp_client = self.ssh_client.open_sftp()
            self.connected = True
            logger.info("CoChem-NODE Bridge securely established.")
            return True
            
        except paramiko.AuthenticationException:
            logger.critical("Authentication failed. Verify SSH key pairs and permissions.")
            return False
        except Exception as e:
            logger.critical(f"Bridge connection severed: {str(e)}")
            return False

    def execute_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """
        Dispatches a command to the HPC head node and returns (exit_status, stdout, stderr).
        """
        if not self.connected:
            logger.error("Cannot execute command. Bridge is disconnected.")
            return 1, "", "Bridge disconnected."

        try:
            logger.debug(f"Dispatching: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode('utf-8').strip()
            err = stderr.read().decode('utf-8').strip()
            return exit_status, out, err
        except Exception as e:
            logger.error(f"Command dispatch failed: {str(e)}")
            return -1, "", str(e)

    def close(self) -> None:
        """Gracefully tears down the SSH and SFTP transports."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        self.connected = False
        logger.info("CoChem-NODE Bridge torn down cleanly.")

# Pre-flight unit test execution
if __name__ == "__main__":
    print("Running CoChem-NODE Pre-Flight Interface Test...")
    bridge = NodeBridge()
    if not bridge.hpc_config:
        print("[NOTICE] No HPC targets currently mapped in local cochem_system_config.json.")
        print("[NOTICE] NodeBridge is ready to accept routing requests once provisioned.")
    else:
        print("[SUCCESS] Discovered HPC configurations.")