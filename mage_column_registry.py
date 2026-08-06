# %%
import os
import json
import yaml
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class NistApiBridge:
    """Handles external queries with strict timeout fail-overs."""
    def __init__(self, timeout_sec=2.5):
        self.timeout = timeout_sec

    def query_smiles(self, smiles):
        try:
            # Mock NIST Webbook API endpoint for architectural demonstration
            response = requests.get(f"https://webbook.nist.gov/cgi/cbook.cgi?Smiles={smiles}", timeout=self.timeout)
            response.raise_for_status()
            return "NIST_MATCH_FOUND"
        except requests.exceptions.Timeout:
            print(f"⚠️ NIST API Timeout ({self.timeout}s) for '{smiles}'. Falling back to ab initio MAGE simulation.")
            return "TIMEOUT_FALLBACK"
        except requests.exceptions.RequestException:
            return "NOT_FOUND"

class MageIngestor:
    """
    Stage 1.0 (Update): Smart Ingestion & Instrument Profiling.
    Now includes Column Intelligence and NIST API Failover checks.
    """
    def __init__(self, sys_config_path="./cochem_setup/cochem_system_config.json", registry_path="mage_column_registry.json"):
        self.sys_config_path = sys_config_path
        self.registry_path = registry_path
        self.system_config = {}
        self.column_registry = {}
        self.instrument_profile = {}
        self.nist_bridge = NistApiBridge()
        
        self._verify_environment()
        self._load_registry()

    def _verify_environment(self):
        if not os.path.exists(self.sys_config_path):
            raise FileNotFoundError(f"❌ MAGE Ingest Error: System config not found at {self.sys_config_path}.")
        with open(self.sys_config_path, 'r') as f:
            self.system_config = json.load(f)

    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"❌ MAGE Ingest Error: Column registry not found at {self.registry_path}.")
        with open(self.registry_path, 'r') as f:
            self.column_registry = json.load(f)

    def load_instrument_profile(self, yaml_path):
        if not os.path.exists(yaml_path):
            default_yaml = {
                "instrument_name": "Agilent_5977B_Default",
                "column_type": "DB-5MS",
                "carrier_gas": "Helium",
                "initial_temperature_C": 50.0,
                "injection_volume_uL": 1.0,
                "split_ratio": 50
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(default_yaml, f)
            
        with open(yaml_path, 'r') as f:
            self.instrument_profile = yaml.safe_load(f)
            
        req_col = self.instrument_profile.get("column_type")
        if req_col not in self.column_registry.get("columns", {}):
            raise ValueError(f"❌ Requested column '{req_col}' not found.")
            
        self.instrument_profile["column_physics"] = self.column_registry["columns"][req_col]
        return self.instrument_profile

    def sanitize_molecule_queue(self, smiles_list):
        valid_queue = []
        tpsa_values = []
        
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                db_status = self.nist_bridge.query_smiles(smi)
                
                valid_queue.append({
                    "id": f"mol_{idx}",
                    "smiles": smi,
                    "rdkit_mol": mol,
                    "status": "SANITIZED",
                    "nist_status": db_status
                })
                tpsa_values.append(Descriptors.TPSA(mol))
            except Exception:
                continue
                
        # Automated Column Intelligence Check
        if tpsa_values and self.instrument_profile.get("column_type") == "DB-5MS":
            median_tpsa = np.median(tpsa_values)
            if median_tpsa > 60.0:
                print(f"💡 COLUMN INTEL: Batch median TPSA is high ({median_tpsa:.1f}). Recommend switching from non-polar DB-5MS to polar DB-WAX to prevent tailing.")
                
        return valid_queue
# %%