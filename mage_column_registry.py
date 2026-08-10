# %%
import os
import json
import yaml
import urllib.parse
import requests
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class NistApiBridge:
    """Handles external PubChem/NIST structure lookup queries with strict timeout fail-overs (MAGE-13)."""
    def __init__(self, timeout_sec=2.5):
        self.timeout = timeout_sec

    def query_smiles(self, smiles):
        try:
            # Official PubChem REST API endpoint for structure query (MAGE-13)
            encoded_smiles = urllib.parse.quote(smiles)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    return "PUBCHEM_MATCH_FOUND"
            return "NOT_FOUND"
        except requests.exceptions.Timeout:
            print(f"⚠️ External API Timeout ({self.timeout}s) for '{smiles}'. Falling back to ab initio MAGE simulation.")
            return "TIMEOUT_FALLBACK"
        except requests.exceptions.RequestException:
            return "NOT_FOUND"

class MageIngestor:
    """
    Stage 1.0 (Update): Smart Ingestion & Instrument Profiling.
    Now includes Column Intelligence and NIST API Failover checks.
    """
    def __init__(self, sys_config_path=None, registry_path="mage_column_registry.json"):
        if sys_config_path is None:
            env_p = os.environ.get("COCHEM_SYSTEM_CONFIG")
            if env_p and os.path.exists(env_p):
                sys_config_path = env_p
            else:
                root_p = Path(__file__).resolve().parent / "cochem_system_config.json"
                parent_root_p = Path(__file__).resolve().parents[1] / "cochem_system_config.json"
                if root_p.exists():
                    sys_config_path = str(root_p)
                elif parent_root_p.exists():
                    sys_config_path = str(parent_root_p)
                else:
                    sys_config_path = str(root_p)
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
            # Create a default system config if absent for testing
            os.makedirs(os.path.dirname(self.sys_config_path), exist_ok=True)
            default_sys = {"project_name": "CoChem-MAGE", "hpc_environments": {}}
            with open(self.sys_config_path, 'w') as f:
                json.dump(default_sys, f)
        with open(self.sys_config_path, 'r') as f:
            self.system_config = json.load(f)

    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            default_registry = {
                "columns": {
                    "DB-5MS": {"length_m": 30.0, "stationary_phase": "5% phenyl", "max_ramp_rate": 40.0},
                    "DB-WAX": {"length_m": 30.0, "stationary_phase": "Polyethylene Glycol", "max_ramp_rate": 25.0}
                }
            }
            with open(self.registry_path, 'w') as f:
                json.dump(default_registry, f)
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
                
        # Automated Column Intelligence Check (MAGE-14)
        if tpsa_values and self.instrument_profile.get("column_type") == "DB-5MS":
            median_tpsa = np.median(tpsa_values)
            if median_tpsa > 60.0:
                print(f"💡 COLUMN INTEL: Batch median TPSA is high ({median_tpsa:.1f}). Recommend silylation (TMS) or trifluoroacetylation (TFA) derivatization protocol prior to GC analysis.")
                
        return valid_queue
# %%