# cochem_canvas_target: cochem_mage_main.py
"""
Main orchestrator module for CoChem-MAGE.
This is the central entry point for the MAGE (Mass and Gas-chromatography Emulator) system.
"""

import os
import sys
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime
try:
    import h5py
except ImportError:
    h5py = None

from mage_fragmenter import MageFragmenter
from cochem_mage_sim import MageChromatographySim
from cochem_mage_export import MageExporter
from cochem_mage_opt import MageOptimizationEngine

class MAGEOrchestrator:
    """
    The main orchestrator that coordinates all MAGE activities.
    """
    
    def __init__(self, config_file: str = "cochem_mage_config.json"):
        """Initialize the MAGE orchestrator."""
        self.config_file = config_file
        self.config = self._load_config()
        self.is_initialized = False
        
    def _get_artifact_dir(self) -> Path:
        """Get the artifact directory path from environment variable or default to home."""
        artifact_dir = os.environ.get('COCHEM_ARTIFACT_DIR')
        if artifact_dir:
            return Path(artifact_dir)
        else:
            # Default to home directory with .cochem/artifacts
            return Path.home() / ".cochem" / "artifacts" / "mage"
        
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file {self.config_file} not found")
            # Return default config
            artifact_dir = self._get_artifact_dir()
            return {
                "project_name": "CoChem-MAGE",
                "version": "0.1.0",
                "data_dir": str(artifact_dir / "data")
            }
        except json.JSONDecodeError as e:
            print(f"❌ Error loading configuration: {e}")
            return {}
            
    def initialize(self):
        """Initialize the MAGE system."""
        print("🚀 Initializing CoChem-MAGE System...")
        
        # Create data directories
        artifact_dir = self._get_artifact_dir()
        data_dir = Path(self.config.get('data_dir', str(artifact_dir / "data")))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different modules
        (data_dir / "rrkm").mkdir(parents=True, exist_ok=True)
        (data_dir / "chrom_opt").mkdir(parents=True, exist_ok=True)
        (data_dir / "output").mkdir(parents=True, exist_ok=True)
        
        self.is_initialized = True
        print("✅ CoChem-MAGE initialized successfully")

    def export_to_h5(self, data: dict, h5_path: str = None) -> str:
        """
        Exports simulation results directly into standardized cochem_state.h5 dataset.
        (MAGE-19)
        """
        if h5_path is None:
            artifact_dir = self._get_artifact_dir()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            h5_path = str(artifact_dir / "cochem_state.h5")

        if h5py is None:
            print("⚠️ h5py not installed, skipping HDF5 serialization.")
            return h5_path

        if not isinstance(data, dict):
            return h5_path

        try:
            with h5py.File(h5_path, "a") as h5f:
                mage_grp = h5f.require_group("mage")
                now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                unique_grp_name = f"run_{now_str}_{uuid4().hex[:6]}"
                sim_grp = mage_grp.create_group(unique_grp_name)
                sim_grp.attrs["LAM_TRIGGER_REQUIRED"] = data.get("LAM_TRIGGER_REQUIRED", False)
                sim_grp.attrs["symmetry_group"] = data.get("symmetry_group", "C1")
                sim_grp.attrs["provenance_tag"] = data.get("provenance_tag", "[D]")
                
                results = data.get("results")
                if isinstance(results, list):
                    valid_results = [r for r in results if isinstance(r, dict)]
                    for idx, res in enumerate(valid_results):
                        sub = sim_grp.create_group(f"item_{idx}")
                        for k, v in res.items():
                            if isinstance(v, (int, float, str, bool)):
                                sub.attrs[k] = v
                        if "provenance_tag" not in sub.attrs:
                            sub.attrs["provenance_tag"] = res.get("provenance_tag", "[D]" if res.get("status") == "COMPUTED" else "[E]")
            print(f"💾 Exported MAGE state to HDF5 lake: {h5_path}")
        except Exception as e:
            print(f"⚠️ HDF5 export warning: {e}")
            
        return h5_path
        
    def run_simulation(self, simulation_type: str, input_data: dict) -> dict:
        """Run a specific type of MAGE simulation."""
        if not self.is_initialized:
            raise RuntimeError("MAGE system must be initialized before running simulations")
            
        if not isinstance(input_data, dict):
            input_data = {}
            
        # Ensure it accepts MPQC JSON payloads and doesn't conflict with MLFF inference
        is_mpqc_payload = "mpqc" in input_data or str(input_data.get("generator", "")).lower() == "mpqc"
        if is_mpqc_payload:
            print("ℹ️ Detected MPQC JSON payload. Ensuring MLFF inference pipeline compatibility with MPQC single-points.")
            # Extract smiles from MPQC payload if nested
            if "smiles" not in input_data and "molecule" in input_data and isinstance(input_data["molecule"], dict):
                input_data["smiles"] = input_data["molecule"].get("smiles")
            
        print(f"🔬 Running {simulation_type} simulation...")
        results = {}
        
        sim_type_str = str(simulation_type or "").lower()
        if sim_type_str == "rrkm":
            fragmenter = MageFragmenter()
            graph_data = input_data.get("graph_data")
            raw_smiles = input_data.get("smiles") or input_data.get("molecule")
            if graph_data is not None:
                results["spectrum"] = fragmenter.simulate_spectrum(graph_data)
            elif raw_smiles is not None:
                smiles = str(raw_smiles)
                smiles_lower = smiles.lower()
                if smiles_lower == "benzene":
                    smiles = "c1ccccc1"
                elif smiles_lower == "aspirin":
                    smiles = "CC(=O)Oc1ccccc1C(=O)O"
                g = fragmenter.graph_from_smiles(smiles)
                results["spectrum"] = fragmenter.simulate_spectrum(g)
            else:
                g = fragmenter.graph_from_smiles("c1ccccc1")
                results["spectrum"] = fragmenter.simulate_spectrum(g)
            results["status"] = "COMPLETED"
            
        elif sim_type_str in ["chromatography", "chrom"]:
            col_config = input_data.get("column_config", {"length_m": 30.0, "stationary_phase": "5% phenyl"})
            sim = MageChromatographySim(col_config)
            jobs = input_data.get("jobs", [
                {"id": "mol_0", "mw": 180.15, "logp": 1.19, "tpsa": 63.6, "status": "CACHED"},
                {"id": "mol_1", "mw": 94.11,  "logp": 1.46, "tpsa": 20.2, "status": "COMPUTED"}
            ])
            sim_jobs = sim.simulate_retention(jobs, temperature_ramp_rate=input_data.get("ramp_rate", 10.0))
            t_axis, trace = sim.build_chromatogram(sim_jobs)
            results["jobs"] = sim_jobs
            results["time_axis"] = t_axis.tolist()
            results["chromatogram"] = trace.tolist()
            results["status"] = "COMPLETED"
        else:
            results["status"] = "UNKNOWN_SIMULATION_TYPE"
        
        print(f"✅ {simulation_type} simulation completed")
        return results
        
    def generate_report(self, output_dir: str = "./reports", job_queue: list = None) -> str:
        """Generate comprehensive report of simulation findings."""
        print(f"📄 Generating MAGE report in {output_dir}")
        exporter = MageExporter(output_dir=output_dir)
        if job_queue is None:
            job_queue = [
                {"smiles": "c1ccccc1", "chemical_class": "Aromatic", "mw": 78.11, "tpsa": 0.0, "logp": 2.13, "status": "COMPUTED", "predicted_tr": 3.5},
                {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "chemical_class": "Ester/Acid", "mw": 180.16, "tpsa": 63.6, "logp": 1.19, "status": "COMPUTED", "predicted_tr": 8.2}
            ]
        
        html_path = exporter.build_interactive_chromatogram(job_queue, filename="mage_chromatogram.html")
        scribe_path = exporter.export_scribe_payload(job_queue, self.config, filename="mage_scribe_payload.json")
        
        # Also export to HDF5
        self.export_to_h5({"results": job_queue})

        print("✅ MAGE report generated")
        return html_path

def main():
    """Main entry point for CoChem-MAGE."""
    print("Starting CoChem-MAGE Orchestrator")
    
    orchestrator = MAGEOrchestrator()
    orchestrator.initialize()
    
    # Example usage
    orchestrator.run_simulation("rrkm", {"molecule": "benzene"})
    orchestrator.generate_report("./reports")
    
if __name__ == "__main__":
    main()