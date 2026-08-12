import logging
logger = logging.getLogger(__name__)
# D3/D4 dispersion correction enabled
import os
import sys
import json
import shutil
import tempfile
from typing import Any
import pytest
import numpy as np
from pathlib import Path

from cochem_mage_config import MAGEConfig
from cochem_mage_sim import determine_product_class, MageChromatographySim
from mage_graph_setup import MageWorkflowDAGCompiler
from cochem_mage_export import MageExporter
from cochem_mage_telemetry import MageTelemetryBridge
from cochem_mage_main import MAGEOrchestrator

try:
    import h5py
except ImportError:
    h5py = None


class TestFocalArea1ConfigAndDirs:
    """Focal Area 1: Config sanitization and missing directory creation."""

    def test_config_sanitization_legacy_methods(self, tmp_path) -> None:
        cfg_file = tmp_path / "custom_config.json"
        raw_data = {
            "defaults": {
                "t1_method": "B3LYP",
                "default_basis": "6-31G*",
                "t2_composite": "B3LYP",
                "t3_geometry": "B3LYP"
            }
        }
        cfg_file.write_text(json.dumps(raw_data), encoding="utf-8")
        
        cfg = MAGEConfig(str(cfg_file))
        assert cfg.get("defaults")["t1_method"] == "r2SCAN-3c"
        assert cfg.get("defaults")["t2_composite"] == "junChS"
        assert cfg.get("defaults")["t3_geometry"] == "CCSD(T)-F12"
        assert "default_basis" not in cfg.get("defaults")

    def test_config_missing_parent_directory_creation(self, tmp_path) -> None:
        nested_cfg_file = tmp_path / "deep" / "nested" / "dir" / "config.json"
        assert not nested_cfg_file.parent.exists()

        cfg = MAGEConfig(str(nested_cfg_file))
        cfg.set("test_key", "test_val")
        
        assert nested_cfg_file.exists()
        with open(nested_cfg_file, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        assert data.get("test_key") == "test_val"

    def test_orchestrator_directory_creation(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("COCHEM_ARTIFACT_DIR", str(tmp_path / "artifact_root"))
        cfg_file = tmp_path / "orch_cfg.json"
        
        orch = MAGEOrchestrator(str(cfg_file))
        orch.initialize()

        data_dir = tmp_path / "artifact_root" / "data"
        assert data_dir.exists()
        assert (data_dir / "rrkm").exists()
        assert (data_dir / "chrom_opt").exists()
        assert (data_dir / "output").exists()
        assert orch.is_initialized is True


class TestFocalArea2ProductClassPrecedence:
    """Focal Area 2: Product Class decision tree precedence (is_difference_calc vs has_measured_parent)."""

    def test_product_class_a_default(self) -> None:
        res = determine_product_class({})
        assert res["product_class"] == "PRODUCT_A"
        assert res["target_accuracy_window"] == "±0.3% to ±0.5%"
        assert res["recommended_tier"] == "T1-30min"
        assert res["provenance_tag"] == "[E]"

    def test_product_class_b_measured_parent(self) -> None:
        res1 = determine_product_class({"measured_parent_isotopologue": "13C_benzene"})
        assert res1["product_class"] == "PRODUCT_B"
        assert res1["target_accuracy_window"] == "±0.03% to ±0.1%"
        assert res1["recommended_tier"] == "T2-12h"
        assert res1["provenance_tag"] == "[D]"

        res2 = determine_product_class({"parent_experimental_spectrum": [10.0, 20.0]})
        assert res2["product_class"] == "PRODUCT_B"
        assert res2["provenance_tag"] == "[D]"

    def test_product_class_c_difference_calc(self) -> None:
        res1 = determine_product_class({"is_difference_calculation": True})
        assert res1["product_class"] == "PRODUCT_C"
        assert res1["target_accuracy_window"] == "Difference cancellation window"
        assert res1["recommended_tier"] == "T1-1h"
        assert res1["provenance_tag"] == "[D]"

        res2 = determine_product_class({"relative_shift_mode": True})
        assert res2["product_class"] == "PRODUCT_C"
        assert res2["provenance_tag"] == "[D]"

    def test_product_class_precedence_c_over_b(self) -> None:
        # Both is_difference_calculation AND measured_parent_isotopologue are set
        data = {
            "is_difference_calculation": True,
            "measured_parent_isotopologue": "13C_benzene",
            "parent_experimental_spectrum": [100.0, 200.0]
        }
        res = determine_product_class(data)
        assert res["product_class"] == "PRODUCT_C"
        assert res["recommended_tier"] == "T1-1h"
        assert res["provenance_tag"] == "[D]"

    def test_product_class_non_dict_safety(self) -> None:
        for invalid in [None, [], "invalid_string", 12345]:
            res = determine_product_class(invalid)
            assert res["product_class"] == "PRODUCT_A"
            assert res["provenance_tag"] == "[E]"


class TestFocalArea3VanDeemterGuards:
    """Focal Area 3: Van Deemter HETP boundary guards (extreme values, None keys, negative velocity)."""

    @pytest.fixture
    def sim(self) -> Any:
        return MageChromatographySim({"length_m": 30.0, "stationary_phase": "5% phenyl"})

    def test_negative_and_zero_velocity(self, sim) -> None:
        # Negative velocity should be clamped to 1e-3
        H_neg, u_neg, tag_neg = sim.compute_van_deemter_hetp(u_cm_s=-25.0)
        assert u_neg == 1e-3
        assert H_neg > 0
        assert tag_neg == "[E]"

        # Zero velocity should also be clamped to 1e-3
        H_zero, u_zero, tag_zero = sim.compute_van_deemter_hetp(u_cm_s=0.0)
        assert u_zero == 1e-3
        assert H_zero > 0

    def test_extreme_high_velocity(self, sim) -> None:
        H_high, u_high, tag = sim.compute_van_deemter_hetp(u_cm_s=1e6)
        assert u_high == 1e6
        assert not np.isnan(H_high)
        assert not np.isinf(H_high)
        assert H_high > 0

    def test_none_keys_in_station_phase_params(self, sim) -> None:
        params_none = {
            "film_thickness_um": None,
            "inner_diameter_mm": None,
            "retention_factor_k": None,
            "binary_diffusion_m2_s": None,
            "stationary_diffusion_m2_s": None
        }
        H, u, tag = sim.compute_van_deemter_hetp(u_cm_s=20.0, station_phase_params=params_none)
        assert tag == "[D]"
        assert not np.isnan(H)
        assert H > 0

    def test_singular_retention_factor(self, sim) -> None:
        # k = -1.0 leads to (1+k) = 0 in denominator without guard
        params_singular = {
            "retention_factor_k": -1.0,
            "film_thickness_um": 0.25,
            "inner_diameter_mm": 0.25,
            "binary_diffusion_m2_s": 1e-5,
            "stationary_diffusion_m2_s": 1e-9
        }
        H, u, tag = sim.compute_van_deemter_hetp(u_cm_s=15.0, station_phase_params=params_singular)
        assert tag == "[D]"
        assert not np.isnan(H)
        assert not np.isinf(H)
        assert H > 0

    def test_zero_diffusion_constants(self, sim) -> None:
        params_zero_diff = {
            "retention_factor_k": 5.0,
            "binary_diffusion_m2_s": 0.0,
            "stationary_diffusion_m2_s": 0.0
        }
        H, u, tag = sim.compute_van_deemter_hetp(u_cm_s=10.0, station_phase_params=params_zero_diff)
        assert tag == "[D]"
        assert not np.isnan(H)
        assert not np.isinf(H)
        assert H > 0


class TestFocalArea4DAGCompilerNodeOrderAndTags:
    """Focal Area 4: DAG compiler node order and provenance tags."""

    def test_dag_compiler_step_order_and_provenance_tags(self) -> None:
        compiler = MageWorkflowDAGCompiler()
        dag = compiler.build_spend_priority_dag({"smiles": "c1ccccc1"})

        assert dag["total_steps"] == 10
        nodes = dag["spend_priority_nodes"]
        assert len(nodes) == 10

        expected_steps = [
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

        for idx, node in enumerate(nodes):
            assert node["step_number"] == idx + 1
            assert node["step_id"] == expected_steps[idx]
            
            # Nodes 1..7 (idx 0..6) get [D], Nodes 8..10 (idx 7..9) get [E]
            if idx < 7:
                assert node["provenance_tag"] == "[D]", f"Node {idx+1} ({node['step_id']}) should be [D]"
            else:
                assert node["provenance_tag"] == "[E]", f"Node {idx+1} ({node['step_id']}) should be [E]"

    def test_dag_compiler_dependency_edges(self) -> None:
        compiler = MageWorkflowDAGCompiler()
        dag = compiler.build_spend_priority_dag({})
        edges = dag["dependency_edges"]

        assert len(edges) == 9
        for i, edge in enumerate(edges):
            assert edge["from"] == f"step_{i+1}"
            assert edge["to"] == f"step_{i+2}"


class TestFocalArea5TelemetryAndHDF5Export:
    """Focal Area 5: Telemetry & HDF5 export consecutive exports and provenance tags."""

    def test_consecutive_hdf5_exports_no_collision(self, tmp_path) -> None:
        if h5py is None:
            pytest.skip("h5py not installed")

        h5_file = str(tmp_path / "cochem_state.h5")
        orch = MAGEOrchestrator()

        sample_data = {
            "LAM_TRIGGER_REQUIRED": True,
            "symmetry_group": "C2v",
            "provenance_tag": "[D]",
            "results": [
                {"smiles": "c1ccccc1", "status": "COMPUTED", "provenance_tag": "[D]"},
                {"smiles": "CC(=O)O", "status": "CACHED", "provenance_tag": "[E]"}
            ]
        }

        # Perform 10 rapid consecutive exports
        for _ in range(10):
            res_path = orch.export_to_h5(sample_data, h5_file)
            assert res_path == h5_file

        with h5py.File(h5_file, "r") as f:
            assert "mage" in f
            mage_grp = f["mage"]
            assert len(mage_grp.keys()) == 10
            
            for key in mage_grp.keys():
                sim_grp = mage_grp[key]
                assert sim_grp.attrs["provenance_tag"] == "[D]"
                assert sim_grp.attrs["symmetry_group"] == "C2v"
                assert "item_0" in sim_grp
                assert sim_grp["item_0"].attrs["provenance_tag"] == "[D]"
                assert "item_1" in sim_grp
                assert sim_grp["item_1"].attrs["provenance_tag"] == "[E]"

    def test_telemetry_bridge_state_updates_and_provenance_tags(self) -> None:
        bridge = MageTelemetryBridge()
        
        # Initial status check
        init_state = bridge.current_state
        assert init_state["status"] == "IDLE"
        assert init_state["provenance_tag"] == "[D]"

        # Update state with [D] tag
        bridge.update_state("RUNNING", 50.0, "Computing RRKM", isomer="Benzene", provenance_tag="[D]")
        s1 = bridge.current_state
        assert s1["status"] == "RUNNING"
        assert s1["progress_percent"] == 50.0
        assert s1["current_operation"] == "Computing RRKM"
        assert s1["active_isomer"] == "Benzene"
        assert s1["provenance_tag"] == "[D]"

        # Update state with [E] tag
        bridge.update_state("COMPLETE", 100.0, "Sim Done", rs=1.62, provenance_tag="[E]")
        s2 = bridge.current_state
        assert s2["status"] == "COMPLETE"
        assert s2["progress_percent"] == 100.0
        assert s2["optimization_rs"] == 1.62
        assert s2["provenance_tag"] == "[E]"
