"""
CoChem-MAGE Deep Module Empirical Stress Test Generator & Adversarial Fuzzing Harness.
Author: EMPIRICAL CHALLENGER
Target: CoChem-MAGE (Milestone M7 Iteration 2 Verification)

This module executes deep, exhaustive stress tests against all CoChem-MAGE subsystems.
It verifies physical boundary guards, input sanitization, zero NaN/ZeroDivision errors,
concurrent HDF5 serializations, and 10x repeatable test execution.
"""

import os
import sys
import json
import math
import shutil
import tempfile
import threading
import numpy as np
import pytest
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# Add CoChem-MAGE root to path
MAGE_DIR = Path(__file__).parent.parent.resolve()
if str(MAGE_DIR) not in sys.path:
    sys.path.insert(0, str(MAGE_DIR))

from cochem_mage_config import MAGEConfig
from cochem_mage_main import MAGEOrchestrator
from cochem_mage_sim import MageChromatographySim, determine_product_class, calculate_kovats_ri_isothermal, calculate_kovats_ri_tp
from cochem_mage_opt import MageOptimizationEngine
from cochem_mage_export import MageExporter
from mage_fragmenter import MageFragmenter
from mage_graph_setup import MageGraphBuilder, MageWorkflowDAGCompiler
from cochem_mage_telemetry import MageTelemetryBridge
from cochem_mage_cache import MageDescriptorCache


# ==============================================================================
# CATEGORY 1: CONFIGURATION SANITIZATION & STRESS
# ==============================================================================

def test_config_sanitization_legacy_overrides(tmp_path):
    """Verifies that prohibited v3 methods (B3LYP, 6-31G*, Calc_Hess) are sanitized."""
    cfg_file = tmp_path / "custom_config.json"
    
    dirty_data = {
        "defaults": {
            "t1_method": "B3LYP",
            "default_basis": "6-31G*",
            "t2_composite": "B3LYP"
        },
        "Calc_Hess": True
    }
    with open(cfg_file, "w") as f:
        json.dump(dirty_data, f)
        
    cfg = MAGEConfig(config_file=str(cfg_file))
    
    defaults = cfg.get("defaults")
    assert defaults["t1_method"] == "r2SCAN-3c"
    assert defaults["t2_composite"] == "junChS"
    assert defaults["t3_geometry"] == "CCSD(T)-F12"
    assert "default_basis" not in defaults


def test_config_missing_parent_directory_creation(tmp_path):
    """Verifies saving config when parent directory does not exist auto-creates directories."""
    nested_dir = tmp_path / "deeply" / "nested" / "dir"
    cfg_file = nested_dir / "mage_config.json"
    
    cfg = MAGEConfig(config_file=str(cfg_file))
    cfg.set("project_name", "CoChem-MAGE-Stress")
    
    assert cfg_file.exists()
    with open(cfg_file, "r") as f:
        loaded = json.load(f)
    assert loaded.get("project_name") == "CoChem-MAGE-Stress"


def test_config_sanitize_non_dict_input():
    """Verifies _sanitize_config handles non-dict inputs without crashing."""
    cfg = MAGEConfig()
    result = cfg._sanitize_config(None)
    assert isinstance(result, dict)
    assert result.get("defaults", {}).get("t1_method") == "r2SCAN-3c"
    
    result_str = cfg._sanitize_config("not a dict")
    assert isinstance(result_str, dict)


# ==============================================================================
# CATEGORY 2: PRODUCT CLASS ROUTING DECISION TREE PRECEDENCE
# ==============================================================================

def test_product_class_non_dict_type_safety():
    """Verifies determine_product_class handles non-dict types safely."""
    for invalid in [None, "invalid_str", 12345, [1, 2, 3], True, 3.14]:
        res = determine_product_class(invalid)
        assert res["product_class"] == "PRODUCT_A"
        assert res["recommended_tier"] == "T1-30min"
        assert res["provenance_tag"] == "[E]"


def test_product_class_precedence_isotopologue_difference():
    """
    Method Matrix §1.2 precedence check:
    Difference calculations with parent experimental spectrum MUST route to PRODUCT_C (T1-1h).
    """
    combined_input = {
        "is_difference_calculation": True,
        "measured_parent_isotopologue": "13C1-benzene",
        "parent_experimental_spectrum": [10.0, 20.0, 30.0]
    }
    res = determine_product_class(combined_input)
    assert res["product_class"] == "PRODUCT_C"
    assert res["recommended_tier"] == "T1-1h"
    assert res["provenance_tag"] == "[D]"


def test_product_class_standard_routing():
    """Verifies standard Product A (de novo) and Product B (measured parent) routing."""
    res_a = determine_product_class({})
    assert res_a["product_class"] == "PRODUCT_A"
    assert res_a["recommended_tier"] == "T1-30min"

    res_b = determine_product_class({"measured_parent_isotopologue": "benzene"})
    assert res_b["product_class"] == "PRODUCT_B"
    assert res_b["recommended_tier"] == "T2-12h"


# ==============================================================================
# CATEGORY 3: CHROMATOGRAPHY SIMULATION & PHYSICAL BOUNDARY GUARDS
# ==============================================================================

def test_van_deemter_hetp_none_parameters():
    """Verifies Golay HETP handles explicit None parameters without TypeError."""
    sim = MageChromatographySim({"length_m": 30.0, "stationary_phase": "5% phenyl"})
    
    none_params = {
        "film_thickness_um": None,
        "inner_diameter_mm": None,
        "retention_factor_k": None,
        "binary_diffusion_m2_s": None,
        "stationary_diffusion_m2_s": None
    }
    hetp_mm, u_cm_s, tag = sim.compute_van_deemter_hetp(station_phase_params=none_params)
    assert not math.isnan(hetp_mm)
    assert not math.isinf(hetp_mm)
    assert hetp_mm > 0.0
    assert tag == "[D]"


def test_van_deemter_hetp_singular_retention_factor():
    """Verifies singular retention factor k = -1.0 produces zero ZeroDivisionError."""
    sim = MageChromatographySim({"length_m": 30.0})
    singular_params = {"retention_factor_k": -1.0}
    
    hetp_mm, u_cm_s, tag = sim.compute_van_deemter_hetp(station_phase_params=singular_params)
    assert not math.isnan(hetp_mm)
    assert not math.isinf(hetp_mm)
    assert hetp_mm > 0.0


def test_van_deemter_hetp_negative_and_extreme_velocities():
    """Verifies negative carrier velocity u < 0 is clamped and does not yield negative HETP."""
    sim = MageChromatographySim({"length_m": 30.0})
    
    hetp_neg, u_neg, tag = sim.compute_van_deemter_hetp(u_cm_s=-50.0)
    assert u_neg == 1e-3
    assert hetp_neg > 0.0
    
    hetp_zero, u_zero, tag = sim.compute_van_deemter_hetp(u_cm_s=0.0)
    assert u_zero == 1e-3
    assert hetp_zero > 0.0


def test_randic_connectivity_invalid_smiles():
    """Verifies Randić connectivity index returns (0.0, 0.0) for invalid SMILES."""
    sim = MageChromatographySim({"length_m": 30.0})
    for bad_smi in [None, "", "INVALID_SMILES", 12345]:
        c0, c1 = sim._compute_chi_indices(bad_smi)
        assert c0 == 0.0
        assert c1 == 0.0


def test_abraham_solvation_extreme_descriptors():
    """Verifies Abraham solvation parameters for extreme/unphysical descriptors."""
    sim = MageChromatographySim({"length_m": 30.0})
    extreme = {"smiles": "c1ccccc1", "mw": 100000.0, "logp": -100.0, "tpsa": 5000.0}
    ab = sim._abraham_solvation_parameters(extreme)
    for k, v in ab.items():
        assert not math.isnan(v)
        assert not math.isinf(v)


def test_build_chromatogram_empty_and_null_jobs():
    """Verifies build_chromatogram with empty jobs or jobs missing retention time."""
    sim = MageChromatographySim({"length_m": 30.0})
    
    t_axis, trace = sim.build_chromatogram([])
    assert len(t_axis) == 10000
    assert np.all(trace == 0.0)

    bad_jobs = [{"id": "mol_bad", "mw": 100.0}]
    t_axis, trace = sim.build_chromatogram(bad_jobs)
    assert len(t_axis) == 10000


def test_kovats_ri_math_functions():
    """Verifies Kovats RI calculation functions handle edge cases (equal retention times)."""
    ri_iso = calculate_kovats_ri_isothermal(t_rx=5.0, t_rn=5.0, t_rN=5.0, n=5, N=6)
    assert ri_iso == 500.0
    
    ri_tp = calculate_kovats_ri_tp(t_rx=5.0, t_rn=5.0, t_rN=5.0, n=5, N=6)
    assert ri_tp == 500.0


# ==============================================================================
# CATEGORY 4: OPTIMIZATION ENGINE & DISASTER RECOVERY
# ==============================================================================

def test_tpgc_peak_width_boundary_guards():
    """Verifies _compute_tpgc_peak_width against N <= 0, negative t_R, dead_time <= 0."""
    opt = MageOptimizationEngine()
    
    opt.plates = -500
    w = opt._compute_tpgc_peak_width(tr=5.0)
    assert not math.isnan(w)
    assert w > 0.0

    opt.plates = 15000
    w_neg = opt._compute_tpgc_peak_width(tr=-20.0)
    assert not math.isnan(w_neg)
    assert w_neg > 0.0

    w_zero_dt = opt._compute_tpgc_peak_width(tr=5.0, dead_time=0)
    assert not math.isnan(w_zero_dt)
    assert w_zero_dt > 0.0


def test_optimization_co_eluting_isomers_disaster_recovery():
    """Verifies co-eluting isomers (delta RI < 1.0) trigger disaster recovery."""
    opt = MageOptimizationEngine()
    co_eluting_matrix = [
        {"id": "iso_1", "predicted_ri": 1000.0, "status": "COMPUTED"},
        {"id": "iso_2", "predicted_ri": 1000.5, "status": "COMPUTED"}
    ]
    res = opt.optimize_separation(co_eluting_matrix)
    for job in res:
        assert job["optimization_status"] == "LOW_FIDELITY_FALLBACK"
        assert job["disaster_recovery_flag"] is True
        assert job["optimal_ramp_rate"] == 15.0
        assert "Critical Co-Elution" in job["disaster_recovery_reason"]
        assert job["provenance_tag"] == "[E]"


def test_optimization_single_or_empty_job_queue():
    """Verifies optimization bypasses mixture < 2 components gracefully."""
    opt = MageOptimizationEngine()
    res_empty = opt.optimize_separation([])
    assert res_empty == []

    single_matrix = [{"id": "single", "predicted_ri": 1000.0, "status": "COMPUTED"}]
    res_single = opt.optimize_separation(single_matrix)
    assert res_single[0]["optimal_ramp_rate"] == 15.0


def test_optimization_successful_convergence():
    """Verifies SciPy optimizer converges on well-separated mixture."""
    opt = MageOptimizationEngine(target_resolution=1.5)
    matrix = [
        {"id": "mol_A", "predicted_ri": 900.0, "status": "COMPUTED"},
        {"id": "mol_B", "predicted_ri": 1100.0, "status": "COMPUTED"}
    ]
    res = opt.optimize_separation(matrix)
    assert res[0]["optimization_status"] == "OPTIMIZED"
    assert res[0]["disaster_recovery_flag"] is False
    assert 2.0 <= res[0]["optimal_ramp_rate"] <= 40.0


# ==============================================================================
# CATEGORY 5: FRAGMENTER & SMILES / GRAPH PARSING
# ==============================================================================

def test_fragmenter_invalid_smiles_and_none_graph():
    """Verifies MageFragmenter handles invalid SMILES and None graphs gracefully."""
    frag = MageFragmenter()
    
    for bad in [None, "", "INVALID_XYZ_123", 9999]:
        g = frag.graph_from_smiles(bad)
        assert g is None
        
    spec_none = frag.simulate_spectrum(None)
    assert spec_none == {0.0: 0.0}

    spec_empty = frag.simulate_spectrum("")
    assert spec_empty == {0.0: 0.0}


def test_fragmenter_beyer_swinehart_rrkm_bounds():
    """Verifies Beyer-Swinehart state counting rates for E <= E0."""
    frag = MageFragmenter()
    k_zero = frag._beyer_swinehart_rrkm_rate(current_energy_ev=2.0, e0_bde_ev=3.5, n_atoms=12)
    assert k_zero == 0.0

    k_pos = frag._beyer_swinehart_rrkm_rate(current_energy_ev=10.0, e0_bde_ev=3.5, n_atoms=12)
    assert k_pos > 0.0


def test_fragmenter_valid_benzene_spectrum():
    """Verifies RRKM fragmentation of benzene returns realistic MS spectrum."""
    frag = MageFragmenter()
    spec = frag.simulate_spectrum("c1ccccc1")
    assert isinstance(spec, dict)
    assert len(spec) > 0
    assert max(spec.values()) == 100.0


# ==============================================================================
# CATEGORY 6: GRAPH BUILDER & WORKFLOW DAG COMPILER
# ==============================================================================

def test_graph_builder_none_mol_raises():
    """Verifies MageGraphBuilder.build_tensor_graph raises ValueError on None mol."""
    builder = MageGraphBuilder()
    with pytest.raises(ValueError, match="Cannot build graph from NoneType"):
        builder.build_tensor_graph(None)


def test_graph_builder_chlorobenzene_tensor():
    """Verifies tensor graph generation for chlorobenzene."""
    from rdkit import Chem
    builder = MageGraphBuilder()
    mol = Chem.MolFromSmiles("C1=CC=C(C=C1)Cl")
    Chem.SanitizeMol(mol)
    
    graph = builder.build_tensor_graph(mol)
    assert graph["num_nodes"] == 7
    assert graph["x"].shape == (7, 2)
    assert graph["edge_index"].shape[0] == 2
    assert graph["edge_bde"].shape[0] == graph["edge_index"].shape[1]


def test_spend_priority_dag_compiler_non_dict():
    """Verifies build_spend_priority_dag handles non-dict inputs safely."""
    compiler = MageWorkflowDAGCompiler()
    for bad in [None, "str", 123, []]:
        dag = compiler.build_spend_priority_dag(bad)
        assert dag["molecule"] == "Unknown"
        assert dag["total_steps"] == 10
        assert len(dag["spend_priority_nodes"]) == 10
        assert len(dag["dependency_edges"]) == 9


# ==============================================================================
# CATEGORY 7: EXPORTER & VISUALIZATION STRESS
# ==============================================================================

def test_parquet_export_empty_and_null_job_queue(tmp_path):
    """Verifies export_to_parquet handles None or empty job queue."""
    exporter = MageExporter(output_dir=str(tmp_path))
    
    import pandas as pd
    res_none = exporter.export_to_parquet(None)
    assert isinstance(res_none, pd.DataFrame) and res_none.empty

    res_empty = exporter.export_to_parquet([])
    assert isinstance(res_empty, pd.DataFrame) and res_empty.empty


def test_interactive_chromatogram_rendering(tmp_path):
    """Verifies building HTML chromatogram with valid and missing fields."""
    exporter = MageExporter(output_dir=str(tmp_path))
    jobs = [
        {"smiles": "c1ccccc1", "status": "COMPUTED", "predicted_tr": 3.5, "mw": 78.11},
        {"smiles": "CC(=O)O", "status": "CACHED", "estimated_rt": 2.1, "mw": 60.05}
    ]
    html_path = exporter.build_interactive_chromatogram(jobs, filename="test_chrom.html")
    assert os.path.exists(html_path)
    assert os.path.getsize(html_path) > 1000


def test_head_to_tail_ms_plot_dict_spectra(tmp_path):
    """Verifies head-to-tail MS plot rendering with dict inputs."""
    exporter = MageExporter(output_dir=str(tmp_path))
    
    exp_spec = {78.0: 100.0, 51.0: 30.0, 39.0: 15.0}
    pred_spec = {78.0: 100.0, 51.0: 25.0, 39.0: 20.0}
    
    out_path = exporter.build_head_to_tail_ms_plot(exp_spec, pred_spec, filename="test_ms.html")
    assert os.path.exists(out_path)


def test_head_to_tail_ms_plot_empty_list_unhandled_exception(tmp_path):
    """
    ADVERSARIAL STRESS TEST: Passing empty list [] as spectrum input.
    Exposes bug in cochem_mage_export.py:123 (normalize_spectrum assumes 2D array for lists).
    Asserting smooth recovery or expected handling.
    """
    exporter = MageExporter(output_dir=str(tmp_path))
    # Passing empty list [] as pred_spectrum
    out_path = exporter.build_head_to_tail_ms_plot({}, [], filename="empty_ms.html")
    assert os.path.exists(out_path)


# ==============================================================================
# CATEGORY 8: ORCHESTRATOR & CONCURRENT HDF5 SERIALIZATION
# ==============================================================================

def test_orchestrator_uninitialized_simulation_raises():
    """Verifies running simulation on uninitialized orchestrator raises RuntimeError."""
    orch = MAGEOrchestrator()
    with pytest.raises(RuntimeError, match="must be initialized"):
        orch.run_simulation("rrkm", {"molecule": "benzene"})


def test_orchestrator_concurrent_hdf5_writes(tmp_path):
    """
    STRESS TEST: 10 concurrent threads exporting HDF5 data simultaneously.
    Verifies microsecond timestamp + UUID group key prevents collisions (ValueError).
    """
    orch = MAGEOrchestrator()
    orch.initialize()
    
    h5_file = str(tmp_path / "concurrent_stress.h5")
    errors = []
    
    def worker_thread(thread_id):
        try:
            for i in range(5):
                data = {
                    "LAM_TRIGGER_REQUIRED": False,
                    "results": [
                        {"id": f"thread_{thread_id}_item_{i}", "mw": 100 + i, "status": "COMPUTED"}
                    ]
                }
                orch.export_to_h5(data, h5_path=h5_file)
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")

    threads = [threading.Thread(target=worker_thread, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent HDF5 export produced errors: {errors}"
    assert os.path.exists(h5_file)


def test_orchestrator_full_simulation_flow(tmp_path):
    """Verifies end-to-end simulation flow in MAGEOrchestrator."""
    orch = MAGEOrchestrator()
    orch.initialize()
    
    res_rrkm = orch.run_simulation("rrkm", {"smiles": "benzene"})
    assert res_rrkm["status"] == "COMPLETED"
    assert "spectrum" in res_rrkm

    res_chrom = orch.run_simulation("chromatography", {})
    assert res_chrom["status"] == "COMPLETED"
    assert len(res_chrom["jobs"]) > 0

    report_html = orch.generate_report(output_dir=str(tmp_path / "reports"))
    assert os.path.exists(report_html)


# ==============================================================================
# CATEGORY 9: TELEMETRY & CACHE INFRASTRUCTURE
# ==============================================================================

def test_telemetry_bridge_thread_safety():
    """Verifies thread-safe telemetry state updates."""
    bridge = MageTelemetryBridge()
    bridge.update_state("RUNNING", 50.0, "Test operation", isomer="Benzene", rs=1.5, provenance_tag="[D]")
    state = bridge.current_state
    assert state["status"] == "RUNNING"
    assert state["progress_percent"] == 50.0
    assert state["active_isomer"] == "Benzene"
    assert state["provenance_tag"] == "[D]"


def test_descriptor_cache_sqlite(tmp_path):
    """Verifies MageDescriptorCache SQLite operations and invalid SMILES handling."""
    db_file = str(tmp_path / "test_cache.db")
    cache = MageDescriptorCache(db_path=db_file)
    
    from rdkit import Chem
    mol = Chem.MolFromSmiles("c1ccccc1")
    props1 = cache.get_or_compute("c1ccccc1", mol)
    assert props1["status"] == "COMPUTED"
    assert props1["mw"] > 0

    props2 = cache.get_or_compute("c1ccccc1", mol)
    assert props2["status"] == "CACHED"

    bad_batch = [{"smiles": "BAD", "rdkit_mol": None, "status": "SANITIZED"}]
    res_batch = cache.process_batch(bad_batch)
    assert res_batch[0]["status"] == "FAILED_PHYSICS"


# ==============================================================================
# CATEGORY 10: CONTAINER / TYPE GUARD REGRESSION TESTS
# ==============================================================================

def test_parquet_export_type_and_element_guards():
    """Verifies export_to_parquet handles non-sequence inputs and non-dict elements."""
    exporter = MageExporter()
    assert exporter.export_to_parquet("not_a_list").empty
    assert exporter.export_to_parquet(None).empty
    res = exporter.export_to_parquet([None, "invalid", {"id": "j1", "predicted_ri": 900.0, "status": "COMPUTED"}])
    assert os.path.exists(res)


def test_optimization_active_matrix_none_element_guard():
    """Verifies optimize_separation handles non-dict/None elements in active_matrix."""
    opt = MageOptimizationEngine()
    matrix = [
        {"id": "a", "predicted_ri": 900.0, "status": "COMPUTED"},
        {"id": "b", "predicted_ri": 1000.0, "status": "CACHED"},
        None,
        "invalid"
    ]
    res = opt.optimize_separation(matrix)
    assert matrix[0]["optimization_status"] == "OPTIMIZED"
    assert matrix[0]["optimal_ramp_rate"] > 0


def test_run_simulation_none_input_data_guard():
    """Verifies run_simulation handles non-dict / None input_data safely."""
    m = MAGEOrchestrator()
    m.initialize()
    res = m.run_simulation("rrkm", None)
    assert res.get("status") == "COMPLETED"


def test_get_full_isotope_cluster_none_formula_dict_guard():
    """Verifies get_full_isotope_cluster handles non-dict / None formula_dict safely."""
    from mage_isotope import HalogenIsotopeGenerator
    g = HalogenIsotopeGenerator()
    res = g.get_full_isotope_cluster(None)
    assert isinstance(res, dict)
    assert res.get(0.0) == 100.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
