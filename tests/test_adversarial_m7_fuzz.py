"""
CoChem-MAGE Comprehensive Adversarial Fuzzing and Stress Harness.
Author: EMPIRICAL CHALLENGER
Target: CoChem-MAGE (Gate M7 Iteration 4 Verification)

This module executes deep, systematic adversarial fuzz testing across all 12 CoChem-MAGE modules.
It evaluates handling of extreme boundary conditions, type corruptions, null lists, massive job queues,
and verifies zero unhandled crashes or state corruptions.
"""

import os
import sys
import math
import shutil
import tempfile
import threading
import numpy as np
import pytest
from pathlib import Path

# Add CoChem-MAGE directory to path
MAGE_DIR = Path(__file__).parent.parent.resolve()
if str(MAGE_DIR) not in sys.path:
    sys.path.insert(0, str(MAGE_DIR))

from cochem_mage_config import MAGEConfig
from cochem_mage_main import MAGEOrchestrator
from cochem_mage_sim import (
    MageChromatographySim,
    determine_product_class,
    calculate_kovats_ri_isothermal,
    calculate_kovats_ri_tp
)
from cochem_mage_opt import MageOptimizationEngine
from cochem_mage_export import MageExporter
from mage_fragmenter import MageFragmenter
from mage_graph_setup import MageGraphBuilder, MageWorkflowDAGCompiler
from mage_isotope import HalogenIsotopeGenerator
from cochem_mage_telemetry import MageTelemetryBridge
from cochem_mage_cache import MageDescriptorCache
from mage_column_registry import MageIngestor, NistApiBridge


# ==============================================================================
# SECTION 1: EXPORT MODULE ADVERSARIAL FUZZING (cochem_mage_export)
# ==============================================================================

def test_exporter_interactive_chromatogram_fuzz(tmp_path):
    exporter = MageExporter(output_dir=str(tmp_path))
    
    # 1. Invalid job queues
    for bad_queue in [None, [], [None, 123, "invalid", []], [{"status": "COMPUTED"}]]:
        out = exporter.build_interactive_chromatogram(bad_queue, filename="test_bad.html")
        assert os.path.exists(out)

    # 2. Corrupt numeric values in job items
    corrupt_queue = [
        {
            "status": "COMPUTED",
            "smiles": "c1ccccc1",
            "predicted_tr": None,
            "estimated_rt": None,
            "logp": -999.0,
            "mw": 0.0,
            "tpsa": -50.0,
            "peak_intensity": 0.0
        },
        {
            "status": "CACHED",
            "smiles": None,
            "predicted_tr": 5.5,
            "chemical_class": None,
            "ccs": None,
            "logp": 2.5
        }
    ]
    out_corrupt = exporter.build_interactive_chromatogram(corrupt_queue, filename="test_corrupt.html")
    assert os.path.exists(out_corrupt)


def test_exporter_scribe_payload_fuzz(tmp_path):
    exporter = MageExporter(output_dir=str(tmp_path))
    
    for bad_profile in [None, "not a dict", 12345, []]:
        out = exporter.export_scribe_payload(None, bad_profile, filename="bad_scribe.json")
        assert os.path.exists(out)

    dirty_queue = [
        {"status": "COMPUTED", "smiles": None, "mw": "not_a_float", "provenance_tag": None},
        {"status": "CACHED", "estimated_rt": 4.2, "status_other": True}
    ]
    out_dirty = exporter.export_scribe_payload(dirty_queue, {"instrument": "test"}, filename="dirty_scribe.json")
    assert os.path.exists(out_dirty)


def test_exporter_head_to_tail_ms_plot_fuzz(tmp_path):
    exporter = MageExporter(output_dir=str(tmp_path))

    # Test empty, non-dict, non-list, corrupt spectrum formats
    bad_inputs = [
        (None, None),
        ({}, {}),
        ([], []),
        ("not_a_spectrum", 12345),
        ({"invalid_mz": "invalid_int"}, []),
        ([(100.0, 50.0), (200.0, 100.0)], {"50.0": 10.0, "150.0": 80.0}),
        ([None, (100.0, 50.0)], None)
    ]
    
    for idx, (exp_spec, pred_spec) in enumerate(bad_inputs):
        out = exporter.build_head_to_tail_ms_plot(exp_spec, pred_spec, filename=f"ms_plot_{idx}.html")
        assert os.path.exists(out)


def test_exporter_parquet_export_fuzz(tmp_path):
    exporter = MageExporter(output_dir=str(tmp_path))

    for bad_queue in [None, "invalid", 12345, [None, 456, "str"]]:
        res = exporter.export_to_parquet(bad_queue, filename="bad_parquet.parquet")
        assert res is not None

    corrupt_records = [
        {
            "id": None,
            "smiles": None,
            "chemical_class": None,
            "mw": None,
            "logp": None,
            "tpsa": None,
            "predicted_ri": None,
            "predicted_tr": None,
            "status": None,
            "provenance_tag": None
        }
    ]
    res_corrupt = exporter.export_to_parquet(corrupt_records, filename="corrupt_parquet.parquet")
    assert os.path.exists(res_corrupt)


# ==============================================================================
# SECTION 2: OPTIMIZATION MODULE ADVERSARIAL FUZZING (cochem_mage_opt)
# ==============================================================================

def test_opt_tpgc_peak_width_extreme_fuzz():
    opt = MageOptimizationEngine()
    
    extreme_cases = [
        (None, None, None, None),
        (-10.0, -5.0, -1.0, {"length_m": -100.0, "hetp_mm": -0.05}),
        (100.0, 100.0, 5.0, {"length_m": 0.0, "hetp_mm": 0.0}),
        (0.0, 0.0, 0.0, {}),
        (1e6, 1.5, 0.05, {"length_m": 100.0, "hetp_mm": 0.001})
    ]
    for tr, dt, alpha, params in extreme_cases:
        w = opt._compute_tpgc_peak_width(tr, dead_time=dt, alpha=alpha, station_phase_params=params)
        assert not math.isnan(w)
        assert not math.isinf(w)
        assert w > 0.0


def test_opt_objective_function_extreme_fuzz():
    opt = MageOptimizationEngine()
    
    # Negative rates, massive rates, empty RI arrays
    assert opt._objective_function(0.5, np.array([100.0, 200.0])) == 9999.0
    assert opt._objective_function(50.0, np.array([100.0, 200.0]), max_ramp=40.0) == 9999.0
    
    obj_empty = opt._objective_function(15.0, np.array([]))
    assert not math.isnan(obj_empty)


def test_opt_optimize_separation_disaster_scenarios():
    opt = MageOptimizationEngine()
    
    # 1. Null / single element active matrix
    for bad_mat in [None, [], [None], [{"status": "COMPUTED"}]]:
        res = opt.optimize_separation(bad_mat)
        assert res == bad_mat or len(res) == len(bad_mat or [])

    # 2. Overlapping isomers with identical RI (ΔRI < 1.0)
    co_eluting = [
        {"id": "iso_1", "predicted_ri": 1000.0, "status": "COMPUTED"},
        {"id": "iso_2", "predicted_ri": 1000.2, "status": "COMPUTED"}
    ]
    res_co = opt.optimize_separation(co_eluting)
    assert res_co[0]["optimization_status"] == "LOW_FIDELITY_FALLBACK"
    assert res_co[0]["disaster_recovery_flag"] is True

    # 3. Mixture where separation cannot reach Rs >= 0.6
    crammed_mixture = [
        {"id": f"mol_{i}", "predicted_ri": 1000.0 + i * 1.5, "status": "COMPUTED"}
        for i in range(20)
    ]
    res_crammed = opt.optimize_separation(crammed_mixture)
    assert res_crammed[0]["optimization_status"] in ["LOW_FIDELITY_FALLBACK", "OPTIMIZED"]


# ==============================================================================
# SECTION 3: CHROMATOGRAPHY SIMULATION ADVERSARIAL FUZZING (cochem_mage_sim)
# ==============================================================================

def test_sim_determine_product_class_precedence_and_types():
    # Test precedence: Difference calc override over measured parent
    combo = {
        "is_difference_calculation": True,
        "measured_parent_isotopologue": "benzene"
    }
    res = determine_product_class(combo)
    assert res["product_class"] == "PRODUCT_C"
    assert res["recommended_tier"] == "T1-1h"

    # Non-dict type safety
    for non_dict in [None, 100, "str", [1, 2], True]:
        r = determine_product_class(non_dict)
        assert r["product_class"] == "PRODUCT_A"


def test_sim_chi_indices_invalid_inputs():
    sim = MageChromatographySim({"length_m": 30.0})
    for bad_input in [None, "", "INVALID_SMILES_STRING", 12345, 3.14]:
        c0, c1 = sim._compute_chi_indices(bad_input)
        assert c0 == 0.0
        assert c1 == 0.0


def test_sim_van_deemter_hetp_extreme_params():
    sim = MageChromatographySim({"length_m": 30.0})
    
    # 1. Zero/Negative velocity
    h1, u1, tag1 = sim.compute_van_deemter_hetp(u_cm_s=0.0)
    assert u1 == 1e-3
    assert h1 > 0.0

    # 2. Explicit None keys in station_phase_params
    none_dict = {
        "film_thickness_um": None,
        "inner_diameter_mm": None,
        "retention_factor_k": None,
        "binary_diffusion_m2_s": None,
        "stationary_diffusion_m2_s": None
    }
    h2, u2, tag2 = sim.compute_van_deemter_hetp(station_phase_params=none_dict)
    assert not math.isnan(h2)
    assert h2 > 0.0
    assert tag2 == "[D]"


def test_sim_massive_job_queue_performance():
    sim = MageChromatographySim({"length_m": 30.0})
    
    massive_jobs = [
        {
            "id": f"mol_{i}",
            "smiles": "c1ccccc1" if i % 2 == 0 else "CC(=O)O",
            "mw": 78.11 + i * 0.1,
            "logp": 2.0,
            "tpsa": 0.0,
            "status": "COMPUTED"
        }
        for i in range(500)
    ]
    sim_jobs = sim.simulate_retention(massive_jobs, temperature_ramp_rate=15.0)
    assert len(sim_jobs) == 500
    assert all("predicted_ri" in j for j in sim_jobs)

    t_axis, trace = sim.build_chromatogram(sim_jobs[:50], t_max=30.0, resolution=2000)
    assert len(t_axis) == 2000
    assert len(trace) == 2000


def test_kovats_ri_math_functions_boundary_cases():
    # Isothermal Kovats: zero denominator fallback
    ri_iso = calculate_kovats_ri_isothermal(t_rx=5.0, t_rn=5.0, t_rN=5.0, n=5, N=6, t_m=1.5)
    assert ri_iso == 500.0

    # Temperature-programmed Kovats: zero denominator fallback
    ri_tp = calculate_kovats_ri_tp(t_rx=5.0, t_rn=5.0, t_rN=5.0, n=5, N=6)
    assert ri_tp == 500.0


# ==============================================================================
# SECTION 4: ISOTOPE PATTERN ADVERSARIAL FUZZING (mage_isotope)
# ==============================================================================

def test_halogen_isotope_generator_fuzz():
    gen = HalogenIsotopeGenerator()
    
    # 1. Non-dict formula input
    for bad in [None, [], 12345, "C6H6"]:
        cluster = gen.get_full_isotope_cluster(bad)
        assert isinstance(cluster, dict)
        assert 0.0 in cluster

    # 2. Formula with zero or negative counts
    cluster_zero = gen.get_full_isotope_cluster({'C': 0, 'H': 0, 'Cl': 0, 'Br': 0})
    assert cluster_zero == {0.0: 100.0}

    # 3. High atom counts (e.g. C50 H100 Cl10 Br5)
    large_formula = {'C': 50, 'H': 100, 'Cl': 10, 'Br': 5}
    cluster_large = gen.get_full_isotope_cluster(large_formula)
    assert isinstance(cluster_large, dict)
    assert len(cluster_large) > 0


# ==============================================================================
# SECTION 5: FRAGMENTER & RRKM ADVERSARIAL FUZZING (mage_fragmenter)
# ==============================================================================

def test_fragmenter_invalid_smiles_and_graphs():
    frag = MageFragmenter()
    
    # Invalid SMILES strings
    for bad_smi in [None, "", "NOT_A_SMILES", 12345, "[invalid_bracket"]:
        g = frag.graph_from_smiles(bad_smi)
        assert g is None

    # Null / empty graph spectrum simulation
    for bad_graph in [None, "", {}, "NOT_A_SMILES"]:
        spec = frag.simulate_spectrum(bad_graph)
        assert spec == {0.0: 0.0}


def test_fragmenter_rrkm_beyer_swinehart_rate_bounds():
    frag = MageFragmenter()

    # 1. Energy below BDE threshold -> rate = 0.0
    r1 = frag._beyer_swinehart_rrkm_rate(current_energy_ev=2.0, e0_bde_ev=3.5, n_atoms=10)
    assert r1 == 0.0

    # 2. Energy above BDE threshold -> positive rate
    r2 = frag._beyer_swinehart_rrkm_rate(current_energy_ev=10.0, e0_bde_ev=3.5, n_atoms=10)
    assert r2 > 0.0

    # 3. Extreme high energy
    r3 = frag._beyer_swinehart_rrkm_rate(current_energy_ev=70.0, e0_bde_ev=2.8, n_atoms=20)
    assert not math.isnan(r3)
    assert not math.isinf(r3)


# ==============================================================================
# SECTION 6: GRAPH BUILDER & DAG COMPILER (mage_graph_setup)
# ==============================================================================

def test_graph_builder_none_mol_raises():
    builder = MageGraphBuilder()
    with pytest.raises(ValueError, match="Cannot build graph from NoneType"):
        builder.build_tensor_graph(None)


def test_dag_compiler_non_dict_and_provenance_tags():
    compiler = MageWorkflowDAGCompiler()
    
    for bad_info in [None, "str", 12345, []]:
        dag = compiler.build_spend_priority_dag(bad_info)
        assert isinstance(dag, dict)
        assert dag["total_steps"] == 10
        assert len(dag["spend_priority_nodes"]) == 10
        # Verify first 7 nodes have [D] tag, last 3 have [E] tag
        for n in dag["spend_priority_nodes"][:7]:
            assert n["provenance_tag"] == "[D]"
        for n in dag["spend_priority_nodes"][7:]:
            assert n["provenance_tag"] == "[E]"


# ==============================================================================
# SECTION 7: MAIN ORCHESTRATOR & TELEMETRY (cochem_mage_main, cochem_mage_telemetry)
# ==============================================================================

def test_orchestrator_uninitialized_and_h5_export(tmp_path):
    orch = MAGEOrchestrator()
    
    # 1. Uninitialized simulation raises RuntimeError
    with pytest.raises(RuntimeError, match="must be initialized"):
        orch.run_simulation("rrkm", {})

    # Initialize
    orch.initialize()

    # 2. HDF5 Export with non-dict input
    h5_file = tmp_path / "cochem_test_state.h5"
    res = orch.export_to_h5("not_a_dict", h5_path=str(h5_file))
    assert res == str(h5_file)

    # 3. HDF5 Export with valid data
    valid_data = {
        "symmetry_group": "C2v",
        "provenance_tag": "[D]",
        "results": [
            {"smiles": "c1ccccc1", "mw": 78.11, "status": "COMPUTED"}
        ]
    }
    res_valid = orch.export_to_h5(valid_data, h5_path=str(h5_file))
    assert os.path.exists(res_valid)


def test_telemetry_bridge_thread_safety_and_none_values():
    bridge = MageTelemetryBridge()

    # Concurrent state updates
    def worker(worker_id):
        for i in range(20):
            bridge.update_state(
                status=f"RUNNING_{worker_id}",
                progress=i * 5.0,
                operation=f"Op {i}",
                isomer=f"Isomer_{worker_id}",
                rs=1.5,
                error=None
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify final state is valid
    state = bridge.current_state
    assert isinstance(state["progress_percent"], float)
    assert state["status"].startswith("RUNNING_")


# ==============================================================================
# SECTION 8: DESCRIPTOR CACHE & COLUMN REGISTRY (cochem_mage_cache, mage_column_registry)
# ==============================================================================

def test_descriptor_cache_sqlite_operations(tmp_path):
    db_file = tmp_path / "test_descriptors.db"
    cache = MageDescriptorCache(db_path=str(db_file))

    # Process batch with corrupt jobs
    corrupt_batch = [
        {"smiles": "c1ccccc1", "rdkit_mol": None, "status": "SANITIZED"},
        {"smiles": "INVALID", "rdkit_mol": None, "status": "FAILED_PHYSICS"}
    ]
    res = cache.process_batch(corrupt_batch)
    assert len(res) == 2
    assert res[0]["status"] == "FAILED_PHYSICS"


def test_ingestor_nist_api_failover(tmp_path):
    sys_cfg = tmp_path / "sys_config.json"
    reg_cfg = tmp_path / "col_registry.json"
    
    ingestor = MageIngestor(sys_config_path=str(sys_cfg), registry_path=str(reg_cfg))
    
    # Sanitize molecule list with valid and invalid SMILES
    smiles_list = ["c1ccccc1", "INVALID_SMILES", "CC(=O)O"]
    sanitized = ingestor.sanitize_molecule_queue(smiles_list)
    assert len(sanitized) == 2
    assert sanitized[0]["status"] == "SANITIZED"
