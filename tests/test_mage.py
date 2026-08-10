import os
import pytest
import numpy as np
from rdkit import Chem

from cochem_mage_main import MAGEOrchestrator
from cochem_mage_sim import MageChromatographySim
from cochem_mage_opt import MageOptimizationEngine
from cochem_mage_export import MageExporter
from mage_fragmenter import MageFragmenter
from mage_graph_setup import MageGraphBuilder
from mage_isotope import HalogenIsotopeGenerator
from mage_column_registry import MageIngestor, NistApiBridge

def test_orchestrator_init():
    orchestrator = MAGEOrchestrator()
    orchestrator.initialize()
    assert orchestrator.is_initialized is True

def test_orchestrator_sim():
    orchestrator = MAGEOrchestrator()
    orchestrator.initialize()
    res = orchestrator.run_simulation("chrom", {})
    assert res["status"] == "COMPLETED"
    assert "jobs" in res

def test_chromatography_sim():
    col_config = {"length_m": 30.0, "stationary_phase": "5% phenyl"}
    sim = MageChromatographySim(col_config)
    jobs = [
        {"id": "mol_0", "smiles": "c1ccccc1", "mw": 78.11, "logp": 2.13, "tpsa": 0.0, "status": "COMPUTED"}
    ]
    sim_jobs = sim.simulate_retention(jobs, temperature_ramp_rate=10.0)
    assert sim_jobs[0]["predicted_ri"] > 0
    assert sim_jobs[0]["predicted_tr"] > 0

    t_axis, trace = sim.build_chromatogram(sim_jobs)
    assert len(trace) == 10000

def test_optimization_engine():
    optimizer = MageOptimizationEngine(target_resolution=1.5)
    matrix = [
        {"id": "A", "predicted_ri": 900.0, "status": "COMPUTED"},
        {"id": "B", "predicted_ri": 1050.0, "status": "COMPUTED"}
    ]
    res = optimizer.optimize_separation(matrix)
    assert res[0]["optimal_ramp_rate"] > 0.0

def test_isotope_generator():
    iso_gen = HalogenIsotopeGenerator()
    cluster = iso_gen.get_isotope_cluster(1, 0) # 1 Cl
    assert len(cluster) >= 2

def test_graph_builder():
    builder = MageGraphBuilder()
    mol = Chem.MolFromSmiles("c1ccccc1")
    graph = builder.build_tensor_graph(mol)
    assert graph["num_nodes"] == 6

def test_fragmenter():
    builder = MageGraphBuilder()
    mol = Chem.MolFromSmiles("CC")
    graph = builder.build_tensor_graph(mol)
    fragmenter = MageFragmenter(impact_energy_ev=70.0)
    spectrum = fragmenter.simulate_spectrum(graph, num_trajectories=10)
    assert len(spectrum) > 0

def test_abraham_stationary_phase_partitioning():
    sim_db5 = MageChromatographySim({"length_m": 30.0, "stationary_phase": "5% phenyl"})
    sim_wax = MageChromatographySim({"length_m": 30.0, "stationary_phase": "DB-Wax"})
    
    polar_job = {"id": "phenol", "smiles": "c1cc(O)ccc1", "mw": 94.11, "logp": 1.46, "tpsa": 20.2, "status": "COMPUTED"}
    
    res_db5 = sim_db5.simulate_retention([polar_job.copy()])
    res_wax = sim_wax.simulate_retention([polar_job.copy()])
    
    # Polar compound must have higher retention index on polar DB-Wax phase than non-polar DB-5 phase
    assert res_wax[0]["predicted_ri"] > res_db5[0]["predicted_ri"]

def test_van_deemter_and_kovats_ri():
    from cochem_mage_sim import calculate_kovats_ri_isothermal, calculate_kovats_ri_tp
    sim = MageChromatographySim({"length_m": 30.0, "stationary_phase": "DB-5"})
    hetp, u, tag = sim.compute_van_deemter_hetp(u_cm_s=25.0)
    assert hetp > 0.0
    assert u == 25.0
    assert tag == '[E]'

    ri_iso = calculate_kovats_ri_isothermal(t_rx=8.5, t_rn=6.0, t_rN=10.0, n=8, N=9, t_m=1.5)
    ri_tp = calculate_kovats_ri_tp(t_rx=8.5, t_rn=6.0, t_rN=10.0, n=8, N=9)
    assert 800.0 <= ri_iso <= 900.0
    assert 800.0 <= ri_tp <= 900.0

def test_mage_exporter_head_to_tail_and_parquet(tmp_path):
    exporter = MageExporter(output_dir=str(tmp_path))
    exp_spec = {50: 10.0, 100: 100.0, 150: 45.0}
    pred_spec = {50: 12.0, 100: 95.0, 150: 50.0}
    html_path = exporter.build_head_to_tail_ms_plot(exp_spec, pred_spec)
    assert os.path.exists(html_path)

    queue = [
        {"id": "mol_0", "smiles": "c1ccccc1", "mw": 78.11, "logp": 2.13, "tpsa": 0.0, "predicted_ri": 655.0, "predicted_tr": 5.2, "status": "COMPUTED"}
    ]
    pq_path = exporter.export_to_parquet(queue)
    assert os.path.exists(pq_path)

def test_v4_config_defaults():
    from cochem_mage_config import MAGEConfig
    cfg = MAGEConfig()
    defaults = cfg.get("defaults", {})
    assert defaults.get("t1_method") == "r2SCAN-3c"
    assert defaults.get("t2_composite") == "junChS"
    assert defaults.get("t3_geometry") == "CCSD(T)-F12"
    assert "B3LYP" not in str(cfg.config)
    assert cfg.get("version") == "0.4.0"
    assert cfg.get("product_class") == "PRODUCT_A"
    assert cfg.get("tier_level") == "T1-30min"
    assert cfg.get("performance", {}).get("node_scheduler_delegated") is True
    spend_seq = cfg.get("spend_priority", [])
    assert len(spend_seq) == 10
    assert spend_seq[0] == "intermolecular_geometry"
    assert spend_seq[-1] == "binding_energy_d0"

def test_product_class_decision_tree():
    from cochem_mage_sim import determine_product_class
    # Product B (measured parent present)
    res_b = determine_product_class({"measured_parent_isotopologue": "13C1-benzene"})
    assert res_b["product_class"] == "PRODUCT_B"
    assert res_b["target_accuracy_window"] == "±0.03% to ±0.1%"
    assert res_b["provenance_tag"] == "[D]"

    # Product C (difference calculation)
    res_c = determine_product_class({"is_difference_calculation": True})
    assert res_c["product_class"] == "PRODUCT_C"
    assert res_c["provenance_tag"] == "[D]"

    # Product A (default de novo)
    res_a = determine_product_class({})
    assert res_a["product_class"] == "PRODUCT_A"
    assert res_a["target_accuracy_window"] == "±0.3% to ±0.5%"
    assert res_a["provenance_tag"] == "[E]"

def test_spend_priority_dag_compiler():
    from mage_graph_setup import MageWorkflowDAGCompiler
    compiler = MageWorkflowDAGCompiler()
    dag = compiler.build_spend_priority_dag({"smiles": "c1ccccc1"})
    assert dag["total_steps"] == 10
    nodes = dag["spend_priority_nodes"]
    assert nodes[0]["step_id"] == "intermolecular_geometry"
    assert nodes[0]["tier_budget"] == "T1-30min"
    assert nodes[1]["step_id"] == "delta_b_vib"
    assert nodes[9]["step_id"] == "binding_energy_d0"
    assert len(dag["dependency_edges"]) == 9

def test_phase_grounded_golay_hetp():
    sim = MageChromatographySim({"length_m": 30.0, "stationary_phase": "DB-5"})
    # Empirical fallback
    hetp_emp, u, tag_emp = sim.compute_van_deemter_hetp()
    assert tag_emp == "[E]"
    assert hetp_emp > 0.0

    # Phase grounded Golay HETP
    phase_params = {
        "film_thickness_um": 0.25,
        "inner_diameter_mm": 0.25,
        "retention_factor_k": 4.5,
        "binary_diffusion_m2_s": 1e-5,
        "stationary_diffusion_m2_s": 1e-9
    }
    hetp_phys, u, tag_phys = sim.compute_van_deemter_hetp(station_phase_params=phase_params)
    assert tag_phys == "[D]"
    assert hetp_phys > 0.0

def test_provenance_tagging_across_modules(tmp_path):
    from cochem_mage_telemetry import MageTelemetryBridge
    from cochem_mage_main import MAGEOrchestrator

    # Telemetry
    telemetry = MageTelemetryBridge()
    assert telemetry.current_state["provenance_tag"] == "[D]"
    telemetry.update_state("RUNNING", 50.0, "Testing", provenance_tag="[E]")
    assert telemetry.current_state["provenance_tag"] == "[E]"

    # HDF5 Export
    orchestrator = MAGEOrchestrator()
    h5_file = str(tmp_path / "test_state.h5")
    data = {
        "provenance_tag": "[D]",
        "results": [{"id": "mol_0", "status": "COMPUTED", "provenance_tag": "[D]"}]
    }
    res_path = orchestrator.export_to_h5(data, h5_path=h5_file)
    assert os.path.exists(res_path)


def test_config_sanitization_and_saving(tmp_path):
    from cochem_mage_config import MAGEConfig
    cfg_path = str(tmp_path / "nested_dir" / "custom_config.json")
    cfg = MAGEConfig(config_file=cfg_path)
    
    # Verify defaults sanitized on init
    assert cfg.get("defaults", {}).get("t1_method") == "r2SCAN-3c"
    
    # Test updating with prohibited legacy functional/basis
    cfg.update_from_dict({"defaults": {"t1_method": "B3LYP", "default_basis": "6-31G*"}})
    
    # Verify sanitization stripped B3LYP and set v4 defaults
    assert cfg.get("defaults", {}).get("t1_method") == "r2SCAN-3c"
    assert "default_basis" not in cfg.get("defaults", {})
    assert os.path.exists(cfg_path)

def test_hdf5_consecutive_export_no_collision(tmp_path):
    from cochem_mage_main import MAGEOrchestrator
    orchestrator = MAGEOrchestrator()
    h5_file = str(tmp_path / "consecutive_test.h5")
    
    data1 = {"results": [{"id": "mol_1", "status": "COMPUTED"}]}
    data2 = {"results": [{"id": "mol_2", "status": "COMPUTED"}]}
    
    res1 = orchestrator.export_to_h5(data1, h5_path=h5_file)
    res2 = orchestrator.export_to_h5(data2, h5_path=h5_file)
    
    assert os.path.exists(res1)
    assert os.path.exists(res2)

def test_determine_product_class_robustness_and_precedence():
    from cochem_mage_sim import determine_product_class
    
    # Non-dict inputs
    assert determine_product_class(None)["product_class"] == "PRODUCT_A"
    assert determine_product_class("invalid_string")["product_class"] == "PRODUCT_A"
    assert determine_product_class([1, 2, 3])["product_class"] == "PRODUCT_A"
    
    # Difference calc WITH parent experimental data -> must be PRODUCT_C
    res_diff_parent = determine_product_class({
        "is_difference_calculation": True,
        "measured_parent_isotopologue": "13C1-benzene"
    })
    assert res_diff_parent["product_class"] == "PRODUCT_C"
    assert res_diff_parent["recommended_tier"] == "T1-1h"

def test_van_deemter_hetp_boundary_guards():
    from cochem_mage_sim import MageChromatographySim
    sim = MageChromatographySim({"length_m": 30.0, "stationary_phase": "DB-5"})
    
    # Explicit None parameters
    params_none = {
        "film_thickness_um": None,
        "inner_diameter_mm": None,
        "retention_factor_k": None,
        "binary_diffusion_m2_s": None,
        "stationary_diffusion_m2_s": None
    }
    hetp_none, u_none, tag_none = sim.compute_van_deemter_hetp(station_phase_params=params_none)
    assert hetp_none > 0.0
    assert tag_none == "[D]"
    
    # Singular k = -1.0
    params_k_neg1 = {"retention_factor_k": -1.0}
    hetp_k, u_k, _ = sim.compute_van_deemter_hetp(station_phase_params=params_k_neg1)
    assert hetp_k > 0.0
    
    # Negative u
    hetp_u_neg, u_clamped, _ = sim.compute_van_deemter_hetp(u_cm_s=-50.0)
    assert hetp_u_neg > 0.0
    assert u_clamped > 0.0

def test_tpgc_peak_width_boundary_guards():
    import numpy as np
    from cochem_mage_opt import MageOptimizationEngine
    opt = MageOptimizationEngine()
    
    # Negative tr
    w_neg_tr = opt._compute_tpgc_peak_width(tr=-50.0, alpha=0.05)
    assert not np.isnan(w_neg_tr)
    assert w_neg_tr > 0.0
    
    # Plates <= 0
    opt_bad_plates = MageOptimizationEngine(theoretical_plates=-100)
    w_bad_plates = opt_bad_plates._compute_tpgc_peak_width(tr=5.0)
    assert not np.isnan(w_bad_plates)
    assert w_bad_plates > 0.0
    
    # dead_time <= 0
    w_zero_dt = opt._compute_tpgc_peak_width(tr=5.0, dead_time=0.0)
    assert not np.isnan(w_zero_dt)
    assert w_zero_dt > 0.0

def test_parquet_and_dag_compiler_input_guards():
    from cochem_mage_export import MageExporter
    exporter = MageExporter()
    df_empty = exporter.export_to_parquet(None)
    assert hasattr(df_empty, "empty")
    
    from mage_graph_setup import MageWorkflowDAGCompiler
    compiler = MageWorkflowDAGCompiler()
    dag_none = compiler.build_spend_priority_dag(None)
    assert dag_none["total_steps"] == 10
    assert dag_none["molecule"] == "Unknown"

def test_fragmenter_invalid_smiles_and_none_graph():
    from mage_fragmenter import MageFragmenter
    frag = MageFragmenter()
    
    # Invalid SMILES returns None safely
    g_invalid = frag.graph_from_smiles("INVALID_SMILES_STRING")
    assert g_invalid is None
    
    # None graph data returns empty spectrum dict
    spec_none = frag.simulate_spectrum(None)
    assert spec_none == {0.0: 0.0}
    
    # Invalid SMILES passed to simulate_spectrum returns empty spectrum dict
    spec_invalid = frag.simulate_spectrum("INVALID_SMILES_STRING")
    assert spec_invalid == {0.0: 0.0}



