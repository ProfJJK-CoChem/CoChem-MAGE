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
