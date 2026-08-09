# **CoChem-MAGE: GC-MS & Chromatographic Emulation**

## **Overview**

**CoChem-MAGE** (Mass and Gas-chromatography Emulator) bridges the computational chemistry pipeline into analytical chemistry. It predicts both the Gas Chromatography Retention Index (RI) and the Electron Ionization Mass Spectrometry (EI-MS) fragmentation pattern of molecular systems.

MAGE utilizes RRKM (Rice-Ramsperger-Kassel-Marcus) statistical rate theory to model molecular fragmentation post-ionization, while its chromatography simulation module applies topological descriptors to predict Kováts Retention Indices and build theoretical chromatograms.

---

## **Scientific & Technical Trade-offs**

* **Heuristic Cleavage vs. Ab Initio Bond Breaking:** To strictly simulate a 70 eV EI-MS collision via ab initio molecular dynamics (AIMD) for 1,000 isomers is computationally prohibitive. MAGE trades extreme ab initio fidelity for statistical speed by utilizing RRKM algorithms and rule-based graph cleavage (McLafferty rearrangements, alpha-cleavage) on molecular graphs.
* **Retention Index Modeling:** MAGE predicts Kováts Retention Indices (RI) on stationary phases (like the non-polar DB-5) using a regression model based on molecular weight (MW), partition coefficient (LogP), and topological polar surface area (TPSA). This allows near-instantaneous GC resolution checks but may degrade in accuracy for highly fluorinated or structurally exotic natural products.

---

## **File Topology & Core Scripts**

MAGE consists of the following key Python scripts:

1. **[cochem_mage_main.py](file:///d:/GitHub-Repo/CoChem-MAGE/cochem_mage_main.py)** (Central System Orchestrator):
   * Coordinates the overall GC-MS simulation pipeline, manages output directories, and compiles logs.
   
2. **[mage_column_registry.py](file:///d:/GitHub-Repo/CoChem-MAGE/mage_column_registry.py)** (Column Intelligence & Ingestor):
   * Profiles the active instrument setup (e.g., Agilent 5977B) and column type (e.g., DB-5MS).
   * Audits batch TPSA descriptors and recommends polar columns (e.g., DB-Wax) if high polar tailing is expected.

3. **[mage_fragmenter.py](file:///d:/GitHub-Repo/CoChem-MAGE/mage_fragmenter.py)** (RRKM Graph-Rewriting Fragmenter):
   * Performs stochastic cleavage of bonds based on the RRKM rate constant proxy:
     $$k(E) = \nu \left(1 - \frac{E_0}{E}\right)^{s-1}$$
     where $E_0$ is the bond dissociation energy (BDE) and $s$ is the active vibrational degrees of freedom.

4. **[cochem_mage_sim.py](file:///d:/GitHub-Repo/CoChem-MAGE/cochem_mage_sim.py)** (Chromatography Simulation & RI Regression):
   * Performs Kováts RI regression based on molecular descriptors and generates theoretical Gaussian peak shapes based on column theoretical plate counts.

---

## **Workflow & How to Run**

To execute a chromatography and fragmentation simulation:

1. **Run the Chromatography Simulator**:
   Generates predicted retention indices ($RI$) and retention times ($t_R$) for a target molecular matrix, and outputs the theoretical chromatogram:
   ```bash
   python cochem_mage_sim.py
   ```

2. **Verify Column Profiles & Ingest molecular queues**:
   Launches column checks and queries NIST webbook APIs if available:
   ```bash
   python mage_column_registry.py
   ```

3. **Run the Central Orchestration Loop**:
   Boots data structures and triggers simulation traces:
   ```bash
   python cochem_mage_main.py
   ```