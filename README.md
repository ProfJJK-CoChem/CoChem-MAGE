# **CoChem-MAGE: GC-MS & Chromatographic Emulation**

## **Overview**

**CoChem-MAGE** (Mass and Gas-chromatography Emulator) bridges the pipeline into analytical instrumentation. It predicts both the Gas Chromatography Retention Index (RI) and the Electron Ionization Mass Spectrometry (EI-MS) fragmentation pattern.

MAGE utilizes RRKM (Rice-Ramsperger-Kassel-Marcus) statistical rate theory to model molecular fragmentation post-ionization, while its Chrom-Opt module applies topological descriptors to predict Kováts Retention Indices.

## **Scientific & Technical Trade-offs**

* **Heuristic Cleavage vs. Ab Initio Bond Breaking:** To strictly simulate a 70 eV EI-MS collision via ab initio molecular dynamics (AIMD) for 1,000 isomers is computationally prohibitive. MAGE trades extreme ab initio fidelity for statistical speed by utilizing RRKM algorithms and rule-based graph cleavage (McLafferty rearrangements, alpha-cleavage).  
* **Retention Index Modeling:** MAGE relies on parameterized molecular descriptors (boiling point estimates, polarizability volumes) rather than explicit solvent-interaction modeling. This allows near-instantaneous GC resolution checks but may degrade in accuracy for highly fluorinated or structurally bizarre natural products absent from the training set.

## **Installation & Setup**

git clone \[https://github.com/CoChem/CoChem-MAGE.git\](https://github.com/CoChem/CoChem-MAGE.git)  
cd CoChem-MAGE

## **How to Run**

MAGE integrates heavily into the master CoChem registry.

1. **Ingest Pre-Computed Geometries:**  
   python cochem\_mage\_ingest.py \--target landscape.h5  
2. **Execute RRKM Fragmentation Simulator:**  
   python cochem\_mage\_rrkm.py  
3. **Compile Chrom-Opt Retention Indices:**  
   python cochem\_mage\_chrom.py