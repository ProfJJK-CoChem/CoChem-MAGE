import logging
from typing import Any
logger = logging.getLogger(__name__)
# %%
import os
import sqlite3
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

class MageDescriptorCache:
    """
    Stage 2.1 (Update): SQLite Descriptor Caching & Physics Generation.
    Now includes SMARTS structural tagging and CCS proxy generation for GC-IMS.
    """
    
    # Pre-compiled SMARTS patterns for rapid structural tagging
    SMARTS_PATTERNS = {
        "Carboxylic Acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
        "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
        "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
        "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
        "Aromatic Halide": Chem.MolFromSmarts("c[F,Cl,Br,I]")
    }

    def __init__(self, db_path="mage_descriptors.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> Any:
        """Initializes or expands the SQLite cache table with new schema elements."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Expanded schema including CCS and chemical classes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS descriptors (
                smiles TEXT PRIMARY KEY,
                mw REAL,
                logp REAL,
                tpsa REAL,
                ccs REAL,
                chemical_class TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _get_chemical_class(self, mol) -> Any:
        """Tags the molecule based on the highest priority SMARTS match."""
        for name, pattern in self.SMARTS_PATTERNS.items():
            if mol.HasSubstructMatch(pattern):
                return name
        return "Aliphatic/Unclassified"

    def _estimate_ccs(self, mol, mw) -> Any:
        """
        Topological proxy for Collisional Cross Section (CCS) in Å^2.
        Rough empirical scaling based on exact molecular weight and atomic topology.
        """
        return round((mw * 0.8) + 40.0, 2)

    def _compute_physics(self, smiles, mol) -> Any:
        """Calculates internal and physical parameters for novel structures."""
        mw = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        ccs = self._estimate_ccs(mol, mw)
        chem_class = self._get_chemical_class(mol)
        
        return {"mw": mw, "logp": logp, "tpsa": tpsa, "ccs": ccs, 
                "chemical_class": chem_class, "status": "COMPUTED"}

    def get_or_compute(self, smiles, mol) -> Any:
        """Checks SQLite for existing descriptors before invoking RDKit."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check Cache
        cursor.execute('SELECT mw, logp, tpsa, ccs, chemical_class, status FROM descriptors WHERE smiles = ?', (smiles,))
        row = cursor.fetchone()
        
        if row:
            conn.close()
            return {"mw": row[0], "logp": row[1], "tpsa": row[2], "ccs": row[3], 
                    "chemical_class": row[4], "status": "CACHED"}
        
        # Compute if novel
        try:
            props = self._compute_physics(smiles, mol)
            cursor.execute('''
                INSERT INTO descriptors (smiles, mw, logp, tpsa, ccs, chemical_class, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (smiles, props["mw"], props["logp"], props["tpsa"], props["ccs"], props["chemical_class"], props["status"]))
            conn.commit()
        except Exception as e:
            logger.error(f"⚠️ Physics Generation Failed for {smiles}: {e}")
            props = {"mw": 0.0, "logp": 0.0, "tpsa": 0.0, "ccs": 0.0, "chemical_class": "ERROR", "status": "FAILED_PHYSICS"}
        finally:
            conn.close()
            
        return props

    def process_batch(self, job_queue) -> Any:
        """Processes the full ingestion queue."""
        processed_queue = []
        for job in job_queue:
            if job.get("status") != "SANITIZED":
                processed_queue.append(job)
                continue
                
            smiles = job["smiles"]
            mol = job.get("rdkit_mol")
            
            if mol is None:
                job["status"] = "FAILED_PHYSICS"
                processed_queue.append(job)
                continue
                
            descriptors = self.get_or_compute(smiles, mol)
            job.update(descriptors)
            processed_queue.append(job)
            
        return processed_queue

# Execute Cache Test
if __name__ == "__main__":
    cache = MageDescriptorCache()
    
    # Sample Validated Data from Stage 1.0 (Segment A)
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    test_job = [{"smiles": smi, "rdkit_mol": Chem.MolFromSmiles(smi), "status": "SANITIZED"}]
    
    result = cache.process_batch(test_job)
    logger.info(f"\\n🧪 Processed Job: {result[0]['smiles']}")
    logger.info(f"Class: {result[0]['chemical_class']} | CCS Proxy: {result[0]['ccs']} Å² | Status: {result[0]['status']}")
# %%
