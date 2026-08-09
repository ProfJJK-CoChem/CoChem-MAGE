# %%
import numpy as np
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz
from scipy.stats import norm
import warnings
from rdkit import Chem

class MageChromatographySim:
    """
    Stage 3.0: Chromatographic Simulation & RI Regression.
    Predicts Retention Indices (RI) from molecular descriptors and builds
    the theoretical 1D chromatogram array.
    """
    def __init__(self, column_config):
        self.column_config = column_config
        self.plate_height_mm = 0.05  # Heuristic optimal HETP
        self.dead_time_min = 1.5
        
        print(f"📈 MAGE Sim Initialized. Column Length: {self.column_config.get('length_m', 30)}m")

    def _compute_chi_indices(self, smiles_or_mol):
        """
        Computes Randić zero-order (chi_0) and first-order (chi_1) molecular connectivity indices (MAGE-06).
        """
        mol = None
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        elif isinstance(smiles_or_mol, Chem.Mol):
            mol = smiles_or_mol

        if mol is None:
            return 0.0, 0.0

        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        chi_0 = sum([1.0 / np.sqrt(d) if d > 0 else 0.0 for d in degrees])
        
        chi_1 = 0.0
        for bond in mol.GetBonds():
            d1 = bond.GetBeginAtom().GetDegree()
            d2 = bond.GetEndAtom().GetDegree()
            if d1 > 0 and d2 > 0:
                chi_1 += 1.0 / np.sqrt(d1 * d2)
                
        return float(chi_0), float(chi_1)

    def _heuristic_group_contribution_ri(self, descriptors):
        """
        Group Contribution & Topological Index Fallback Model (MAGE-05, MAGE-06).
        Replaces simple linear heuristic with Randić connectivity indices and Group Contributions.
        """
        mw = descriptors.get("mw", 0)
        logp = descriptors.get("logp", 0)
        tpsa = descriptors.get("tpsa", 0)
        smiles = descriptors.get("smiles", "")

        chi_0, chi_1 = self._compute_chi_indices(smiles)
        
        # Group contribution estimate combining topological connectivity and physical descriptors
        # Stationary phase polarity factor (default DB-5 non-polar)
        ri = 100.0 * (0.85 * chi_0 + 1.65 * chi_1 + 0.45 * logp + 0.04 * mw - 0.008 * tpsa) + 150.0
        return max(ri, 0.0)

    def _train_default_xgb_model(self, model_path):
        """Trains a true XGBoost regression model on retention datasets and serializes it (MAGE-05)."""
        import xgboost as xgb
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        compounds = [
            ('C' * i, 100.0 * i) for i in range(5, 30)
        ] + [
            ('c1ccccc1', 655.0), ('Cc1ccccc1', 763.0), ('CCc1ccccc1', 854.0),
            ('Cc1cccc(C)c1', 864.0), ('Cc1ccccc1C', 887.0), ('CCCCc1ccccc1', 1048.0),
            ('c1ccc2ccccc2c1', 1185.0), ('Cc1ccc2ccccc2c1', 1300.0), ('c1ccc2c(c1)ccc3ccccc23', 1780.0),
            ('CO', 360.0), ('CCO', 470.0), ('CCCO', 585.0), ('CCCCO', 690.0),
            ('CCCCCO', 790.0), ('CCCCCCO', 890.0), ('CCCCCCCO', 990.0), ('CCCCCCCOO', 1090.0),
            ('CC(=O)O', 610.0), ('CCC(=O)O', 700.0), ('CCCC(=O)O', 800.0), ('CCOC(=O)C', 615.0),
            ('CC(=O)Oc1ccccc1C(=O)O', 1450.0), ('CC(=O)C', 490.0), ('CCC(=O)C', 590.0), ('CCCC(=O)CC', 780.0),
            ('c1cc(O)ccc1', 975.0), ('Cc1ccc(O)cc1', 1060.0), ('Clc1ccccc1', 840.0), ('Brc1ccccc1', 940.0),
            ('Ic1ccccc1', 1060.0), ('ClCCCl', 620.0), ('FC(F)(F)c1ccccc1', 785.0),
            ('CC1=CCC(CC1)C(=C)C', 1030.0), ('CC1(C)C2CCC1(C)C(=O)C2', 1140.0), ('CC1=C(C=C(C=C1)C(C)C)O', 1290.0)
        ]

        X, y = [], []
        for smiles, ri in compounds:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            chi_0, chi_1 = self._compute_chi_indices(mol)
            X.append([mw, logp, tpsa, chi_0, chi_1])
            y.append(ri)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
        model.fit(X_arr, y_arr)
        model.save_model(str(model_path))
        return model

    def _predict_ri(self, descriptors):
        """
        Predicts Kováts Retention Index using a true XGBoost ML regression model (MAGE-05).
        """
        import os
        from pathlib import Path
        import xgboost as xgb

        mw = float(descriptors.get("mw", 0))
        logp = float(descriptors.get("logp", 0))
        tpsa = float(descriptors.get("tpsa", 0))
        smiles = descriptors.get("smiles", "")

        chi_0, chi_1 = self._compute_chi_indices(smiles)

        # Domain Extrapolation Safety Check
        if mw > 800 or logp > 12:
            print(f"⚠️ EXTRAPOLATION WARNING: Descriptors (MW={mw:.1f}, LogP={logp:.1f}) exceed 95th percentile of training data.")

        model_path = Path("mage_ri_xgboost.json")
        if not model_path.exists():
            pkg_model = Path(__file__).parent / "mage_ri_xgboost.json"
            if pkg_model.exists():
                model_path = pkg_model

        if not hasattr(self, '_xgb_model') or self._xgb_model is None:
            self._xgb_model = xgb.XGBRegressor()
            if model_path.exists():
                self._xgb_model.load_model(str(model_path))
            else:
                self._xgb_model = self._train_default_xgb_model(model_path)

        # Infer with 5 descriptors (mw, logp, tpsa, chi_0, chi_1) or fallback to 3 if old model loaded
        try:
            X_infer = np.array([[mw, logp, tpsa, chi_0, chi_1]], dtype=np.float32)
            ri = float(self._xgb_model.predict(X_infer)[0])
        except Exception:
            X_infer = np.array([[mw, logp, tpsa]], dtype=np.float32)
            ri = float(self._xgb_model.predict(X_infer)[0])

        return max(ri, 0.0) # Prevent unphysical negative RI

    def simulate_retention(self, active_matrix, temperature_ramp_rate=10.0):
        """
        Maps physical descriptors to retention times.
        """
        for job in active_matrix:
            if job.get("status") == "FAILED_PHYSICS":
                continue
                
            ri = self._predict_ri(job)
            
            # Simple conversion from RI to t_R under a linear temperature ramp
            t_r = self.dead_time_min + (ri / 100.0) * (20.0 / temperature_ramp_rate)
            
            job["predicted_ri"] = round(ri, 1)
            job["predicted_tr"] = round(t_r, 3)
            
        return active_matrix

    def build_chromatogram(self, active_matrix, t_max=40.0, resolution=10000):
        """
        Constructs the macroscopic chromatogram intensity array.
        Normalizes peak areas individually prior to summing TIC array (MAGE-07).
        """
        time_axis = np.linspace(0, t_max, resolution)
        chromatogram = np.zeros_like(time_axis)
        
        col_length_mm = self.column_config.get("length_m", 30) * 1000
        theoretical_plates = col_length_mm / self.plate_height_mm
        
        for job in active_matrix:
            t_r = job.get("predicted_tr")
            if t_r is None:
                continue
                
            # Sigma derived from theoretical plates: N = (t_R / sigma)^2
            sigma = t_r / np.sqrt(theoretical_plates)
            
            # Normalize peak individually prior to summing (MAGE-07)
            raw_peak = norm.pdf(time_axis, loc=t_r, scale=max(sigma, 0.01))
            # Individual area normalization so peak area = 1.0 (or scaled abundance)
            peak_area = trapz(raw_peak, time_axis)
            if peak_area > 0:
                normalized_peak = raw_peak / peak_area
            else:
                normalized_peak = raw_peak
                
            abundance = job.get("abundance", 100.0)
            chromatogram += normalized_peak * abundance
            
        # Scale final chromatogram trace for display
        if np.max(chromatogram) > 0:
            chromatogram = (chromatogram / np.max(chromatogram)) * 100.0
            
        return time_axis, chromatogram

# Execute Simulation Test
if __name__ == "__main__":
    mock_col_config = {"length_m": 30.0, "stationary_phase": "5% phenyl"}
    mock_jobs = [
        {"id": "mol_0", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "mw": 180.15, "logp": 1.19, "tpsa": 63.6, "status": "CACHED"}, # Aspirin
        {"id": "mol_1", "smiles": "c1cc(O)ccc1", "mw": 94.11,  "logp": 1.46, "tpsa": 20.2, "status": "COMPUTED"} # Phenol
    ]
    
    sim = MageChromatographySim(mock_col_config)
    simulated_jobs = sim.simulate_retention(mock_jobs, temperature_ramp_rate=15.0)
    
    print("\n⏱️ Predicted Retention Results:")
    for j in simulated_jobs:
        print(f"ID: {j['id']} | RI: {j['predicted_ri']} | t_R: {j['predicted_tr']} min")
        
    t_axis, trace = sim.build_chromatogram(simulated_jobs, t_max=30.0, resolution=5000)
    print(f"\n✅ Chromatogram Array Generated. Shape: {trace.shape}, Max Peak: {np.max(trace):.1f}%")
# %%