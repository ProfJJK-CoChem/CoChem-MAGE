import logging
from typing import Any
logger = logging.getLogger(__name__)
# %%
import numpy as np
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz
from scipy.stats import norm
import warnings
from rdkit import Chem

def determine_product_class(input_data: dict) -> dict:
    """
    Product Class A/B/C Routing Decision Tree per Method Matrix §1.1-1.5.
    Returns dictionary with product_class and target_accuracy_window.
    """
    if not isinstance(input_data, dict):
        input_data = {}

    is_difference_calc = bool(input_data.get("is_difference_calculation") or input_data.get("relative_shift_mode"))
    has_measured_parent = bool(input_data.get("measured_parent_isotopologue") or input_data.get("parent_experimental_spectrum"))

    if is_difference_calc:
        return {
            "product_class": "PRODUCT_C",
            "target_accuracy_window": "Difference cancellation window",
            "recommended_tier": "T1-1h",
            "provenance_tag": "[D]"
        }
    elif has_measured_parent:
        return {
            "product_class": "PRODUCT_B",
            "target_accuracy_window": "±0.03% to ±0.1%",
            "recommended_tier": "T2-12h",
            "provenance_tag": "[D]"
        }
    else:
        return {
            "product_class": "PRODUCT_A",
            "target_accuracy_window": "±0.3% to ±0.5%",
            "recommended_tier": "T1-30min",
            "provenance_tag": "[E]"
        }

class MageChromatographySim:
    """
    Stage 3.0: Chromatographic Simulation & RI Regression.
    Predicts Retention Indices (RI) from molecular descriptors and builds
    the theoretical 1D chromatogram array.
    """
    def __init__(self, column_config) -> None:
        self.column_config = column_config
        self.plate_height_mm = 0.05  # Heuristic optimal HETP
        self.dead_time_min = 1.5
        
        logger.info(f"📈 MAGE Sim Initialized. Column Length: {self.column_config.get('length_m', 30)}m")

    def _compute_chi_indices(self, smiles_or_mol) -> Any:
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

    def _heuristic_group_contribution_ri(self, descriptors) -> Any:
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

    def _abraham_solvation_parameters(self, descriptors) -> Any:
        """
        Estimates solute Abraham solvation parameters (E, S, A, B, V) from descriptors and SMILES.
        E: Excess molar refraction
        S: Dipolarity / polarizability
        A: H-bond acidity
        B: H-bond basicity
        V: McGowan volume (cm^3/mol / 100)
        """
        smiles = descriptors.get("smiles", "")
        mw = float(descriptors.get("mw", 0))
        tpsa = float(descriptors.get("tpsa", 0))
        logp = float(descriptors.get("logp", 0))
        
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        
        # McGowan Volume V ~ MW / 100
        V = mw / 100.0
        
        if mol is not None:
            # H-bond acidity A: OH, NH donors
            A = float(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['O', 'N'] and atom.GetTotalNumHs() > 0))
            # H-bond basicity B: O, N acceptors
            B = float(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['O', 'N']))
            # Excess refraction E: aromatic rings + halogens
            aromatic_rings = len(Chem.GetSSSR(mol))
            halogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])
            E = 0.8 * aromatic_rings + 0.2 * halogens
        else:
            A = max(0.0, tpsa / 30.0)
            B = max(0.0, tpsa / 20.0)
            E = max(0.0, logp * 0.2)
            
        # Dipolarity S: related to TPSA / MW ratio
        S = (tpsa / (mw + 1.0)) * 2.5 + (0.1 if logp < 1.0 else 0.0)
        return {"E": float(E), "S": float(S), "A": float(A), "B": float(B), "V": float(V)}

    def _apply_stationary_phase_partitioning(self, base_ri, descriptors) -> Any:
        """
        Modifies Retention Index (RI) based on stationary phase polarity (DB-5 vs DB-Wax)
        using Abraham solvation parameters (MAGE Suggestion 66).
        """
        phase = str(self.column_config.get("stationary_phase", "5% phenyl")).lower()
        abraham = self._abraham_solvation_parameters(descriptors)
        
        # DB-Wax (polar polyethylene glycol phase) vs DB-5 (non-polar 5% phenyl methylpolysiloxane)
        if "wax" in phase or "peg" in phase or "polyethylene" in phase:
            # Polar phase strongly interacts with dipolarity S, H-bond acidity A and basicity B
            delta_ri = 100.0 * (1.25 * abraham["S"] + 1.80 * abraham["A"] + 0.45 * abraham["B"] + 0.20 * abraham["E"])
            return base_ri + delta_ri
        else:
            # DB-5 non-polar phase baseline
            return base_ri

    def compute_van_deemter_hetp(self, u_cm_s=None, station_phase_params=None) -> Any:
        """
        Computes Golay/van Deemter peak height HETP H = B/u + (C_s + C_m)*u (MAGE-06).
        Uses station phase data (d_f, d_c, D_s, D_m) when provided.
        Returns (H_mm, u_cm_s, provenance_tag).
        """
        if u_cm_s is None:
            # Carrier gas linear velocity u = L / t_m
            length_m = float(self.column_config.get("length_m", 30.0))
            t_m_sec = self.dead_time_min * 60.0
            u_cm_s = (length_m * 100.0) / t_m_sec # cm/s

        u_cm_s = max(float(u_cm_s), 1e-3)

        if station_phase_params and isinstance(station_phase_params, dict):
            # Phase-grounded Golay equation with non-None parameter fallbacks
            d_f_um = station_phase_params.get("film_thickness_um")
            d_f_um = float(d_f_um) if d_f_um is not None else 0.25
            d_f_cm = d_f_um * 1e-4

            d_c_mm = station_phase_params.get("inner_diameter_mm")
            d_c_mm = float(d_c_mm) if d_c_mm is not None else 0.25
            d_c_cm = d_c_mm * 0.1

            k_val = station_phase_params.get("retention_factor_k")
            k = float(k_val) if k_val is not None else 5.0

            D_m_val = station_phase_params.get("binary_diffusion_m2_s")
            D_m_val = float(D_m_val) if D_m_val is not None else 1e-5
            D_m = D_m_val * 10000.0 # cm^2/s

            D_s_val = station_phase_params.get("stationary_diffusion_m2_s")
            D_s_val = float(D_s_val) if D_s_val is not None else 1e-9
            D_s = D_s_val * 10000.0 # cm^2/s

            B = 2.0 * D_m # Longitudinal diffusion
            denom_k = max(abs(1.0 + k), 1e-6) ** 2
            C_s = (2.0 / 3.0) * (k / denom_k) * ((d_f_cm**2) / max(D_s, 1e-12))
            C_m = ((1.0 + 6.0*k + 11.0*(k**2)) / (24.0 * denom_k)) * ((d_c_cm**2) / max(D_m, 1e-12))
            
            H_cm = (B / u_cm_s) + (C_s + C_m) * u_cm_s
            H_mm = H_cm * 10.0
            tag = "[D]"
        else:
            # Empirical fallback HETP
            A = 0.01
            B = 0.5
            C = 0.001
            H_mm = A + (B / u_cm_s) + (C * u_cm_s)
            tag = "[E]"

        return float(H_mm), float(u_cm_s), tag

    def _train_default_xgb_model(self, model_path) -> Any:
        """Trains a true XGBoost regression model on dataset loaded from cochem_mage_data/mage_ri_dataset.json (MAGE-05/Suggestion 65)."""
        import json
        from pathlib import Path
        import xgboost as xgb
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        dataset_path = Path(__file__).parent / "cochem_mage_data" / "mage_ri_dataset.json"
        if not dataset_path.exists():
            alt_path = Path("cochem_mage_data/mage_ri_dataset.json")
            if alt_path.exists():
                dataset_path = alt_path
            else:
                raise FileNotFoundError(f"Training dataset 'cochem_mage_data/mage_ri_dataset.json' not found at {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            compounds_data = json.loads(f.read())

        X, y = [], []
        for item in compounds_data:
            smiles = item.get("smiles")
            ri = item.get("ri")
            if not smiles or ri is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mw = float(item.get("mw", Descriptors.MolWt(mol)))
            logp = float(item.get("logp", Descriptors.MolLogP(mol)))
            tpsa = float(item.get("tpsa", Descriptors.TPSA(mol)))
            chi_0, chi_1 = self._compute_chi_indices(mol)
            X.append([mw, logp, tpsa, chi_0, chi_1])
            y.append(float(ri))

        if not X:
            raise ValueError(f"No valid training records found in {dataset_path}")

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
        model.fit(X_arr, y_arr)
        model.save_model(str(model_path))
        return model

    def _predict_ri(self, descriptors) -> Any:
        """
        Predicts Kováts Retention Index using XGBoost ML regression model and Abraham solvation stationary phase scaling.
        """
        import os
        from pathlib import Path
        import xgboost as xgb

        if not isinstance(descriptors, dict):
            descriptors = {}

        mw = float(descriptors.get("mw") or 0)
        logp = float(descriptors.get("logp") or 0)
        tpsa = float(descriptors.get("tpsa") or 0)
        smiles = descriptors.get("smiles") or ""

        chi_0, chi_1 = self._compute_chi_indices(smiles)

        # Domain Extrapolation Safety Check
        if mw > 800 or logp > 12:
            logger.warning(f"⚠️ EXTRAPOLATION WARNING: Descriptors (MW={mw:.1f}, LogP={logp:.1f}) exceed 95th percentile of training data.")

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
            base_ri = float(self._xgb_model.predict(X_infer)[0])
        except Exception:
            X_infer = np.array([[mw, logp, tpsa]], dtype=np.float32)
            base_ri = float(self._xgb_model.predict(X_infer)[0])

        base_ri = max(base_ri, 0.0)
        # Apply stationary phase liquid-film partitioning model (DB-5 vs DB-Wax)
        final_ri = self._apply_stationary_phase_partitioning(base_ri, descriptors)
        return max(final_ri, 0.0)

    def simulate_retention(self, active_matrix, temperature_ramp_rate=10.0) -> Any:
        """
        Maps physical descriptors to retention times.
        Attaches product_class, accuracy_window, and provenance_tag.
        """
        if not active_matrix or not isinstance(active_matrix, (list, tuple)):
            return []

        valid_matrix = [j for j in active_matrix if isinstance(j, dict)]

        for job in valid_matrix:
            if job.get("status") == "FAILED_PHYSICS":
                continue
                
            p_info = determine_product_class(job)
            job["product_class"] = p_info["product_class"]
            job["accuracy_window"] = p_info["target_accuracy_window"]
            job["provenance_tag"] = p_info["provenance_tag"]

            ri = self._predict_ri(job)
            
            # Conversion from RI to t_R under a linear temperature ramp
            t_r = self.dead_time_min + (ri / 100.0) * (20.0 / temperature_ramp_rate)
            
            job["predicted_ri"] = round(ri, 1)
            job["predicted_tr"] = round(t_r, 3)
            
        return active_matrix

    def build_chromatogram(self, active_matrix, t_max=40.0, resolution=10000) -> Any:
        """
        Constructs the macroscopic chromatogram intensity array.
        Normalizes peak areas individually prior to summing TIC array (MAGE-07).
        Uses van Deemter peak height broadening H(u).
        """
        time_axis = np.linspace(0, t_max, resolution)
        chromatogram = np.zeros_like(time_axis)
        
        col_length_mm = self.column_config.get("length_m", 30.0) * 1000.0
        hetp_mm, u_cm_s, tag = self.compute_van_deemter_hetp()
        self.plate_height_mm = hetp_mm
        theoretical_plates = col_length_mm / hetp_mm
        
        if not active_matrix or not isinstance(active_matrix, (list, tuple)):
            active_matrix = []
        valid_matrix = [job for job in active_matrix if isinstance(job, dict)]

        for job in valid_matrix:
            t_r = job.get("predicted_tr")
            if t_r is None:
                continue
                
            # Sigma derived from theoretical plates: N = (t_R / sigma)^2
            sigma = t_r / np.sqrt(theoretical_plates)
            
            # Normalize peak individually prior to summing (MAGE-07)
            raw_peak = norm.pdf(time_axis, loc=t_r, scale=max(sigma, 0.01))
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

def calculate_kovats_ri_isothermal(t_rx: float, t_rn: float, t_rN: float, n: int, N: int, t_m: float = 1.5) -> float:
    """
    Calculates isothermal Kovats Retention Index using log adjusted retention times.
    I = 100 * [n + (N - n) * (log(t_rx') - log(t_rn')) / (log(t_rN') - log(t_rn'))]
    """
    t_adj_x = max(t_rx - t_m, 1e-6)
    t_adj_n = max(t_rn - t_m, 1e-6)
    t_adj_N = max(t_rN - t_m, 1e-6)
    denom = np.log(t_adj_N) - np.log(t_adj_n)
    if abs(denom) < 1e-12:
        return float(100 * n)
    return float(100.0 * (n + (N - n) * (np.log(t_adj_x) - np.log(t_adj_n)) / denom))

def calculate_kovats_ri_tp(t_rx: float, t_rn: float, t_rN: float, n: int, N: int) -> float:
    """
    Calculates temperature-programmed Kovats Retention Index (van Den Dool & Kratz formula).
    I_TP = 100 * [n + (N - n) * (t_rx - t_rn) / (t_rN - t_rn)]
    """
    denom = t_rN - t_rn
    if abs(denom) < 1e-12:
        return float(100 * n)
    return float(100.0 * (n + (N - n) * (t_rx - t_rn) / denom))

# Execute Simulation Test
if __name__ == "__main__":
    mock_col_config = {"length_m": 30.0, "stationary_phase": "5% phenyl"}
    mock_jobs = [
        {"id": "mol_0", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "mw": 180.15, "logp": 1.19, "tpsa": 63.6, "status": "CACHED"}, # Aspirin
        {"id": "mol_1", "smiles": "c1cc(O)ccc1", "mw": 94.11,  "logp": 1.46, "tpsa": 20.2, "status": "COMPUTED"} # Phenol
    ]
    
    sim = MageChromatographySim(mock_col_config)
    simulated_jobs = sim.simulate_retention(mock_jobs, temperature_ramp_rate=15.0)
    
    logger.info("\n⏱️ Predicted Retention Results:")
    for j in simulated_jobs:
        logger.info(f"ID: {j['id']} | RI: {j['predicted_ri']} | t_R: {j['predicted_tr']} min")
        
    t_axis, trace = sim.build_chromatogram(simulated_jobs, t_max=30.0, resolution=5000)
    logger.info(f"\n✅ Chromatogram Array Generated. Shape: {trace.shape}, Max Peak: {np.max(trace):.1f}%")
# %%