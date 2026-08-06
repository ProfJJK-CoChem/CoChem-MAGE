# %%
import numpy as np
from scipy.stats import norm
import warnings

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

    def _predict_ri(self, descriptors):
        """
        Predicts Kováts Retention Index using a heuristic linear model.
        (In production, replace with loaded sklearn/XGBoost model).
        """
        mw = descriptors.get("mw", 0)
        logp = descriptors.get("logp", 0)
        tpsa = descriptors.get("tpsa", 0)

        # Domain Extrapolation Safety Check
        if mw > 800 or logp > 12:
            print(f"⚠️ EXTRAPOLATION WARNING: Descriptors (MW={mw:.1f}, LogP={logp:.1f}) exceed 95th percentile of training data.")
        
        # Heuristic weights for a 5% phenyl (DB-5) non-polar column
        # LogP heavily correlates with non-polar retention, MW adds bulk, TPSA reduces it slightly.
        ri = 100 * (1.2 * logp + 0.08 * mw - 0.01 * tpsa) + 300
        
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
            # (Roughly 1 min per 100 RI units depending on flow/ramp)
            t_r = self.dead_time_min + (ri / 100.0) * (20.0 / temperature_ramp_rate)
            
            job["predicted_ri"] = round(ri, 1)
            job["predicted_tr"] = round(t_r, 3)
            
        return active_matrix

    def build_chromatogram(self, active_matrix, t_max=40.0, resolution=10000):
        """
        Constructs the macroscopic chromatogram intensity array.
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
            
            # Add Gaussian peak to chromatogram
            # Assuming equal concentration/response factor for this theoretical trace
            peak = norm.pdf(time_axis, loc=t_r, scale=max(sigma, 0.01))
            chromatogram += peak
            
        # Normalize maximum intensity to 100%
        if np.max(chromatogram) > 0:
            chromatogram = (chromatogram / np.max(chromatogram)) * 100.0
            
        return time_axis, chromatogram

# Execute Simulation Test
if __name__ == "__main__":
    # Mock parameters from Stage 0/1/2
    mock_col_config = {"length_m": 30.0, "stationary_phase": "5% phenyl"}
    mock_jobs = [
        {"id": "mol_0", "mw": 180.15, "logp": 1.19, "tpsa": 63.6, "status": "CACHED"}, # Aspirin
        {"id": "mol_1", "mw": 94.11,  "logp": 1.46, "tpsa": 20.2, "status": "COMPUTED"} # Phenol
    ]
    
    sim = MageChromatographySim(mock_col_config)
    simulated_jobs = sim.simulate_retention(mock_jobs, temperature_ramp_rate=15.0)
    
    print("\\n⏱️ Predicted Retention Results:")
    for j in simulated_jobs:
        print(f"ID: {j['id']} | RI: {j['predicted_ri']} | t_R: {j['predicted_tr']} min")
        
    t_axis, trace = sim.build_chromatogram(simulated_jobs, t_max=15.0, resolution=5000)
    print(f"\\n✅ Chromatogram Array Generated. Shape: {trace.shape}, Max Peak: {np.max(trace):.1f}%")
# %%