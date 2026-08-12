# %%
from typing import Any
import numpy as np
import logging
from scipy.optimize import minimize
import warnings

def _safe_float(val, default=0.0) -> Any:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

logger = logging.getLogger("MageOptimizationEngine")

class MageOptimizationEngine:
    """
    Stage 4.0: Van Deemter Optimization & Disaster Recovery for CoChem-MAGE.
    Optimizes GC temperature ramp rates to maximize adjacent peak resolution (Rs).
    Gracefully falls back to raw topological estimates if separation is impossible.
    """
    def __init__(self, target_resolution=1.5, theoretical_plates=15000, max_time_min=60.0, instrument_profile=None) -> None:
        self.target_rs = target_resolution
        self.plates = theoretical_plates
        self.max_time = max_time_min
        self.instrument_profile = instrument_profile or {}
        logger.info(f"⚙️ MAGE Optimizer initialized. Target Rs: {self.target_rs} | Max Time: {self.max_time} min")

    def _simulate_tr_array(self, ri_array, ramp_rate) -> Any:
        """Simulates retention times for a given ramp rate (isothermal proxy)."""
        dead_time = 1.5
        # Proxy scaling: higher ramp rates compress the chromatogram
        return dead_time + (ri_array / 100.0) * (20.0 / ramp_rate)

    def _compute_tpgc_peak_width(self, tr, dead_time=1.5, alpha=0.05, station_phase_params=None) -> Any:
        """
        Temperature-Programmed GC (TPGC) peak width model grounded in experimental phase parameters (MAGE-06).
        w = w_0 * (1 + alpha * t_R)^(1/2) where w_0 = 4 * dead_time / sqrt(N).
        Safely guarded against N <= 0, negative t_R, and dead_time <= 0.
        """
        safe_dead_time = max(_safe_float(dead_time, 1.5), 1.0)
        plates = self.plates
        if station_phase_params and isinstance(station_phase_params, dict):
            length_mm = _safe_float(station_phase_params.get("length_m"), 30.0) * 1000.0
            hetp_mm = _safe_float(station_phase_params.get("hetp_mm"), 0.05)
            plates = length_mm / max(hetp_mm, 1e-4)

        safe_plates = max(_safe_float(plates, 15000.0), 1.0)
        w_0 = 4.0 * (safe_dead_time / np.sqrt(safe_plates))
        tr_val = _safe_float(tr, 0.0)
        safe_alpha = _safe_float(alpha, 0.05)
        sqrt_term = np.sqrt(max(1.0 + safe_alpha * tr_val, 1e-6))
        return float(w_0 * sqrt_term)

    def _objective_function(self, ramp_rate, ri_array, max_ramp=40.0) -> Any:
        """
        The cost function for SciPy. We want to MAXIMIZE minimum resolution, 
        which means MINIMIZING the negative minimum resolution, with penalties.
        """
        # Unpack scalar from SciPy array
        rate = ramp_rate[0] if isinstance(ramp_rate, np.ndarray) else ramp_rate
        
        # Enforce physical bounds (cannot have negative or near-zero ramp)
        if rate < 1.0: return 9999.0
        if rate > max_ramp: return 9999.0

        if ri_array is None or len(ri_array) == 0:
            return 9999.0

        tr_array = self._simulate_tr_array(ri_array, rate)
        if len(tr_array) == 0:
            return 9999.0
        max_tr = np.max(tr_array)
        
        # Calculate Resolutions between adjacent peaks using TPGC peak width (MAGE-03)
        resolutions = []
        for i in range(len(tr_array) - 1):
            tr1, tr2 = tr_array[i], tr_array[i+1]
            w1 = self._compute_tpgc_peak_width(tr1)
            w2 = self._compute_tpgc_peak_width(tr2)
            rs = (tr2 - tr1) / (0.5 * (w1 + w2))
            resolutions.append(rs)
            
        min_rs = min(resolutions) if resolutions else self.target_rs
        
        # Penalty for exceeding max time
        time_penalty = max(0, max_tr - self.max_time) * 10.0
        
        # We want to minimize (-min_rs) to push Rs higher, bounded by target_rs
        # Once Rs hits target (e.g., 1.5), we prioritize faster run times (higher ramp)
        if min_rs > self.target_rs:
            return -self.target_rs + (max_tr * 0.1) # Gently push for speed
        else:
            return -min_rs + time_penalty

    def _disaster_recovery(self, active_matrix, error_msg) -> Any:
        """
        Flags the batch when optimization fails, preserving the data pipeline.
        Logs formal warning telemetry and sets inspectable exception flags (MAGE-04).
        """
        warning_str = f"DISASTER RECOVERY TRIGGERED: {error_msg}"
        logger.warning(f"⚠️ {warning_str}")
        logger.warning(warning_str)
        if not active_matrix or not isinstance(active_matrix, (list, tuple)):
            return []
        for job in [j for j in active_matrix if isinstance(j, dict)]:
            job["optimization_status"] = "LOW_FIDELITY_FALLBACK"
            job["optimal_ramp_rate"] = 15.0 # Default safe fallback
            job["disaster_recovery_flag"] = True
            job["disaster_recovery_reason"] = error_msg
            job["provenance_tag"] = "[E]"
        return active_matrix

    def optimize_separation(self, active_matrix, instrument_profile=None) -> Any:
        """Main entry point to solve for the best chromatographic parameters."""
        if not active_matrix or not isinstance(active_matrix, (list, tuple)):
            return []

        profile = instrument_profile or self.instrument_profile
        max_ramp = profile.get("column_physics", {}).get("max_ramp_rate", 40.0) if profile else 40.0

        valid_jobs = [j for j in active_matrix if isinstance(j, dict) and j.get("status") in ["COMPUTED", "CACHED"]]
        
        if len(valid_jobs) < 2:
            logger.info("✅ Optimization bypassed: < 2 valid components in mixture.")
            for job in valid_jobs:
                job["optimal_ramp_rate"] = 15.0
                job["provenance_tag"] = job.get("provenance_tag", "[E]")
            return active_matrix
            
        # Extract and sort RI array
        ri_values = sorted([_safe_float(j.get("predicted_ri"), 0.0) for j in valid_jobs])
        ri_array = np.array(ri_values)
        if len(ri_array) < 2:
            return active_matrix
        
        # Catch structurally identical overlapping isomers (RI difference ≈ 0)
        if np.min(np.diff(ri_array)) < 1.0:
            return self._disaster_recovery(active_matrix, "Critical Co-Elution Detected (ΔRI < 1.0).")

        try:
            initial_guess = min(10.0, max_ramp / 2.0)
            # SciPy bounded minimization with dynamic ramp bounds (MAGE-02)
            res = minimize(
                self._objective_function,
                x0=np.array([initial_guess]),
                args=(ri_array, max_ramp),
                bounds=[(2.0, max_ramp)],
                method='L-BFGS-B'
            )
            
            if res.success:
                optimal_ramp = round(float(res.x[0]), 2)
                # Calculate the final achieved critical resolution using TPGC peak width
                tr_array = self._simulate_tr_array(ri_array, optimal_ramp)
                resolutions = [(tr_array[i+1]-tr_array[i]) / (0.5 * (self._compute_tpgc_peak_width(tr_array[i]) + self._compute_tpgc_peak_width(tr_array[i+1]))) for i in range(len(tr_array)-1)]
                min_rs = min(resolutions) if resolutions else 1.5
                
                logger.info(f"✅ Optimization converged. Optimal Ramp: {optimal_ramp} °C/min | Critical Rs: {min_rs:.2f}")
                
                if min_rs < 0.6:
                    return self._disaster_recovery(active_matrix, f"Max theoretical Rs ({min_rs:.2f}) below threshold.")
                    
                for job in valid_jobs:
                    job["optimal_ramp_rate"] = optimal_ramp
                    job["optimization_status"] = "OPTIMIZED"
                    job["disaster_recovery_flag"] = False
                    if "provenance_tag" not in job:
                        job["provenance_tag"] = "[D]"
            else:
                return self._disaster_recovery(active_matrix, "SciPy Optimizer failed to converge.")

        except Exception as e:
            return self._disaster_recovery(active_matrix, f"Mathematical Fault: {str(e)}")
            
        return active_matrix

# Execute Optimization Test
if __name__ == "__main__":
    mock_matrix = [
        {"id": "mol_A", "predicted_ri": 950.0, "status": "CACHED"},
        {"id": "mol_B", "predicted_ri": 980.0, "status": "COMPUTED"},
        {"id": "mol_C", "predicted_ri": 1150.0, "status": "CACHED"}
    ]
    
    optimizer = MageOptimizationEngine(target_resolution=1.5)
    optimized_matrix = optimizer.optimize_separation(mock_matrix)
    
    logger.info("\n📊 Final Status Matrix:")
    for m in optimized_matrix:
        logger.info(f"{m['id']} | RI: {m['predicted_ri']} | Ramp: {m.get('optimal_ramp_rate')} °C/min | State: {m.get('optimization_status')}")
# %%