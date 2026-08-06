# %%
import numpy as np
from scipy.optimize import minimize
import warnings

class MageOptimizationEngine:
    """
    Stage 4.0: Van Deemter Optimization & Disaster Recovery for CoChem-MAGE.
    Optimizes GC temperature ramp rates to maximize adjacent peak resolution (Rs).
    Gracefully falls back to raw topological estimates if separation is impossible.
    """
    def __init__(self, target_resolution=1.5, theoretical_plates=15000, max_time_min=60.0):
        self.target_rs = target_resolution
        self.plates = theoretical_plates
        self.max_time = max_time_min
        print(f"⚙️ MAGE Optimizer initialized. Target Rs: {self.target_rs} | Max Time: {self.max_time} min")

    def _simulate_tr_array(self, ri_array, ramp_rate):
        """Simulates retention times for a given ramp rate (isothermal proxy)."""
        dead_time = 1.5
        # Proxy scaling: higher ramp rates compress the chromatogram
        return dead_time + (ri_array / 100.0) * (20.0 / ramp_rate)

    def _objective_function(self, ramp_rate, ri_array):
        """
        The cost function for SciPy. We want to MAXIMIZE minimum resolution, 
        which means MINIMIZING the negative minimum resolution, with penalties.
        """
        # Unpack scalar from SciPy array
        rate = ramp_rate[0] if isinstance(ramp_rate, np.ndarray) else ramp_rate
        
        # Enforce physical bounds (cannot have negative or near-zero ramp)
        if rate < 1.0: return 9999.0
        if rate > 50.0: return 9999.0

        tr_array = self._simulate_tr_array(ri_array, rate)
        max_tr = np.max(tr_array)
        
        # Calculate Resolutions between adjacent peaks
        resolutions = []
        for i in range(len(tr_array) - 1):
            tr1, tr2 = tr_array[i], tr_array[i+1]
            # sigma = tR / sqrt(N)
            w1 = 4 * (tr1 / np.sqrt(self.plates))
            w2 = 4 * (tr2 / np.sqrt(self.plates))
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

    def _disaster_recovery(self, active_matrix, error_msg):
        """Flags the batch when optimization fails, preserving the data pipeline."""
        print(f"⚠️ DISASTER RECOVERY TRIGGERED: {error_msg}")
        for job in active_matrix:
            job["optimization_status"] = "LOW_FIDELITY_FALLBACK"
            job["optimal_ramp_rate"] = 15.0 # Default safe fallback
        return active_matrix

    def optimize_separation(self, active_matrix):
        """Main entry point to solve for the best chromatographic parameters."""
        valid_jobs = [j for j in active_matrix if j.get("status") in ["COMPUTED", "CACHED"]]
        
        if len(valid_jobs) < 2:
            print("✅ Optimization bypassed: < 2 valid components in mixture.")
            for job in valid_jobs: job["optimal_ramp_rate"] = 15.0
            return active_matrix
            
        # Extract and sort RI array
        ri_values = sorted([j.get("predicted_ri", 0) for j in valid_jobs])
        ri_array = np.array(ri_values)
        
        # Catch structurally identical overlapping isomers (RI difference ≈ 0)
        if np.min(np.diff(ri_array)) < 1.0:
            return self._disaster_recovery(active_matrix, "Critical Co-Elution Detected (ΔRI < 1.0).")

        try:
            # SciPy bounded minimization
            res = minimize(
                self._objective_function,
                x0=np.array([10.0]), # Initial guess: 10 C/min
                args=(ri_array,),
                bounds=[(2.0, 40.0)], # Physical oven limits
                method='L-BFGS-B'
            )
            
            if res.success:
                optimal_ramp = round(float(res.x[0]), 2)
                # Calculate the final achieved critical resolution
                tr_array = self._simulate_tr_array(ri_array, optimal_ramp)
                min_rs = min([(tr_array[i+1]-tr_array[i]) / (2 * (tr_array[i]/np.sqrt(self.plates) + tr_array[i+1]/np.sqrt(self.plates))) for i in range(len(tr_array)-1)])
                
                print(f"✅ Optimization converged. Optimal Ramp: {optimal_ramp} °C/min | Critical Rs: {min_rs:.2f}")
                
                if min_rs < 0.6:
                    return self._disaster_recovery(active_matrix, f"Max theoretical Rs ({min_rs:.2f}) below threshold.")
                    
                for job in active_matrix:
                    job["optimal_ramp_rate"] = optimal_ramp
                    job["optimization_status"] = "OPTIMIZED"
            else:
                return self._disaster_recovery(active_matrix, "SciPy Optimizer failed to converge.")

        except Exception as e:
            return self._disaster_recovery(active_matrix, f"Mathematical Fault: {str(e)}")
            
        return active_matrix

# Execute Optimization Test
if __name__ == "__main__":
    # Mocking output from Stage 3.0
    mock_matrix = [
        {"id": "mol_A", "predicted_ri": 950.0, "status": "CACHED"},
        {"id": "mol_B", "predicted_ri": 980.0, "status": "COMPUTED"},
        {"id": "mol_C", "predicted_ri": 1150.0, "status": "CACHED"}
    ]
    
    optimizer = MageOptimizationEngine(target_resolution=1.5)
    optimized_matrix = optimizer.optimize_separation(mock_matrix)
    
    print("\n📊 Final Status Matrix:")
    for m in optimized_matrix:
        print(f"{m['id']} | RI: {m['predicted_ri']} | Ramp: {m.get('optimal_ramp_rate')} °C/min | State: {m.get('optimization_status')}")
# %%