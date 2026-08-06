# %%
import math
from collections import defaultdict

class HalogenIsotopeGenerator:
    """
    Segment 2.2a (HRMS): Isotope Pattern Generator for CoChem-MAGE.
    Calculates exact-mass natural abundance isotope clusters for fragments 
    containing Chlorine and Bromine using binomial expansions.
    """
    
    # Standard Terrestrial Natural Abundances & Exact Masses
    # Cl: 35Cl (34.96885 Da), 37Cl (36.96590 Da). Delta: +1.99705 Da
    CL_BASE_MASS = 34.96885
    CL_DELTA = 1.99705
    CL_ABUNDANCE = {0: 0.7578, 1: 0.2422} # 0 = Base, 1 = Heavy
    
    # Br: 79Br (78.91834 Da), 81Br (80.91629 Da). Delta: +1.99795 Da
    BR_BASE_MASS = 78.91834
    BR_DELTA = 1.99795
    BR_ABUNDANCE = {0: 0.5069, 1: 0.4931}

    def __init__(self):
        self._cache = {}

    def _binomial_probability(self, n, k, p_light, p_heavy):
        coef = math.comb(n, k)
        return coef * (p_heavy ** k) * (p_light ** (n - k))

    def _get_single_element_cluster(self, n_atoms, delta, p_light, p_heavy):
        cluster = defaultdict(float)
        for k_heavy in range(n_atoms + 1):
            exact_shift = k_heavy * delta
            prob = self._binomial_probability(n_atoms, k_heavy, p_light, p_heavy)
            cluster[exact_shift] = prob
        return cluster

    def get_isotope_cluster(self, num_cl: int, num_br: int):
        if num_cl == 0 and num_br == 0:
            return {0.0: 100.0}
            
        cache_key = (num_cl, num_br)
        if cache_key in self._cache:
            return self._cache[cache_key]

        cl_dist = self._get_single_element_cluster(num_cl, self.CL_DELTA, self.CL_ABUNDANCE[0], self.CL_ABUNDANCE[1])
        br_dist = self._get_single_element_cluster(num_br, self.BR_DELTA, self.BR_ABUNDANCE[0], self.BR_ABUNDANCE[1])

        combined_dist = defaultdict(float)
        if num_cl > 0 and num_br > 0:
            for shift_cl, prob_cl in cl_dist.items():
                for shift_br, prob_br in br_dist.items():
                    combined_dist[shift_cl + shift_br] += prob_cl * prob_br
        elif num_cl > 0:
            combined_dist = cl_dist
        else:
            combined_dist = br_dist

        max_prob = max(combined_dist.values())
        # Rounding to 5 decimal places for typical HRMS resolution
        normalized_cluster = {round(shift, 5): round((prob / max_prob) * 100.0, 2) 
                              for shift, prob in combined_dist.items()}
        
        self._cache[cache_key] = normalized_cluster
        return normalized_cluster
# %%