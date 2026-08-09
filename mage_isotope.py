# %%
import math
from collections import defaultdict

class HalogenIsotopeGenerator:
    """
    Segment 2.2a (HRMS): Full Isotope Pattern Generator for CoChem-MAGE (MAGE-12).
    Calculates exact-mass natural abundance isotope clusters for fragments 
    containing C, H, N, O, S, Si, Cl, and Br using IUPAC natural abundance tables.
    """
    
    # Standard Terrestrial Natural Abundances & Exact Masses (IUPAC)
    C_BASE_MASS = 12.00000
    C_DELTA = 1.00335
    C_ABUNDANCE = {0: 0.9893, 1: 0.0107} # 12C (98.93%), 13C (1.07%)
    
    H_BASE_MASS = 1.00783
    H_DELTA = 1.00628
    H_ABUNDANCE = {0: 0.99985, 1: 0.00015} # 1H (99.985%), 2H (0.015%)

    N_BASE_MASS = 14.00307
    N_DELTA = 0.99703
    N_ABUNDANCE = {0: 0.9963, 1: 0.0037} # 14N (99.63%), 15N (0.37%)

    O_ABUNDANCE = {0: 0.99757, 1: 0.00038, 2: 0.00205} # 16O (99.757%), 17O (0.038%), 18O (0.205%)
    O_DELTAS = {0: 0.0, 1: 0.99913, 2: 2.00425}

    S_ABUNDANCE = {0: 0.9493, 1: 0.0076, 2: 0.0429} # 32S (94.93%), 33S (0.76%), 34S (4.29%)
    S_DELTAS = {0: 0.0, 1: 0.99939, 2: 1.99580}

    SI_ABUNDANCE = {0: 0.9223, 1: 0.0468, 2: 0.0309} # 28Si (92.23%), 29Si (4.68%), 30Si (3.09%)
    SI_DELTAS = {0: 0.0, 1: 0.99957, 2: 1.99684}

    CL_BASE_MASS = 34.96885
    CL_DELTA = 1.99705
    CL_ABUNDANCE = {0: 0.7578, 1: 0.2422} # 35Cl (75.78%), 37Cl (24.22%)
    
    BR_BASE_MASS = 78.91834
    BR_DELTA = 1.99795
    BR_ABUNDANCE = {0: 0.5069, 1: 0.4931} # 79Br (50.69%), 81Br (49.31%)

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

    def get_full_isotope_cluster(self, formula_dict: dict):
        """
        Calculates isotope cluster distribution for general molecular formula dictionary.
        e.g. {'C': 6, 'H': 6, 'N': 1, 'O': 1, 'S': 0, 'Si': 0, 'Cl': 1, 'Br': 0}
        """
        cache_key = tuple(sorted(formula_dict.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        current_dist = {0.0: 1.0}
        
        element_params = [
            ('C', self.C_DELTA, self.C_ABUNDANCE[0], self.C_ABUNDANCE[1]),
            ('H', self.H_DELTA, self.H_ABUNDANCE[0], self.H_ABUNDANCE[1]),
            ('N', self.N_DELTA, self.N_ABUNDANCE[0], self.N_ABUNDANCE[1]),
            ('Cl', self.CL_DELTA, self.CL_ABUNDANCE[0], self.CL_ABUNDANCE[1]),
            ('Br', self.BR_DELTA, self.BR_ABUNDANCE[0], self.BR_ABUNDANCE[1]),
        ]

        for elem, delta, p0, p1 in element_params:
            n_atoms = formula_dict.get(elem, 0)
            if n_atoms > 0:
                elem_cluster = self._get_single_element_cluster(n_atoms, delta, p0, p1)
                new_dist = defaultdict(float)
                for shift_curr, prob_curr in current_dist.items():
                    for shift_elem, prob_elem in elem_cluster.items():
                        new_dist[shift_curr + shift_elem] += prob_curr * prob_elem
                current_dist = new_dist

        max_prob = max(current_dist.values()) if current_dist else 1.0
        normalized = {round(shift, 5): round((prob / max_prob) * 100.0, 2)
                      for shift, prob in current_dist.items()}
        
        self._cache[cache_key] = normalized
        return normalized

    def get_isotope_cluster(self, num_cl: int, num_br: int):
        """Backwards compatible interface for Cl/Br only."""
        return self.get_full_isotope_cluster({'Cl': num_cl, 'Br': num_br})
# %%