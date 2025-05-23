#!/usr/bin/env python3
"""
Optimize wind turbine placement on a 10x10 plot to maximize minimum separation p.
Forbidden regions:
 1. Circular region center (5,5), radius 2
 2. Semi-circle center (0,0), radius 1 (y >= 0)
 3. Semi-circle center (0,10), radius 1 (y <= 10)

For each n in [2..10], uses multiple restarts of Differential Evolution plus local Nelder-Mead refinement
(with domain-enforced penalties) to maximize p = minimum pairwise distance between turbines.
Writes results to a CSV file: windfarm_efficiency.csv (no header row), with columns: n, p.

Dependencies:
  - numpy
  - scipy
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import csv

#--------------------------------------------------
# Forbidden region and domain checks
#--------------------------------------------------
def in_middle_circle(x, y, center=(5, 5), r=2):
    return (x - center[0])**2 + (y - center[1])**2 <= r**2


def in_semi_circle(x, y, center, r, require_above):
    dx = x - center[0]; dy = y - center[1]
    inside = dx*dx + dy*dy <= r*r
    return inside and ((y >= center[1]) if require_above else (y <= center[1]))


def is_valid_configuration(pts):
    """Return False if any turbine lies outside domain or in a forbidden region."""
    for x, y in pts:
        # domain check: inside [0,10]x[0,10]
        if x < 0 or x > 10 or y < 0 or y > 10:
            return False
        # forbidden regions
        if in_middle_circle(x, y):
            return False
        if in_semi_circle(x, y, (0, 0), 1, require_above=True):
            return False
        if in_semi_circle(x, y, (0, 10), 1, require_above=False):
            return False
    return True

#--------------------------------------------------
# Minimum separation and objective
#--------------------------------------------------
def min_separation(x, n):
    pts = x.reshape(n, 2)
    if not is_valid_configuration(pts):
        return -1e6  # strong penalty
    # compute pairwise distances
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def objective(x, n):
    # minimize negative separation
    return -min_separation(x, n)

#--------------------------------------------------
# Optimization per turbine count
#--------------------------------------------------
def optimize_for_n(n, restarts=5):
    bounds = [(0.0, 10.0)] * (2 * n)
    best_p = -np.inf
    best_x = None

    for seed in range(restarts):
        # Global Differential Evolution
        result = differential_evolution(
            objective,
            bounds,
            args=(n,),
            strategy='best1bin',
            popsize=30,
            maxiter=500,
            tol=1e-8,
            polish=False,
            seed=seed,
            disp=False
        )
        x0 = result.x
        # Local Nelder-Mead (with penalty inside objective)
        res_local = minimize(
            lambda x: objective(x, n),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol':1e-6, 'fatol':1e-6}
        )
        p_val = min_separation(res_local.x, n)
        if p_val > best_p:
            best_p, best_x = p_val, res_local.x

    return best_p, best_x

#--------------------------------------------------
# Main execution
#--------------------------------------------------
def main():
    results = []
    for n in range(2, 11):
        p_opt, _ = optimize_for_n(n, restarts=7)
        print(f"n={n}, p={p_opt:.6f}")
        results.append((n, p_opt))

    # write CSV
    with open('windfarm_efficiency.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for n, p in results:
            writer.writerow([n, p])
    print("Results saved to windfarm_efficiency.csv")

if __name__ == '__main__':
    main()
