#!/usr/bin/env python3
import numpy as np
from scipy.optimize import dual_annealing, minimize
import csv
import time

# Forbidden region checks
def in_middle_circle(x, y, center=(5, 5), r=2):
    return (x - center[0])**2 + (y - center[1])**2 <= r**2

def in_semi_circle(x, y, center, r, require_above):
    dx = x - center[0]; dy = y - center[1]
    inside = dx*dx + dy*dy <= r*r
    return inside and ((y >= center[1]) if require_above else (y <= center[1]))

def is_valid(pts):
    for x, y in pts:
        if not (0 <= x <= 10 and 0 <= y <= 10):
            return False
        if in_middle_circle(x, y):
            return False
        if in_semi_circle(x, y, (0, 0), 1, True):
            return False
        if in_semi_circle(x, y, (0, 10), 1, False):
            return False
    return True

# Minimum separation calculation
def min_separation(x, n):
    pts = x.reshape(n, 2)
    if not is_valid(pts):
        return -1e6
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d)

# Objective: maximize p => minimize -p
def objective(x, n):
    return -min_separation(x, n)

# Optimize for a given n with extensive search
def optimize_for_n(n, runs=10):
    bounds = [(0.0, 10.0)] * (2 * n)
    best_p = -np.inf
    best_x = None

    for i in range(runs):
        start = time.time()
        print(f"  Run {i+1}/{runs} for n={n}...")
        # Extensive global search
        result = dual_annealing(
            lambda x: objective(x, n),
            bounds,
            maxiter=10000,
            initial_temp=5230.0,
            restart_temp_ratio=1e-6,
            visit=2.72,
            accept=-5.0
        )
        # Deep local refinement
        res_local = minimize(
            lambda x: objective(x, n),
            result.x,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-9,
                'fatol': 1e-9,
                'disp': False
            }
        )
        p_val = min_separation(res_local.x, n)
        elapsed = time.time() - start
        print(f"    p={p_val:.6f} (time: {elapsed:.1f}s)")
        if p_val > best_p:
            best_p = p_val
            best_x = res_local.x

    return best_p, best_x

# Main execution
def main():
    results = []
    for n in range(2, 11):
        print(f"Optimizing for n={n} turbines:")
        p_opt, _ = optimize_for_n(n, runs=10)
        print(f"=> Best p for n={n}: {p_opt:.6f}\n")
        results.append((n, p_opt))

    with open('windfarm_efficiency.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for n, p in results:
            writer.writerow([n, p])
    print("Results saved to windfarm_efficiency.csv")

if __name__ == '__main__':
    main()
