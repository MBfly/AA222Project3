#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing, minimize

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

# Objective and separation
def min_separation(x, n):
    pts = x.reshape(n, 2)
    if not is_valid(pts):
        return -1e6
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d)

def objective(x, n):
    return -min_separation(x, n)

# Optimization
def optimize_for_n(n):
    bounds = [(0.0, 10.0)] * (2 * n)
    # global search
    result = dual_annealing(lambda x: objective(x, n), bounds,
                             maxiter=5000)
    # local polish
    res_local = minimize(lambda x: objective(x, n), result.x,
                         method='Nelder-Mead',
                         options={'maxiter':2000, 'xatol':1e-6, 'fatol':1e-6})
    p_best = min_separation(res_local.x, n)
    return res_local.x.reshape(n, 2), p_best

# Plotting
def plot_layout(pts, n):
    fig, ax = plt.subplots(figsize=(6,6))
    # Plot forbidden regions
    # Middle circle
    circle = plt.Circle((5,5), 2, color='red', alpha=0.3)
    ax.add_patch(circle)
    # Bottom-left semicircle (y >= 0)
    from matplotlib.patches import Wedge
    semi1 = Wedge((0,0), 1, 0, 180, color='red', alpha=0.3)
    ax.add_patch(semi1)
    # Top-left semicircle (y <= 10)
    semi2 = Wedge((0,10), 1, 180, 360, color='red', alpha=0.3)
    ax.add_patch(semi2)
    # Plot turbines
    xs, ys = pts[:,0], pts[:,1]
    ax.scatter(xs, ys, c='blue', s=50)
    for i,(x,y) in enumerate(pts):
        ax.text(x, y, str(i+1), color='white', fontsize=8,
                ha='center', va='center')
    # Formatting
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Wind Turbine Layout (n={n})")
    ax.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot optimized wind farm layout')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of turbines (2-10)')
    args = parser.parse_args()
    n = args.n
    if not (2 <= n <= 10):
        raise ValueError('n must be between 2 and 10')

    print(f"Optimizing layout for n={n}...")
    pts, p = optimize_for_n(n)
    print(f"Done. Max min-separation p = {p:.4f}")
    plot_layout(pts, n)

if __name__ == '__main__':
    main()
