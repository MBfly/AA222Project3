#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

def optimize(f, g, c, x0, n, count, prob):
    rng = np.random.RandomState(42)

    # 1) Start from x0; if it's infeasible, try to find a nearby feasible point
    x_best = np.array(x0, dtype=float)
    if np.any(c(x_best) > 0) and count() < n:
        for _ in range(500):
            if count() + 1 > n:
                break
            # small random shake around x0
            x_try = x_best + rng.randn(*x_best.shape)
            if np.all(c(x_try) <= 0):
                x_best = x_try
                break

    # 2) Evaluate its objective (if budget remains)
    if count() + 1 > n:
        return x_best
    f_best = f(x_best)

    # 3) Randomized local search with shrinking step‐size
    max_iters = max(1, n // 2)
    init_step = 1.0  # you can tune this

    for i in range(int(max_iters)):
        # stop if not enough budget for one constraint + one f‐eval
        if count() + 2 > n:
            break

        # shrink step‐size linearly
        step = init_step * (1 - i / max_iters)

        # propose a candidate
        x_cand = x_best + step * rng.randn(*x_best.shape)

        # check feasibility
        c_vals = c(x_cand)   # costs 1 eval
        if np.any(c_vals > 0):
            continue

        # check objective
        f_cand = f(x_cand)   # costs 1 eval
        if f_cand < f_best:
            x_best, f_best = x_cand, f_cand

    return x_best
