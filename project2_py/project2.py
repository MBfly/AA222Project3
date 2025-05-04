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
    """
    A simple budget-aware randomized local search.
    Works for all problems (including secret1 and secret2).
    """
    rng = np.random.RandomState(42)

    x_best = np.array(x0, dtype=float)
    if np.any(c(x_best) > 0) and count() < n:
        for _ in range(500):
            if count() + 1 > n:
                break
            x_try = x_best + rng.randn(*x_best.shape)
            if np.all(c(x_try) <= 0):
                x_best = x_try
                break

    if count() + 1 > n:
        return x_best
    f_best = f(x_best)

    max_iters = max(1, n // 2)
    init_step = 1.0  

    for i in range(int(max_iters)):
        if count() + 2 > n:
            break

        step = init_step * (1 - i / max_iters)

        x_cand = x_best + step * rng.randn(*x_best.shape)

        c_vals = c(x_cand)   
        if np.any(c_vals > 0):
            continue

        f_cand = f(x_cand)   
        if f_cand < f_best:
            x_best, f_best = x_cand, f_cand

    return x_best
