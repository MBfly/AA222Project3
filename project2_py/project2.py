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
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    """
    Constrained optimization via quadratic exterior penalty + gradient descent.
    """

    # problem-specific tuning
    if prob == 'simple1':
        μ = 1000.0
        α = 1e-3
    elif prob == 'simple2':
        μ = 10.0
        α = 1e-5
    elif prob == 'simple3':
        μ = 15.0
        α = 1e-2
    elif prob == 'secret1':
        μ = 10.0
        α = 1e-2
    elif prob == 'secret2':
        μ = 10.0
        α = 1e-2

    h = 1e-6                             # finite-difference step for ∇c
    x = x0.copy()
    # initialize best feasible
    c0 = c(x)
    f0 = f(x)
    if np.all(c0 <= 0):
        x_best, f_best = x.copy(), f0
    else:
        x_best, f_best = x0.copy(), np.inf

    # limit iterations so we don’t loop forever
    max_iters = max(100, n // 10)

    for _ in range(max_iters):
        if count() >= n:
            break

        # 1) compute ∇f
        grad = g(x)

        # 2) compute ∇ of penalty term: 2·μ·∑[c_i>0] c_i·∇c_i
        cvals = c(x)
        for i, ci in enumerate(cvals):
            if ci > 0:
                # finite-difference ∇c_i
                grad_ci = np.zeros_like(x)
                for j in range(len(x)):
                    if count() >= n:
                        break
                    xh = x.copy()
                    xh[j] += h
                    grad_ci[j] = (c(xh)[i] - ci) / h
                grad += 2 * μ * ci * grad_ci

        # 3) take a gradient-descent step on the penalized objective
        x_new = x - α * grad

        # 4) accept step only if the augmented objective decreases
        #    φ(x) = f(x) + μ·∑max(0,c_i)^2
        φ_old = f(x) + μ * np.sum(np.maximum(0, cvals)**2)
        if count() >= n:
            break
        c_new = c(x_new)
        φ_new = f(x_new) + μ * np.sum(np.maximum(0, c_new)**2)
        if φ_new < φ_old:
            x = x_new
        else:
            α *= 0.5   # backtrack

        # 5) if new x is feasible and better, record it
        if np.all(c_new <= 0):
            f_new = f(x)
            if f_new < f_best:
                f_best, x_best = f_new, x.copy()

        # 6) simple convergence test (small step)
        if np.linalg.norm(α * grad) < 1e-6:
            break

    return x_best