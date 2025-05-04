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

def optimize(f, g, c, x0, n, count, prob=None):
    """
    Args:
        f (function): Function to be optimized.
        g (function): Gradient function for f.
        c (function): Function evaluating constraints (c_i(x) <= 0).
        x0 (np.array): Initial position to start from.
        n (int): Maximum number of evaluations (f or c cost 1, g costs 2).
        count (function): Returns current evaluation count.
        prob (str): Ignored.
    Returns:
        x_best (np.array): Best feasible solution found.
    """
    # Initialize
    x = np.copy(x0)
    x_best = np.copy(x0)
    # Evaluate initial point
    f_best = f(x_best)
    # Step size and tolerance
    alpha = 1.0
    tol = 1e-6

    # Main gradient descent loop with feasibility backtracking
    while count() + 2 <= n:
        # Compute gradient of f (costs 2 evaluations)
        grad = g(x)
        # Propose new point
        x_new = x - alpha * grad

        # Ensure feasibility via backtracking line search
        backtracks = 0
        while count() < n and np.any(c(x_new) > 0):
            alpha *= 0.5
            x_new = x - alpha * grad
            backtracks += 1
            if backtracks > 10 or alpha < 1e-8:
                break
        # If still infeasible, stop
        if np.any(c(x_new) > 0):
            break

        # Evaluate objective at candidate (costs 1 evaluation)
        f_new = f(x_new)

        # Accept if improvement
        if f_new < f_best:
            x = x_new.copy()
            x_best = x_new.copy()
            f_best = f_new
            # Try increasing step size
            alpha *= 1.2
        else:
            # Reduce step size if no improvement
            alpha *= 0.5

        # Convergence check on step norm
        if np.linalg.norm(alpha * grad) < tol:
            break

    return x_best
