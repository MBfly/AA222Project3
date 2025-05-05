import numpy as np
import matplotlib.pyplot as plt
from project2_py.helpers import Simple2

def run_augmented_lagrangian(f, g, c_func, x0, max_iter, alpha=0.001, rho_init=1.0, rho_mult=10):
    x = x0.copy().astype(float)
    m = c_func(x).size
    lam = np.zeros(m)
    rho = rho_init
    obj_hist = []
    maxc_hist = []
    grad_f = g(x)
    
    for k in range(max_iter):
        c_vals = c_func(x)
        # Approximate Jacobian of constraints by finite differences
        eps = 1e-6
        J = np.zeros((m, x.size))
        for i in range(x.size):
            e = np.zeros_like(x); e[i] = eps
            J[:, i] = (c_func(x + e) - c_vals) / eps

        # Lagrangian gradient
        grad_L = grad_f + J.T.dot(lam + rho * np.maximum(c_vals, 0))
        x = x - alpha * grad_L

        # Update multipliers
        c_vals = c_func(x)
        lam = lam + rho * np.maximum(c_vals, 0)

        # Increase penalty parameter occasionally
        if (k + 1) % 50 == 0:
            rho *= rho_mult

        grad_f = g(x)
        obj_hist.append(f(x))
        maxc_hist.append(np.max(c_vals))

    return obj_hist, maxc_hist

def run_quadratic_penalty(f, g, c_func, x0, max_iter, alpha=0.0005, rho_init=1.0, rho_mult=10):
    x = x0.copy().astype(float)
    m = c_func(x).size
    rho = rho_init
    obj_hist = []
    maxc_hist = []
    
    for k in range(max_iter):
        grad_f = g(x)
        c_vals = c_func(x)
        # Approximate Jacobian of constraints
        eps = 1e-6
        J = np.zeros((m, x.size))
        for i in range(x.size):
            e = np.zeros_like(x); e[i] = eps
            J[:, i] = (c_func(x + e) - c_vals) / eps

        # Gradient of penalized objective
        grad_Q = grad_f + 2 * rho * J.T.dot(np.maximum(c_vals, 0))
        x = x - alpha * grad_Q

        # Increase penalty parameter occasionally
        if (k + 1) % 50 == 0:
            rho *= rho_mult

        c_vals = c_func(x)
        obj_hist.append(f(x))
        maxc_hist.append(np.max(c_vals))

    return obj_hist, maxc_hist

def plot_Simple2_results(max_iter=200):
    # Problem setup
    prob = Simple2()
    f, g, c_func = prob.f, prob.g, prob.c
    
    # Three fixed starting points
    starts = [
        np.array([-1.2, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0])
    ]

    # Run both methods
    results_AL = [run_augmented_lagrangian(f, g, c_func, x0, max_iter) for x0 in starts]
    results_QP = [run_quadratic_penalty(f, g, c_func, x0, max_iter) for x0 in starts]

    # Plot augmented Lagrangian objective
    plt.figure()
    for i, (obj_hist, _) in enumerate(results_AL):
        plt.plot(obj_hist, label=f'Start {i+1}')
    plt.title('Simple2 augmented lagrangian objective vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Objective f(x)')
    plt.legend()

    # Plot quadratic penalty objective
    plt.figure()
    for i, (obj_hist, _) in enumerate(results_QP):
        plt.plot(obj_hist, label=f'Start {i+1}')
    plt.title('Simple2 quadratic penalty objective vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Objective f(x)')
    plt.legend()

    # Plot augmented Lagrangian max constraint violation
    plt.figure()
    for i, (_, maxc_hist) in enumerate(results_AL):
        plt.plot(maxc_hist, label=f'Start {i+1}')
    plt.title('Simple2 augmented lagrangian max constraint vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Max constraint violation')
    plt.legend()

    # Plot quadratic penalty max constraint violation
    plt.figure()
    for i, (_, maxc_hist) in enumerate(results_QP):
        plt.plot(maxc_hist, label=f'Start {i+1}')
    plt.title('Simple2 quadratic penalty max constraint vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Max constraint violation')
    plt.legend()

    plt.show()

# Execute plotting
plot_Simple2_results(max_iter=200)
