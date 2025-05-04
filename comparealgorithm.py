import numpy as np
import matplotlib.pyplot as plt
from project2_py.helpers import Simple1, Simple2

def jacobian(c_func, x, eps=1e-6):
    x = np.asarray(x, float)
    m = len(c_func(x))
    J = np.zeros((m, x.size))
    for j in range(x.size):
        dx = np.zeros_like(x); dx[j] = eps
        J[:, j] = (c_func(x + dx) - c_func(x - dx)) / (2*eps)
    return J

def optimize_augmented_lagrangian(x0, f, g, c,
                                  rho=10.0, alpha=1e-2, n_iter=50):
    x = x0.astype(float)
    m = len(c(x))
    lam = np.zeros(m)
    path = [x.copy()]
    for _ in range(n_iter):
        ci = c(x); J = jacobian(c, x)
        grad = g(x).copy()
        for i in range(m):
            grad += (lam[i] + rho * ci[i]) * J[i]
        x = x - alpha * grad
        lam += rho * ci
        path.append(x.copy())
    return np.array(path)

def optimize_penalty(x0, f, g, c,
                     rho=10.0, alpha=1e-2, n_iter=50):
    x = x0.astype(float)
    path = [x.copy()]
    for _ in range(n_iter):
        ci = c(x); J = jacobian(c, x)
        grad = g(x).copy()
        for i in range(len(ci)):
            if ci[i] > 0:
                grad += rho * ci[i] * J[i]
        x = x - alpha * grad
        path.append(x.copy())
    return np.array(path)

def plot_paths(prob_name,
               initials=None,
               bounds=(-3, 3),
               resolution=200,
               contour_levels=50):
    name = prob_name.lower()
    if name == 'simple1':
        prob = Simple1()
    elif name == 'simple2':
        prob = Simple2()
    else:
        raise ValueError("prob_name must be 'simple1' or 'simple2'")
    f, g, c = prob.f, prob.g, prob.c

    if initials is None:
        initials = [
            np.array([-2.0,  2.0]),
            np.array([ 2.0,  2.0]),
            np.array([ 2.0, -2.0]),
        ]

    x_min, x_max = bounds
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(x_min, x_max, resolution)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            Z[j, i] = f(np.array([xi, yj]))  # explicit np.array

    # 4) plot
    fig, ax = plt.subplots(figsize=(7,6))
    cs = ax.contour(X, Y, Z, levels=contour_levels, cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=8)

    methods = {
        'Augmented Lagrangian': optimize_augmented_lagrangian,
        'Quadratic Penalty'   : optimize_penalty
    }

    for method_name, solver in methods.items():
        for x0 in initials:
            path = solver(x0, f, g, c)
            ax.plot(
                path[:,0], path[:,1], '-o', markersize=3,
                label=f'{method_name}, start {tuple(x0)}'
            )

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'All trajectories on {prob_name}')
    ax.legend(loc='upper right', fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_paths('simple1')
    plot_paths('simple2')
