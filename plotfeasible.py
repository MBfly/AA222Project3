import numpy as np
import matplotlib.pyplot as plt
from project2_py.helpers import Simple1, Simple2

def plot_feasible_region(prob_name, resolution=300, contour_levels=50):

    name = prob_name.lower()
    if name == 'simple1':
        prob = Simple1()
        x_bounds = (0.0, 1.2)
        y_bounds = (0.0, 1.2)
    elif name == 'simple2':
        prob = Simple2()
        x_bounds = (-1.0, 2.0)
        y_bounds = (-1.0, 3.0)

    f = prob.f
    c = prob.c

    xs = np.linspace(*x_bounds, resolution)
    ys = np.linspace(*y_bounds, resolution)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])

    Z = np.array([f(p) for p in points]).reshape(X.shape)
    C = np.array([c(p) for p in points])        
    feasible = np.all(C <= 0, axis=1).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, Z, levels=contour_levels, cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=8)

    ax.contourf(
        X, Y, feasible,
        levels=[-0.5, 0.5, 1.5],
        colors=['none', 'lightgreen'],
        alpha=0.4
    )

    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'Feasible Region for {prob_name}')
    plt.show()


if __name__ == '__main__':
    plot_feasible_region('simple1')
    plot_feasible_region('simple2', resolution=400, contour_levels=60)
