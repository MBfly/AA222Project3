import numpy as np
import matplotlib.pyplot as plt
from project2_py.helpers import Simple2

def jacobian(c_func, x, eps=1e-6):
    x = np.asarray(x, float)
    m = len(c_func(x))
    J = np.zeros((m, x.size))
    for j in range(x.size):
        dx = np.zeros_like(x); dx[j] = eps
        J[:, j] = (c_func(x + dx) - c_func(x - dx)) / (2*eps)
    return J

def augment_lagrangian_trace(x0, f, g, c, rho, alpha, n_iter):
    x   = x0.astype(float)
    lam = np.zeros(len(c(x)))
    objs = np.zeros(n_iter+1)
    viol = np.zeros(n_iter+1)

    for it in range(n_iter+1):
        ci = c(x)
        objs[it] = f(x)
        viol[it] = np.max(np.maximum(ci, 0.0))

        # don’t take a step after the last record
        if it == n_iter:
            break

        # ∇ₓL = ∇f + ∑ (λᵢ + ρ cᵢ) ∇cᵢ
        J = jacobian(c, x)
        grad = g(x).copy()
        for i in range(len(ci)):
            grad += (lam[i] + rho * ci[i]) * J[i]

        x   = x - alpha * grad
        lam = lam + rho * ci

    return objs, viol

def penalty_trace(x0, f, g, c, rho, alpha, n_iter):
    x = x0.astype(float)
    objs = np.zeros(n_iter+1)
    viol = np.zeros(n_iter+1)

    for it in range(n_iter+1):
        ci = c(x)
        objs[it] = f(x)
        viol[it] = np.max(np.maximum(ci, 0.0))

        if it == n_iter:
            break

        # ∇P = ∇f + ρ ∑ max(0,cᵢ) ∇cᵢ
        J = jacobian(c, x)
        grad = g(x).copy()
        for i in range(len(ci)):
            if ci[i] > 0:
                grad += rho * ci[i] * J[i]

        x = x - alpha * grad

    return objs, viol

def plot_simple2_diagnostics(n_iter=100,
                             rho=10.0,
                             alpha=1e-2,
                             initials=None):
    prob = Simple2()
    f, g, c = prob.f, prob.g, prob.c

    if initials is None:
        initials = [
            np.array([-2.0,  2.0]),
            np.array([ 2.0,  2.0]),
            np.array([ 2.0, -2.0]),
        ]

    # collect histories
    all_al_objs = []
    all_al_viol = []
    all_qp_objs = []
    all_qp_viol = []

    for x0 in initials:
        o_al, v_al = augment_lagrangian_trace(x0, f, g, c, rho, alpha, n_iter)
        o_qp, v_qp = penalty_trace(x0, f, g, c, rho, alpha, n_iter)
        all_al_objs.append(o_al)
        all_al_viol.append(v_al)
        all_qp_objs.append(o_qp)
        all_qp_viol.append(v_qp)

        # sanity check:
        assert len(o_al) == n_iter+1, f"AL objs length = {len(o_al)}"
        assert len(v_al) == n_iter+1, f"AL viol length = {len(v_al)}"
        assert len(o_qp) == n_iter+1, f"QP objs length = {len(o_qp)}"
        assert len(v_qp) == n_iter+1, f"QP viol length = {len(v_qp)}"

    iters = np.arange(n_iter+1)

    # 1) AugLag objective
    plt.figure()
    for idx, o in enumerate(all_al_objs):
        plt.plot(iters, o, '-o', markersize=4, label=f'start {tuple(initials[idx])}')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('simple2 — Augmented Lagrangian: Objective vs Iter')
    plt.legend()
    plt.grid(True)

    # 2) QuadPen objective
    plt.figure()
    for idx, o in enumerate(all_qp_objs):
        plt.plot(iters, o, '-o', markersize=4, label=f'start {tuple(initials[idx])}')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('simple2 — Quadratic Penalty: Objective vs Iter')
    plt.legend()
    plt.grid(True)

    # 3) AugLag violation
    plt.figure()
    for idx, v in enumerate(all_al_viol):
        plt.plot(iters, v, '-o', markersize=4, label=f'start {tuple(initials[idx])}')
    plt.xlabel('Iteration')
    plt.ylabel('max constraint violation')
    plt.title('simple2 — Augmented Lagrangian: Violation vs Iter')
    plt.legend()
    plt.grid(True)

    # 4) QuadPen violation
    plt.figure()
    for idx, v in enumerate(all_qp_viol):
        plt.plot(iters, v, '-o', markersize=4, label=f'start {tuple(initials[idx])}')
    plt.xlabel('Iteration')
    plt.ylabel('max constraint violation')
    plt.title('simple2 — Quadratic Penalty: Violation vs Iter')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    # now request a long run; you should see all 101 iterations
    plot_simple2_diagnostics(n_iter=100)
