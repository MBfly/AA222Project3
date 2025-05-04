import numpy as np

def optimize(f, g, c, x0, n, count, prob):

    x       = np.asarray(x0, float).copy()
    best_x  = x.copy()
    fx      = f(x)                           
    cx      = np.asarray(c(x), float)       
    if np.all(cx <= 0):
        best_f = fx
    else:
        best_f = np.inf

    dim     = x.size

    P       = 2.0        # initial penalty weight
    T       = 2.0        # initial temperature
    T_min   = 1e-3       
    alpha   = 0.95       

    def penalized(x, fx=None, cx=None):
        if fx is None:
            fx = f(x)                  
        if cx is None:
            cx = np.asarray(c(x), float)  
        penalty = P * np.sum(np.maximum(0, cx)**2)
        return fx + penalty, fx, cx

    while count() + 2 <= n and T > T_min:
        x_new = x + T * np.random.randn(dim)

        F_new, f_new, c_new = penalized(x_new)  
        F_cur, f_cur, c_cur = penalized(x, fx, cx)  

        Δ = F_new - F_cur
        if Δ < 0 or np.random.rand() < np.exp(-Δ / T):
            x, fx, cx = x_new, f_new, c_new
            F_cur = F_new
            if np.all(c_new <= 0) and f_new < best_f:
                best_f  = f_new
                best_x  = x_new.copy()

        T *= alpha
        # penalty increase
        P *= 1.5

    return best_x
