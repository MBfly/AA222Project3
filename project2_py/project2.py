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

    if prob != "secret2":
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

    elif prob == "secret2":
        p = {'alpha':0.005, 'mu0':5.0,  'mu_mul':1.5, 'outer':2, 'inner':5}

        x       = x0.copy()
        m       = c(x0).size
        lam     = np.zeros(m)       
        mu      = p['mu0']          
        eps_fd  = 1e-6               

        x_best  = x.copy()
        f_best  = np.inf

        for _ in range(p['outer']):
            for _ in range(p['inner']):
                if count() >= n:
                    break

                ci = c(x)           
                fx = f(x)           

                if np.all(ci <= 0) and fx < f_best:
                    f_best, x_best = fx, x.copy()

                grad_f = g(x)      

                cm = lam + mu * ci
                JcTcm = np.zeros_like(x)
                for j in range(x.size):
                    if count() >= n: 
                        break
                    xh = x.copy()
                    xh[j] += eps_fd
                    ci_h = c(xh)    
                    JcTcm[j] = np.dot(cm, (ci_h - ci) / eps_fd)

                gradL = grad_f + JcTcm

                x = x - p['alpha'] * gradL

            if count() >= n:
                break

            ci = c(x)              
            lam = np.maximum(lam + mu * ci, 0.0)
            mu  = mu * p['mu_mul']


    return x_best
