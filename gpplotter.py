#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Theoretical mean function
def theoretical_efficiency(p):
    return 1.0 / (1.0 + 1.0 / p)

# Load CSV data
data = np.loadtxt('simulated_efficiency.csv', delimiter=',')
p = data[:, 0]
y_noisy = data[:, 1]
# Compute theoretical mean and residuals
mu_th = theoretical_efficiency(p)
y_residual = y_noisy - mu_th

# Prepare GP
# Fixed RBF kernel (no optimization of length scale)
kernel = RBF(length_scale=8.0, length_scale_bounds='fixed')
# Alpha = known noise variance (0.01^2), optimizer=None disables hyperparam tuning
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.01**2,
                              optimizer=None,
                              normalize_y=False)

# Fit GP on 1D input p
X = p.reshape(-1, 1)
gp.fit(X, y_residual)

# Prediction grid
p_pred = np.linspace(np.min(p), np.max(p), 500)
X_pred = p_pred.reshape(-1, 1)
# GP predictive mean and standard deviation of residuals
delta_mu, sigma = gp.predict(X_pred, return_std=True)
# Add back theoretical mean to obtain η predictions
mu_pred = delta_mu + theoretical_efficiency(p_pred)

# 95% confidence interval
ci = 1.96 * sigma

# Plotting
plt.figure(figsize=(8, 6))
# Raw noisy data
plt.scatter(p, y_noisy, color='k', label='Noisy data')
# Theoretical curve
plt.plot(p_pred, theoretical_efficiency(p_pred), 'r--', label='Theoretical $\eta(p)$')
# GP posterior mean
plt.plot(p_pred, mu_pred, 'b-', label='GP mean')
# Confidence band
plt.fill_between(p_pred, mu_pred - ci, mu_pred + ci, color='blue', alpha=0.2, label='95% CI')

plt.xlabel('Minimum separation $p$')
plt.ylabel('Efficiency $\eta$')
plt.title('Gaussian Process Fit of Wind Farm Efficiency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
