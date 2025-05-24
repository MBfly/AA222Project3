#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


theo = np.loadtxt('windfarm_efficiency_full.csv', delimiter=',')
n_arr = theo[:, 0].astype(int)
p_opt = theo[:, 1]
sim = np.loadtxt('simulated_efficiency.csv', delimiter=',')
p_sim = sim[:, 0]
eta_sim = sim[:, 1]

#----- Theoretical efficiency -----
def theoretical_efficiency(p):
    return 1.0 / (1.0 + 1.0 / p)

# Compute theoretical mean at simulated p and residuals
y_th_sim = theoretical_efficiency(p_sim)
y_res = eta_sim - y_th_sim

#----- Fit Gaussian Process on residuals -----
kernel = RBF(length_scale=8.0, length_scale_bounds='fixed')
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.01**2,
                              optimizer=None,
                              normalize_y=False)
X_sim = p_sim.reshape(-1, 1)
gp.fit(X_sim, y_res)

# Predict at optimal separations
X_pred = p_opt.reshape(-1, 1)
delta_mu, sigma = gp.predict(X_pred, return_std=True)
# GP-predicted efficiency mean and uncertainty
eta_pred = delta_mu + theoretical_efficiency(p_opt)

#----- Profit calculation -----
p_turbine = 3300 * 8670 * 0.75
v_kWh = 0.33
c_turbine = 5400000

profit_mean = n_arr * eta_pred * p_turbine * v_kWh - n_arr * c_turbine
profit_sigma = n_arr * p_turbine * v_kWh * sigma
ci = 1.96 * profit_sigma

# Theoretical profit (no uncertainty)
eta_th = theoretical_efficiency(p_opt)
profit_th = n_arr * eta_th * p_turbine * v_kWh - n_arr * c_turbine

#----- Plotting -----
plt.figure(figsize=(8, 6))
# GP mean profit
plt.plot(n_arr, profit_mean, 'b-', label='GP mean profit')
# 95% confidence band
plt.fill_between(n_arr, profit_mean - ci, profit_mean + ci,
                 color='blue', alpha=0.2, label='95% CI')
# Theoretical profit curve
plt.plot(n_arr, profit_th, 'r--', label='Theoretical profit')

plt.scatter(n_arr, profit_th, color='red')  # points for theoretical

plt.xlabel('Number of turbines (n)')
plt.ylabel('Annual profit ($)')
plt.title('Wind Farm Profit vs. Turbine Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
