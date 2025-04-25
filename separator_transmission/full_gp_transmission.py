import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic, ExpSineSquared, WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_csv('transmission_data_an.csv')
theta = df['theta'].values
theta_sig = df['theta_sig'].values
phi = df['phi'].values
phi_sig = df['phi_sig'].values
de = df['de'].values
transmission = df['transm'].values
transmission_sig = df['transm_sig'].values
input = np.column_stack((theta, theta_sig, phi, phi_sig, de))


def choose_kernel(X, Y, alpha):
    kernels = [RBF(), Matern(), DotProduct(), RationalQuadratic(), ExpSineSquared()]
    kernel_options = [
    (RBF(), {
        'gp__kernel': [RBF(length_scale=l) for l in [1, 5, 10, 100, 200]],
    }),
    (Matern(), {
        'gp__kernel': [Matern(length_scale=l, nu=n) for l in [1, 5, 10, 100, 200] for n in [0.5, 1.5, 2.5]],
    }),
    (DotProduct(), {
        'gp__kernel': [DotProduct()],
    }),
    (RationalQuadratic(), {
        'gp__kernel': [RationalQuadratic(length_scale=l, alpha=a) for l in [1, 5, 10, 100, 200] for a in [0.01, 0.5, 1.0]],
    }),
    (ExpSineSquared(), {
        'gp__kernel': [ExpSineSquared(length_scale=l, periodicity=p) for l in [1, 5, 10, 100, 200] for p in [5, 10, 100]],
    })]

    param_grids = []
    for kernel, kernel_grid in kernel_options:
        grid = {'gp__alpha': [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 100]}
        grid.update(kernel_grid)
        param_grids.append(grid)
    
    pipe = Pipeline([('gp', GaussianProcessRegressor())])
    # Let alpha be passed in from outside, but still test multiple values
    search = GridSearchCV(pipe, param_grid=param_grids, cv=4, n_jobs=-1)
    search.fit(X, Y)

    # Get the best estimator
    print("Best Estimator:")
    print(search.best_estimator_)
    return search.best_estimator_

def gaussian_process(X,transmission_true,transmission_sigma):
   
    kernel = RationalQuadratic(length_scale=10, length_scale_bounds=(0.1, 1000.0)) #+ WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=np.square(transmission_sigma))

    gp.fit(X, transmission_true)

    transm_pred, transm_sig_pred = gp.predict(X, return_std=True)
    return(transm_pred, transm_sig_pred)

def plot_gp_results(X, transmission_true, transmission_pred, transmission_sigma):
    theta = X[:, 0]
    phi = X[:, 2]
    de = X[:, 4]
    
    # Plot 1: Energy offset vs Transmission (with uncertainty)
    plt.figure(figsize=(10, 5))
    # plt.errorbar(de, transmission_true, yerr=transmission_sigma, fmt='o', label='True', alpha=0.6)
    plt.scatter(de, transmission_pred, color='r', s=30, label='GP Prediction', marker='x')
    plt.xlabel('Energy offset (de)')
    plt.ylabel('Transmission (%)')
    plt.title('Energy vs Transmission')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Theta vs Phi with predicted transmission as color
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(theta, phi, c=transmission_pred, cmap='viridis', s=60, edgecolor='k', alpha=0.8)
    plt.colorbar(scatter, label='Predicted Transmission (%)')
    plt.xlabel('Theta')
    plt.ylabel('Phi')
    plt.title('Theta vs Phi Colored by Transmission')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# best_gp = choose_kernel(input, transmission, alpha=transmission_sig ** 2)
transmission_pred=gaussian_process(input,transmission,transmission_sig)
plot_gp_results(input,transmission,transmission_pred,transmission_sig)
