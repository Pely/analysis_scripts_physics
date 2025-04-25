import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

data = [
    0.00, 4.0, 0.699,
    1.50, 4.0, 0.325,
    -1.5, 4.0, 0.737,
    -3.0, 4.0, 0.248,
    1.00, 4.0, 0.503,
    0.50, 4.0, 0.630,
    0.00, 10.0, 0.277,
    -1.5, 10.0, 0.304,
    -3.0, 10.0, 0.184,
    3.00, 10.0, 0.060,
    1.50, 10.0, 0.173,
    2.00, 10.0, 0.170
]
errors = [
    0.25, 0.5, 0.0175,
    0.25, 0.5, 0.0045,
    0.25, 0.5, 0.0104,
    0.25, 0.5, 0.0021,
    0.25, 0.5, 0.0072,
    0.25, 0.5, 0.0065,
    0.25, 0.5, 0.0429,
    0.25, 0.5, 0.0428,
    0.25, 0.5, 0.0296,
    0.25, 0.5, 0.0111,
    0.25, 0.5, 0.0283,
    0.25, 0.5, 0.0283
]

# Split data into input (X) and output (y)
X = np.array(data).reshape(-1, 3)  # Reshape to create the input array
transmission = X[:, 2]  # Transmission values
X = X[:, :2]  # Energy and angle as input features

# Create an array for errors
error_energy = errors[0::3]  # Energy errors
error_angle = errors[1::3]  # Angle errors
error_transmission = errors[2::3]  # Transmission errors

# Perform PCA to reduce dimensionality (optional)
n_components = 2  # Choose the number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Define a kernel
#kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 2e6)) + 1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 2e6)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e3))
# Define a kernel for the Gaussian Process
#kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
kernel = 1.0 * RBF(length_scale=0.50, length_scale_bounds=(0.1, 5.0)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 10))

# Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=np.square(error_transmission))

gp.fit(X_pca, transmission)  # Use transformed transmission as the output 
#gp.fit(X, transmission)

# Create a 3D plot to visualize GP predictions 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the range of energy and angle values
energy_range = np.linspace(-4.0, 4.0, 200)
angle_range = np.linspace(0, 15, 150)
X_mesh = np.array(np.meshgrid(energy_range, angle_range)).T.reshape(-1, 2)

# Perform PCA transformation on the meshgrid for prediction
X_pca_for_prediction = pca.transform(X_mesh)

# Make predictions with GP
#transmission_pred, sigma = gp.predict(X_mesh, return_std=True)

# Make predictions with GP based on the reduced-dimensional representation
transmission_pred, sigma = gp.predict(X_pca_for_prediction, return_std=True)

# Inverse transform the predictions
transmission_pred = 100*transmission_pred
transmission_pred_sigma = 100*sigma

# Plot the GP predictions
ax.plot_trisurf(X_mesh[:, 0], X_mesh[:, 1], transmission_pred, cmap='viridis', linewidth=0.1)
ax.set_xlabel('Energy offset (%)')
ax.set_ylabel('Angle (mrad)')
ax.set_zlabel('Transmission (%)')
ax.set_zlim(0,100)
plt.title('3D Plot of GP Predictions')
plt.show()

# Scatter plot with energy on the x-axis and angle on the y-axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

# Plot 1: Scatter plot with transmission as color
scatter1 = axes[0].scatter(X_mesh[:, 0], X_mesh[:, 1], marker='o', s=8, c=transmission_pred, cmap='viridis')
cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Transmission (%)')
scatter1.set_clim(0,100) 
cbar1.set_ticks(np.arange(0, 110, 10))
axes[0].set_xlabel('Energy offset (%)')
axes[0].set_ylabel('Angle (mrad)')
axes[0].set_title('Scatter Plot of Energy vs Angle with Transmission Contour')
scatter_data=axes[0].scatter(X[:, 0], X[:, 1], c=100*transmission, cmap='viridis', marker='s', s=50, edgecolor='black', linewidth=1)
axes[0].errorbar(X[:, 0], X[:, 1], xerr=np.array(error_energy), yerr=np.array(error_angle), fmt='none', color='yellow', capsize=2, alpha=0.5 )
scatter_data.set_clim(0,100)

# Plot 2: Scatter plot with uncertainties as color
scatter2 = axes[1].scatter(X_mesh[:, 0], X_mesh[:, 1], marker='o', s=8, c=transmission_pred_sigma, cmap='viridis')
cbar2 = plt.colorbar(scatter2, ax=axes[1], label='Uncertainty (%)')
scatter2.set_clim(0,50) 
cbar2.set_ticks(np.arange(0, 55, 5))
axes[1].set_xlabel('Energy offset (%)')
axes[1].set_ylabel('Angle (mrad)')
axes[1].set_title('Scatter Plot of Energy vs Angle with Uncertainty Contour')
plt.tight_layout()
plt.show()
