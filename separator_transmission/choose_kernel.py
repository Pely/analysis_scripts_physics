from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic, ExpSineSquared, WhiteKernel
from sklearn.model_selection import GridSearchCV
import numpy as np



X = np.array(data).reshape(-1, 3)  # Reshape to create the input array
transmission = X[:, 2]  # Transmission values
X = X[:, :2]  # Energy and angle as input features
y = transmission

# Define the kernels to be tested
kernels = [RBF(), Matern(), DotProduct(), RationalQuadratic(), ExpSineSquared()]

# Define the parameter grid for each kernel
param_grids = [
    {'kernel__length_scale': [2, 10, 100]},
    {'kernel__length_scale': [2, 10, 100], 'kernel__nu': [0.5, 1.5, 2.5]},
    {},
    {'kernel__length_scale': [2, 10, 100], 'kernel__alpha': [0.01, 0.5, 1.0]},
    {'kernel__length_scale': [2, 10, 100], 'kernel__periodicity': [5, 10, 100]},
]

# Define the parameter grid for the GaussianProcessRegressor
param_grid = {
    'kernel': kernels,
    'alpha': [1e-7, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 100],
}

# Define the GaussianProcessRegressor
gp = GaussianProcessRegressor()

# Define the GridSearchCV
gs = GridSearchCV(gp, param_grid=param_grid, cv=4)

# Fit the GridSearchCV
gs.fit(X, y)

# Get the best estimator
best_estimator = gs.best_estimator_

# Print the best estimator
print("Results!!!!")
print(best_estimator)