from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic, ExpSineSquared, WhiteKernel
from sklearn.model_selection import GridSearchCV
import numpy as np

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
    2.00, 10.0, 0.170,
    0.00, 20.0, 0.294,  
    3.00, 20.0, 0.159,  
    1.50, 20.0, 0.250,  
    -1.50, 20.0, 0.258, 
    -3.00, 20.0, 0.199  
]

errors = [
    0.25, 0.5, 0.0175,
    0.25, 0.5, 0.0045,
    0.25, 0.5, 0.0104,
    0.25, 0.5, 0.0021,
    0.25, 0.5, 0.0072,
    0.25, 0.5, 0.0065,
    0.25, 1.0, 0.0429,
    0.25, 1.0, 0.0428,
    0.25, 1.0, 0.0296,
    0.25, 1.0, 0.0111,
    0.25, 1.0, 0.0283,
    0.25, 1.0, 0.0283,
    0.25, 1.0, 0.0244,
    0.25, 1.0, 0.0170,
    0.25, 1.0, 0.0221,
    0.25, 1.0, 0.0226,
    0.25, 1.0, 0.0194
]

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