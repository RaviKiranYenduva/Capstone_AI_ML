
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import os

# --- Bayesian Optimization Core Definitions ---
# Define the kernel for the Gaussian Process
kernel = Matern(length_scale=0.1, nu=2.5)

# Define the UCB acquisition function
def ucb_acquisition(x, gpr, kappa=2.5):
    x = x.reshape(1, -1)
    mu, sigma = gpr.predict(x, return_std=True)
    return mu + kappa * sigma

# Maximize the acquisition function by minimizing its negative
def negative_ucb_acquisition(x, gpr, kappa=2.5):
    return -ucb_acquisition(x, gpr, kappa)

# Number of random starts for optimization
n_starts = 100

# Define the base path for the functions' data in Google Drive
drive_base_path = '/content/drive/MyDrive/AI ML Imperial/CapstoneProj/Week1/'

# --- Helper function to generate query points for a specific function ---
def generate_next_query_point_for_function(function_num, kernel, negative_ucb_acquisition, n_starts):
    function_path = f'{drive_base_path}function_{function_num}'

    # Load initial inputs and outputs for this function
    initial_inputs = np.load(f'{function_path}/initial_inputs.npy')
    initial_output = np.load(f'{function_path}/initial_outputs.npy')

    # Re-initialize the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                   optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10,
                                   normalize_y=True, random_state=0)

    # Fit the GPR to the initial data
    gpr.fit(initial_inputs, initial_output)

    # Define the search space bounds (assuming all inputs are between 0 and 1)
    bounds = [(0.0, 1.0)] * initial_inputs.shape[1]

    # Find the next query point by optimizing the acquisition function
    x_next = None
    best_acquisition_value = -np.inf

    for _ in range(n_starts):
        x0 = np.random.uniform(0, 1, size=initial_inputs.shape[1])
        res = minimize(negative_ucb_acquisition, x0=x0, bounds=bounds,
                       args=(gpr, 2.5), method='L-BFGS-B')

        if -res.fun > best_acquisition_value:
            best_acquisition_value = -res.fun
            x_next = res.x

    # Format the query point as required (six decimal places)
    formatted_query = '-'.join([f'{val:.6f}' for val in x_next])
    return formatted_query

# --- Main execution to generate and print all query points ---
if __name__ == '__main__':
    print("Generating consolidated query points for all functions:")
    all_function_query_points = {}
    for i in range(1, 9):
        query_point = generate_next_query_point_for_function(
            i, kernel, negative_ucb_acquisition, n_starts
        )
        all_function_query_points[f'Function {i}'] = query_point

    print("
Consolidated Query Points for All Functions:")
    for func_name, query in all_function_query_points.items():
        print(f"* {func_name}: {query}")
