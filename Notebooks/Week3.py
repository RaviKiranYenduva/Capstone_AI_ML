import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import ast # Added to parse string representations of lists/arrays

# =============================
# LOAD TXT DATA
# =============================
def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    processed_data = []
    for line in content.splitlines():
        stripped_line = line.strip()
        if not stripped_line: # Skip empty lines
            continue

        # Remove 'array(', 'np.float64(' and ')' which might be present from numpy's string representation
        # Also handle potential trailing ']' and leading '[' if present
        cleaned_line = stripped_line.replace('array(', '').replace('np.float64(', '').replace(')', '').strip('[] ')

        # Split by comma and process each item
        items = cleaned_line.split(',')
        parsed_record = []
        for item in items:
            trimmed_item = item.strip()
            if trimmed_item.lower() == 'nan':
                parsed_record.append(np.nan)
            elif trimmed_item:
                try:
                    parsed_record.append(float(trimmed_item))
                except ValueError:
                    print(f"Warning: Could not convert '{trimmed_item}' to float. Skipping or treating as NaN.")
                    parsed_record.append(np.nan) # Handle unexpected non-numeric values as NaN

        if parsed_record:
            processed_data.append(parsed_record)

    return np.array(processed_data)


# The input_data and output_data from files are not directly used for X_funcs and y_funcs
# in this revised approach, as the initial data is from blackbox_functions_data.
# These lines are kept if they are intended for loading future rounds of data, 
# but their filtering logic would need to be carefully revisited.
input_data = parse_data_file("inputs.txt")
output_data = parse_data_file("outputs.txt")

# =============================
# GROUP BY FUNCTION (Revised)
# =============================
# Initialize X_funcs and y_funcs with the initial single data point for each function
# This ensures all 8 functions are correctly processed.
X_funcs, y_funcs = [], []
for func_data in blackbox_functions_data:
    X_funcs.append(np.array([func_data['X']])) # Each X should be (1, dim)
    y_funcs.append(np.array([func_data['y']]).ravel()) # Each y should be (1,)

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 4000
    local_n = 2000

    if func_id == 2:
        noise = 1e-3
    if func_id == 5:
        xi = 0.001
    if func_id == 8:
        global_n = 6000

    return xi, noise, global_n, local_n

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id, original_dim):
    if len(X) == 0: # Handle case where there's no data for the function
        return np.random.uniform(0, 1, original_dim)

    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Step 1: Standardize features ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Step 2: Fit SVR (SVM regression) ----
    svr = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr.fit(X_scaled, y)
    y_svr_pred = svr.predict(X_scaled)
    residuals = y - y_svr_pred

    # ---- Step 3: GP on residuals for uncertainty ----
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(X_scaled, residuals)

    # ---- Step 4: Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    X_global_scaled = scaler.transform(X_global)

    best_x = X[np.argmax(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    # ---- Step 5: SVR + GP predictions ----
    svr_pred = svr.predict(X_candidates_scaled)
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)

    mu = svr_pred + gp_pred
    sigma = gp_std

    # ---- Step 6: Expected Improvement ----
    mu_sample = svr.predict(X_scaled) + gp.predict(X_scaled)
    mu_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return X_candidates[np.argmax(ei)]

# =============================
# FORMAT QUERY
# =============================
def format_query(x):
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP
# =============================
all_queries = []

for i in range(len(blackbox_functions_data)): # Iterate through the original functions
    print(f"\n===== Function {i+1} ====")
    
    # Get original X and y for the current function from the blackbox_functions_data list
    func_original_X = np.array([blackbox_functions_data[i]['X']])
    func_original_y = np.array([blackbox_functions_data[i]['y']])
    original_dim = len(blackbox_functions_data[i]['X'])

    # Use the pre-processed X and y from X_funcs and y_funcs
    X = X_funcs[i]
    y = y_funcs[i]

    print("Data points:", len(y))
    
    x_next = propose_next(X, y, i+1, original_dim) # Pass original_dim

    query = format_query(x_next)
    all_queries.append(query)

    print("Next query:", query)

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")