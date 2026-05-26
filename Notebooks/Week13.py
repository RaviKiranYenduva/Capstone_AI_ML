import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import ast
import re # Import re module for regular expressions

# Helper function for recursive flattening
def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# =============================
# CUSTOM DATA PARSER
# =============================
def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    parsed_entries_raw = [] # This will hold the result of ast.literal_eval for each logical entry
    current_entry_str = ""
    for line in lines:
        current_entry_str += line.strip()

        # Aggressively clean the string for ast.literal_eval using regex
        # Replace 'array(value)' with '[value]' for numpy array representations
        # Replace 'np.float64(value)' with 'value' for numpy float representations
        cleaned_str = re.sub(r'array\\((.*?)\\)', r'[\\1]', current_entry_str)
        # Corrected regex: np\\.float64 instead of np\\.float64
        cleaned_str = re.sub(r'np\\.float64\\((.*?)\\)', r'\\1', cleaned_str)

        try:
            val = ast.literal_eval(cleaned_str)
            parsed_entries_raw.append(val)
            current_entry_str = "" # Reset for next entry
        except (SyntaxError, ValueError):
            # If parsing fails, it's likely a multi-line entry or an incomplete string, continue accumulating
            pass

    # Now, process parsed_entries_raw to ensure a consistent 2D structure
    final_structured_data = []
    for entry in parsed_entries_raw:
        if isinstance(entry, list):
            # Recursively flatten all nested lists into a single row
            flat_row = list(flatten(entry))
            if flat_row: # Only add row if it has content
                final_structured_data.append(flat_row)
        elif isinstance(entry, (int, float)):
            final_structured_data.append([entry]) # Wrap single numbers in a list to form a row
        else: # Handle other potential direct scalar values that ast.literal_eval might return
            try:
                final_structured_data.append([float(entry)])
            except (ValueError, TypeError):
                # This case should ideally not be hit with good input files
                pass

    if not final_structured_data:
        return np.array([])

    # Ensure all rows have the same dimension, padding with 0.0 if necessary
    max_dim = 0
    if final_structured_data:
        max_dim = max(len(row) for row in final_structured_data)

    padded_data = []
    for row in final_structured_data:
        current_row = [float(x) for x in row] # Ensure all elements are floats
        if len(current_row) < max_dim:
            current_row.extend([0.0] * (max_dim - len(current_row)))
        padded_data.append(current_row)

    return np.array(padded_data)


# =============================
# LOAD DATA
# =============================
# Use the custom parser instead of np.loadtxt
# This assumes inputs.txt and outputs.txt have been created in the environment
input_data_from_file = parse_data_file("inputs.txt")
output_data_from_file = parse_data_file("outputs.txt")


X_raw = input_data_from_file
# The problem description indicates:
# `input_data` represents the raw features (`X_raw`) for all observations.
# `output_data` contains `y` values where each column corresponds to a different function (8 functions in total).

# =============================
# GROUP BY FUNCTION (ADJUSTED)
# =============================
X_funcs, y_funcs = [], []
num_functions = 8 # As per the loop range in the original code

# For this setup, `X_raw` contains the inputs for all functions (one per row)
# and `output_data_from_file` should have 8 columns for 8 functions.
# However, if output_data_from_file is (N_obs, 8), and X_raw is (N_obs, Dim),
# the peer's code assumes each function uses the SAME X_raw.
# Given our current data, X_raw will be (8, max_dim_of_inputs)
# and output_data_from_file will be (1, 8).
# This means we must use the individual X from X_raw for each function.

for i in range(num_functions):
    # Each function will use its corresponding row from X_raw as its input
    # and its corresponding column from output_data_from_file as its output (which is just one value here).
    X_funcs.append(X_raw[i:i+1, :]) # Take one row to keep it 2D
    y_funcs.append(output_data_from_file[0, i:i+1]) # Take one value from the single row


# =============================
# NEURAL NETWORK SURROGATE
# =============================
class SurrogateNN(nn.Module):
    def __init__(self,input_dim,hidden_layers):
        super().__init__()
        layers=[]
        in_dim=input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim,h))
            layers.append(nn.ReLU())
            in_dim=h
        layers.append(nn.Linear(in_dim,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

def train_nn(X,y,hidden_layers,lr,epochs):
    model = SurrogateNN(X.shape[1], hidden_layers)
    X_tensor = torch.tensor(X,dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1,1),dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# =============================
# PROPOSE NEXT QUERY USING RL-INSPIRED STRATEGY
# =============================
def propose_next(X, y):
    dim = X.shape[1]

    # Hyperparameters
    if dim <= 4: hidden=[32,16]; lr=0.004; epochs=500; scale=0.05
    elif dim <= 6: hidden=[64,32]; lr=0.0035; epochs=600; scale=0.04
    else: hidden=[128,64,32]; lr=0.003; epochs=700; scale=0.03

    # Standardize
    scaler = StandardScaler() # This will be ineffective with 1 sample
    X_scaled = scaler.fit_transform(X)

    # Train NN surrogate - will overfit with 1 sample
    nn_model = train_nn(X_scaled, y, hidden, lr, epochs)

    # GP on residuals - will be degenerate with 1 sample
    nn_pred = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten()
    residuals = y - nn_pred
    kernel = ConstantKernel(1.0)*Matern(nu=2.5)+WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled,residuals)

    # PCA for main directions - not meaningful with 1 sample
    # Adjust n_components to be at most the number of samples
    pca_n_components = min(3, dim, X.shape[0])
    pca = PCA(n_components=pca_n_components)
    pca.fit(X_scaled)
    dirs = pca.components_ # This will likely be empty or trivial if pca_n_components is 0 or 1

    # Candidate points
    n_candidates=3000
    X_candidates=np.random.uniform(0,1,(n_candidates,dim))

    # The following loop for adding directions will have no effect if dirs is empty
    for dir_vec in dirs:
        X_candidates += 0.02 * np.random.randn(n_candidates, dim) * dir_vec
    X_candidates = np.clip(X_candidates,0,1)

    # StandardScaler.transform on single point fit will be problematic
    X_candidates_scaled = scaler.transform(X_candidates)

    # NN+GP predictions
    nn_pred_cand = nn_model(torch.tensor(X_candidates_scaled,dtype=torch.float32)).detach().numpy().flatten()
    gp_pred_cand, gp_std_cand = gp.predict(X_candidates_scaled,return_std=True) # GP prediction with 1 fit point is unreliable
    mu = nn_pred_cand + gp_pred_cand
    sigma = gp_std_cand

    # RL-inspired epsilon-greedy exploration
    epsilon = 0.15
    best_idx = np.argmax(mu)
    if np.random.rand() < epsilon:
        best_idx = np.random.randint(0,n_candidates)

    return X_candidates[best_idx]

# =============================
# FORMAT QUERY
# =============================
def format_query(x):
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP
# =============================
all_queries=[]
for i in range(8):
    X = X_funcs[i]
    y = y_funcs[i]
    x_next = propose_next(X,y)
    query = format_query(x_next)
    all_queries.append(query)
    print(f"Function {i+1} next query: {query}")

print("\n===== FINAL SUBMISSION =====")
for i,q in enumerate(all_queries,1):
    print(f"Function {i}: {q}")