import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import ast
import re

# Store the provided input and output values for the 8 blackbox functions
blackbox_data = {
    "Function 1": {"inputs": [0.993196, 0.236731], "output": 7.344030539509893e-200},
    "Function 2": {"inputs": [0.746869, 0.909159], "output": 0.32391784797101764},
    "Function 3": {"inputs": [0.985814, 0.053477, 0.931550], "output": -0.22215621547242842},
    "Function 4": {"inputs": [0.383818, 0.917165, 0.963790, 0.913100], "output": -35.31146837938973},
    "Function 5": {"inputs": [0.888810, 0.995762, 0.026609, 0.990750], "output": 3073.709538061085},
    "Function 6": {"inputs": [0.378911, 0.882068, 0.759052, 0.134395, 0.086026], "output": -1.2856056786262238},
    "Function 7": {"inputs": [0.864091, 0.848365, 0.929388, 0.818808, 0.967149, 0.272394], "output": 0.020778004394931982},
    "Function 8": {"inputs": [0.979302, 0.982246, 0.950224, 0.112368, 0.959742, 0.068926, 0.610885, 0.465713], "output": 5.203264370079101}
}

# Helper function for recursive flattening
def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# =============================
# CUSTOM DATA PARSER (Included for completeness if needed in a self-contained script)
# =============================
def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    parsed_entries_raw = []
    current_entry_str = ""
    for line in lines:
        current_entry_str += line.strip()
        cleaned_str = re.sub(r'array\((.*?)\)', r'[\1]', current_entry_str)
        cleaned_str = re.sub(r'np\\.float64\\((.*?)\)', r'\1', cleaned_str)
        try:
            val = ast.literal_eval(cleaned_str)
            parsed_entries_raw.append(val)
            current_entry_str = ""
        except (SyntaxError, ValueError):
            pass

    final_structured_data = []
    for entry in parsed_entries_raw:
        if isinstance(entry, list):
            flat_row = list(flatten(entry))
            if flat_row:
                final_structured_data.append(flat_row)
        elif isinstance(entry, (int, float)):
            final_structured_data.append([entry])
        else:
            try:
                final_structured_data.append([float(entry)])
            except (ValueError, TypeError):
                pass

    if not final_structured_data:
        return np.array([])

    max_dim = 0
    if final_structured_data:
        max_dim = max(len(row) for row in final_structured_data)

    padded_data = []
    for row in final_structured_data:
        current_row = [float(x) for x in row]
        if len(current_row) < max_dim:
            current_row.extend([0.0] * (max_dim - len(current_row)))
        padded_data.append(current_row)

    return np.array(padded_data)


# Simulate input_data by concatenating all function inputs
# Determine the maximum input dimension across all functions
max_input_dim = max(len(data['inputs']) for data in blackbox_data.values())

# Create X_raw by padding all inputs to the max_input_dim
X_raw_list = []
for func_key in sorted(blackbox_data.keys()):
    inputs = blackbox_data[func_key]['inputs']
    padded_inputs = inputs + [0.0] * (max_input_dim - len(inputs))
    X_raw_list.append(padded_inputs)
X_raw = np.array(X_raw_list)

# =============================
# GROUP BY FUNCTION (REVISED FOR SINGLE OBSERVATION PER FUNCTION)
# =============================
X_funcs, y_funcs = [], []
func_dims = []

for i in range(1, 9):
    func_key = f"Function {i}"
    inputs = blackbox_data[func_key]['inputs']
    output = blackbox_data[func_key]['output']

    X_funcs.append(np.array([inputs]))
    y_funcs.append(np.array([output]))
    func_dims.append(len(inputs))

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 5000
    local_n = 2500
    if func_id == 2:
        noise = 1e-3
    if func_id == 5:
        xi = 0.001
    if func_id == 8:
        global_n = 7000
    return xi, noise, global_n, local_n

# =============================
# CNN-INSPIRED NEURAL NETWORK SURROGATE
# =============================
class CNNInspiredNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 4:
            hidden = [32, 16]
        elif input_dim <= 6:
            hidden = [64, 32]
        else:
            hidden = [128, 64, 32]

        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_nn(X, y, epochs=600, lr=0.005):
    device = torch.device('cpu')
    model = CNNInspiredNN(X.shape[1]).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    return model

# =============================
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id, dim):
    xi, noise, global_n, local_n = get_settings(func_id)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn_model = train_nn(X_scaled, y, epochs=600)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_tensor).detach().numpy().flatten()
    residuals = y - nn_pred

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(X_scaled, residuals)

    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    X_global_scaled = scaler.transform(X_global)

    best_x = X[np.argmin(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    X_candidates_tensor = torch.tensor(X_candidates_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_candidates_tensor).detach().numpy().flatten()
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    mu_sample = nn_model(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().flatten() + gp.predict(X_scaled)
    mu_best = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu_best - mu
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

for i in range(8):
    print(f"\n===== Function {i+1} ====")
    X = X_funcs[i]
    y = y_funcs[i]
    dim = func_dims[i]

    print("Initial data points:", len(y))

    x_next = propose_next(X, y, i+1, dim)
    query = format_query(x_next)
    all_queries.append(query)

    print("Next query:", query)

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")