!pip install scikit-learn torch scipy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import ast # Import ast for literal_eval

# Data provided by the user in the prompt
initial_function_data = {
    "Function 1": {"inputs": [[0.028872, 1.000000]], "outputs": [0.0]},
    "Function 2": {"inputs": [[0.955331, 0.001673]], "outputs": [-0.04435924748200469]},
    "Function 3": {"inputs": [[0.992097, 0.067374, 0.975576]], "outputs": [-0.3935373674858445]},
    "Function 4": {"inputs": [[0.961943, 0.938351, 1.000000, 0.902278]], "outputs": [-47.870333836311396]},
    "Function 5": {"inputs": [[0.010065, 0.902878, 0.995016, 0.939627]], "outputs": [2665.444199890151]},
    "Function 6": {"inputs": [[0.788772, 0.037728, 0.033951, 0.959999, 0.314198]], "outputs": [-1.6069717587774175]},
    "Function 7": {"inputs": [[0.046481, 0.529228, 0.875521, 1.000000, 0.826492, 0.937951]], "outputs": [0.01504097247446136]},
    "Function 8": {"inputs": [[0.883017, 0.893446, 0.967498, 0.424951, 0.907135, 0.889902, 0.036251, 0.057695]], "outputs": [5.800450583072]}
}

# Create dummy inputs.txt and outputs.txt files
with open('inputs.txt', 'w') as f_inputs:
    # Assuming all inputs are the same for each function in this context, 
    # based on how X_raw was used previously in the peer code logic.
    # For simplicity, taking the inputs from the first function as representative.
    # In a real scenario, inputs.txt would contain all X values if they differ per function.
    for val in initial_function_data["Function 1"]["inputs"][0]:
        f_inputs.write(f"[{val}]\n")

with open('outputs.txt', 'w') as f_outputs:
    # Each column in outputs.txt represents a function's output
    # Here we are simulating each function having one output value
    # but the parse_data_file expects y_all_funcs[:, i] so we need to put them side by side
    # for this simulation, we'll write each function's single output as a column
    output_line = []
    for func_name in initial_function_data:
        output_line.append(str(initial_function_data[func_name]["outputs"][0]))
    f_outputs.write("[" + ", ".join(output_line) + "]\n")

# =============================
# LOAD TXT DATA
# =============================
def parse_data_file(filepath):
    with open(filepath, 'r') as f:
        # Remove 'array(', 'np.float64(' and ')' which wrap the actual data
        content = f.read().replace('array(', '').replace('np.float64(', '').replace(')', '')

    records = []
    current_record_parts = []
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('['):
            # New record starts
            if current_record_parts:
                records.append(''.join(current_record_parts))
            current_record_parts = [stripped_line]
        elif stripped_line:
            # Continuation of the current record
            current_record_parts.append(stripped_line)
    if current_record_parts:
        records.append(''.join(current_record_parts))

    processed_data = []
    for record_str in records:
        try:
            # ast.literal_eval can parse the string into a list of lists (or tuples) or a list of floats
            parsed_record = ast.literal_eval(record_str.replace('\n', ''))

            flattened_record = []
            if isinstance(parsed_record, (list, tuple)):
                # Iterate through elements of the parsed record
                for element in parsed_record:
                    if isinstance(element, (list, tuple)):
                        # If an element is a sublist/tuple, iterate through its items
                        for item in element:
                            flattened_record.append(float(item))
                    elif isinstance(element, (int, float)):
                        # If an element is a single float/int, just add it
                        flattened_record.append(float(element))
                    else:
                        raise TypeError(f"Unexpected element type in parsed_record: {type(element)}")
            elif isinstance(parsed_record, (int, float)):
                # If the entire record is a single float/int, wrap it in a list
                flattened_record.append(float(parsed_record))
            else:
                raise TypeError(f"Unexpected type for parsed_record: {type(type(parsed_record))}")

            processed_data.append(flattened_record)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing record: {record_str[:100]}... Error: {e}")
            raise

    return np.array(processed_data)

# Load data from files
input_data = parse_data_file("inputs.txt")
output_data = parse_data_file("outputs.txt")

X_raw = input_data
y_all_funcs = output_data

# =============================
# GROUP BY FUNCTION
# =============================
# The original function_data was fixed to 8 functions.
# We need to map the loaded data to this structure.
function_data = {}
for i in range(8):
    func_name = f"Function {i + 1}"
    # Assuming X_raw contains all inputs, and y_all_funcs has columns for each function's outputs
    # Ensure y_all_funcs has at least i-th column
    if y_all_funcs.shape[1] > i:
        # X_raw should reflect the dimensionality of the inputs for a given function
        # In the provided data, function 1 has 2 inputs, function 3 has 3, etc.
        # This current setup of writing inputs.txt assumes all functions share the same input dimensionality. 
        # To correctly handle varied input dimensions from a single inputs.txt, 
        # the structure of inputs.txt and how X_raw is parsed would need to be more complex.
        # For now, we will use X_raw as is and assume it has been parsed correctly for current needs.
        
        # Re-creating 'function_data' to match the initial prompt structure based on the parsed files
        # This assumes the first row of input_data has the correct number of dimensions
        # for function 'i+1'. This might be problematic if functions have different input dims.
        # A more robust solution would write distinct input lines for each function if dims vary.
        function_data[func_name] = {"inputs": [initial_function_data[func_name]["inputs"][0]], "outputs": y_all_funcs[:, i].tolist()}
    else:
        print(f"Warning: Not enough output columns in outputs.txt for {func_name}. Using empty outputs.")
        function_data[func_name] = {"inputs": [], "outputs": []} # Empty inputs if we can't determine them robustly

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 5000
    local_n = 2500
    if func_id == 2:  # noisy
        noise = 1e-3
    if func_id == 5:  # unimodal
        xi = 0.001
    if func_id == 8:  # high-dim
        global_n = 7000
    return xi, noise, global_n, local_n

# =============================
# DEEPER NEURAL NETWORK SURROGATE
# =============================
class DeepSurrogateNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_nn(X, y, epochs=700, lr=0.005):
    device = torch.device('cpu')
    model = DeepSurrogateNN(X.shape[1]).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32).to(device)
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
# PROPOSE NEXT POINT
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Step 1: Standardize ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Step 2: Train deeper neural network ----
    nn_model = train_nn(X_scaled, y, epochs=700)

    # ---- Step 3: GP on residuals ----
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_tensor).detach().numpy().flatten()
    residuals = y - nn_pred

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled, residuals)

    # ---- Step 4: Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    # Ensure scaler is fitted with the correct number of features for transform
    # If X_scaled is 1D (single data point), StandardScaler might need reshaping.
    # This assumes X contains at least two points to correctly fit the scaler, or the user manually passes mean/std.
    # For this specific scenario where X is only one point, transform might fail or result in 0 std.
    # However, to follow the peer code, we proceed assuming scaler can transform correctly.
    X_global_scaled = scaler.transform(X_global)

    best_x = X[np.argmax(y)]
    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    # ---- Step 5: Predictions ----
    X_candidates_tensor = torch.tensor(X_candidates_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_candidates_tensor).detach().numpy().flatten()
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    # ---- Step 6: Expected Improvement ----
    # Need to handle case where X_scaled has only one data point for nn_model
    mu_sample_nn = nn_model(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().flatten()
    mu_sample_gp = gp.predict(X_scaled)
    mu_sample = mu_sample_nn + mu_sample_gp
    
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

# Iterate through the existing function_data dictionary
for i, (func_name, data) in enumerate(function_data.items()):
    # Extract inputs and outputs, convert to numpy arrays
    X = np.array(data["inputs"])
    y = np.array(data["outputs"])

    print(f"\n===== {func_name} ====")
    print("Data points:", len(y))

    # Pass func_id (1-indexed for get_settings)
    x_next = propose_next(X, y, i + 1)
    query = format_query(x_next)
    all_queries.append(query)

    print("Next query:", query)

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")