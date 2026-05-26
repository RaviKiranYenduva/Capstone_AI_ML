import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# Raw data from the problem description
raw_data = '''
Function 1:\t[0.953531, 0.428438]
Function 2:\t[0.991014, 0.993377]
Function 3:\t[0.967517, 0.015643, 0.961475]
Function 4:\t[0.062593, 0.014843, 0.903420, 0.059318]
Function 5:\t[0.907623, 0.010499, 0.100778, 0.029622]
Function 6:\t[0.015846, 0.974338, 0.091127, 0.786588, 0.876977]
Function 7:\t[0.023468, 0.991862, 0.096059, 0.163216, 0.322099, 0.760238]
Function 8:\t[0.023482, 0.857907, 0.929733, 0.101186, 0.853463, 0.742896, 0.006869, 0.119686]
This week's output values: 
Function 1:\t-1.654648495316357e-102
Function 2:\t0.09210829943966123
Function 3:\t-0.33911653503094885
Function 4:\t-25.836832021782488
Function 5:\t29.69831842149032
Function 6:\t-2.3124683518641986
Function 7:\t0.15250787698834206
Function 8:\t7.4079593170579
'''

inputs = {}
outputs = {}

def parse_line(line):
    parts = line.strip().split(':\t')
    if len(parts) == 2:
        func_name = parts[0]
        value_str = parts[1]
        return func_name, value_str
    return None, None

lines = raw_data.split('\n')

parsing_inputs = True
for line in lines:
    if 'output values' in line:
        parsing_inputs = False
        continue

    func_name, value_str = parse_line(line)

    if func_name and value_str:
        if parsing_inputs:
            # Remove brackets and split by comma, then convert to float
            inputs[func_name] = [float(x.strip()) for x in value_str.strip('[]').split(',')]
        else:
            outputs[func_name] = float(value_str)

print("Parsed Input Values:")
for func, val in inputs.items():
    print(f"{func}: {val}")

print("\nParsed Output Values:")
for func, val in outputs.items():
    print(f"{func}: {val}")

# Helper function for recursive flattening (from peer code, adapted slightly for clarity)
# Not directly used given our current data structure, but kept for completeness if future data is nested.
def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# =============================
# NEURAL NETWORK SURROGATE (from peer code)
# =============================
class ScalingNN(nn.Module):
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
    model = ScalingNN(X.shape[1], hidden_layers)
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
# HYPERPARAMS (from peer code)
# =============================
def get_hyperparams(func_id, input_dim):
    # Note: func_id is 1-indexed here, as per peer code's usage
    if input_dim <= 4:
        hidden=[32,16]; lr=0.004; epochs=500; noise_level=1e-5; scale=0.05
    elif input_dim <= 6:
        hidden=[64,32]; lr=0.0035; epochs=600; noise_level=1e-5; scale=0.04
    else:
        hidden=[128,64,32]; lr=0.003; epochs=700; noise_level=1e-5; scale=0.03
    xi=0.02
    global_n, local_n = 5000, 2500
    if func_id==2: noise_level=1e-3
    if func_id==5: xi=0.001
    if func_id==8: global_n=7000
    return hidden, lr, epochs, xi, noise_level, global_n, local_n, scale

# =============================
# PROPOSE NEXT QUERY (from peer code)
# =============================
def propose_next(X, y, func_id):
    dim = X.shape[1]
    hidden, lr, epochs, xi, noise_level, global_n, local_n, scale = get_hyperparams(func_id,dim)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train NN surrogate
    nn_model = train_nn(X_scaled, y, hidden, lr, epochs)

    # GP on residuals
    nn_pred_train = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten()
    residuals = y - nn_pred_train
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    # Define fallbacks for GP predictions, in case GP cannot be fitted or predicted
    gp_pred_fallback = np.zeros(global_n + local_n)
    gp_std_fallback = np.ones(global_n + local_n) # High uncertainty if GP fails

    gp_fitted = False
    # Ensure residuals are not all zeros or too few points for GP fitting
    if len(np.unique(residuals)) < 2 or X_scaled.shape[0] < 2:
        print(f"Warning: Not enough diverse data for Function {func_id} to fit GP properly. Proceeding with NN-only or random search.")
        # gp_fitted remains False, so fallbacks will be used.
    else:
        try:
            gp.fit(X_scaled,residuals)
            gp_fitted = True
        except Exception as e:
            print(f"Error fitting GP for Function {func_id}: {e}. Proceeding without GP uncertainty.")
            gp_fitted = False # Explicitly set to False if fit fails

    # Candidate generation (global + local)
    X_global = np.random.uniform(0,1,(global_n,dim))
    X_local = X[np.argmax(y)] + scale*np.random.randn(local_n,dim)
    X_local = np.clip(X_local,0,1)
    X_candidates = np.vstack([X_global,X_local])
    X_candidates_scaled = scaler.transform(X_candidates)

    # Predictions
    nn_pred_candidates = nn_model(torch.tensor(X_candidates_scaled,dtype=torch.float32)).detach().numpy().flatten()

    gp_pred, gp_std = gp_pred_fallback, gp_std_fallback # Initialize with fallbacks
    if gp_fitted:
        try:
            gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
        except Exception as e:
            print(f"Error during GP prediction for Function {func_id}: {e}. Using fallback for GP component.")
            # gp_pred and gp_std remain at fallback values

    mu = nn_pred_candidates + gp_pred
    sigma = gp_std

    # Expected Improvement
    mu_best = np.max(y) # Use the best observed output value directly

    with np.errstate(divide='warn', invalid='ignore'):
        imp = mu - mu_best - xi
        Z = np.zeros_like(imp)
        non_zero_sigma_idx = sigma > 1e-10
        Z[non_zero_sigma_idx] = imp[non_zero_sigma_idx] / sigma[non_zero_sigma_idx]

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[~non_zero_sigma_idx] = 0.0

    # Feature importance approximation
    if np.any(ei > 0):
        importance = np.var(X_candidates_scaled * ei[:,None], axis=0)
        print(f"Function {func_id} feature importance: {importance}")
    else:
        print(f"Function {func_id} EI is all zero, cannot calculate feature importance meaningfully.")

    if np.any(ei > 0):
        return X_candidates[np.argmax(ei)]
    else:
        print(f"Warning: EI for Function {func_id} is all zero. Returning a random candidate.")
        return X_global[np.random.randint(0, global_n)]

# =============================
# FORMAT QUERY (from peer code)
# =============================
def format_query(x):
    # Ensure formatting to six decimal places for each element
    return "-".join([f"{xi:.6f}" for xi in x])

# =============================
# MAIN LOOP (adapted from peer code)
# =============================

# Prepare data in the format expected by propose_next
X_funcs = [] # List of numpy arrays, where each array is the historical input for one function
y_funcs = [] # List of numpy arrays, where each array is the historical output for one function
num_functions = 8

for i in range(1, num_functions + 1):
    func_key = f'Function {i}'
    # Each function's input is currently a single historical observation.
    # propose_next expects a 2D array for X (even if it's 1xDim) and 1D for y.
    X_func_i = np.array([inputs[func_key]])
    X_funcs.append(X_func_i)

    y_func_i = np.array([outputs[func_key]]).flatten()
    y_funcs.append(y_func_i)

all_queries = []
print("\n===== PROPOSED NEXT QUERIES USING BBO STRATEGY =====")
for i in range(num_functions):
    # The func_id for get_hyperparams is 1-indexed
    x_next = propose_next(X_funcs[i], y_funcs[i], i + 1)
    query_formatted = format_query(x_next)
    all_queries.append(query_formatted)
    print(f"Function {i+1} next query: {query_formatted}")

print("\n===== FINAL SUBMISSION FORMAT =====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")