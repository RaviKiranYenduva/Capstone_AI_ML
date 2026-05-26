import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import ast
import re # Import re module for regular expressions

# --- REMOVE UNUSED HELPER FUNCTION AND DATA PARSER ---
# The original flatten and parse_data_file functions are not needed
# as data is provided directly in the prompt.

# =============================
# DATA FROM PROMPT
# =============================
# Extract input and output data directly from the provided text in the prompt.
raw_input_values_str = """Function 1:	[0.525077, 0.917667]
Function 2:	[0.446370, 0.494934]
Function 3:	[0.953166, 0.397032, 0.256989]
Function 4:	[0.100347, 0.815430, 0.052306, 0.091887]
Function 5:	[0.318415, 0.929234, 0.434503, 0.019737]
Function 6:	[0.317192, 0.604213, 0.369984, 0.725351, 0.683665]
Function 7:	[0.582632, 0.330345, 0.672070, 0.544367, 0.776977, 0.275198]
Function 8:	[0.147139, 0.028788, 0.457741, 0.956610, 0.799926, 0.643331, 0.390367, 0.003097]"""

raw_output_values_str = """Function 1:	-1.745004766692742e-66
Function 2:	0.24201582757197876
Function 3:	-0.10046357925633227
Function 4:	-20.82651382625456
Function 5:	71.82335719397292
Function 6:	-1.2992335196946576
Function 7:	0.09162547930870235
Function 8:	8.8793490122531"""

def parse_function_data_from_str(data_str, is_input=True):
    parsed_data = []
    lines = data_str.strip().split('\n')
    for line in lines:
        parts = line.split(':')
        if len(parts) > 1:
            value_str = parts[1].strip()
            if is_input:
                # Use ast.literal_eval for lists of input values
                # Wrap the list in another list to form a 2D array of shape (1, D)
                parsed_data.append(np.array([ast.literal_eval(value_str)]))
            else:
                # Convert to float for output values
                # Wrap the scalar in an array to form a 1D array of shape (1,)
                parsed_data.append(np.array([float(value_str)]))
    return parsed_data

X_funcs = parse_function_data_from_str(raw_input_values_str, is_input=True)
y_funcs = parse_function_data_from_str(raw_output_values_str, is_input=False)


# =============================
# HYPERPARAMS & SCALING
# =============================
def get_hyperparams(func_id, input_dim):
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
# NEURAL NETWORK SURROGATE
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
# PROPOSE NEXT QUERY
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
    nn_pred = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten()
    residuals = y - nn_pred
    kernel = ConstantKernel(1.0)*Matern(nu=2.5)+WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled,residuals)

    # Candidate generation (global + local with scaling)
    X_global = np.random.uniform(0,1,(global_n,dim))
    X_local = X[np.argmax(y)] + scale*np.random.randn(local_n,dim)
    X_local = np.clip(X_local,0,1)
    X_candidates = np.vstack([X_global,X_local])
    X_candidates_scaled = scaler.transform(X_candidates)

    # Predictions
    nn_pred = nn_model(torch.tensor(X_candidates_scaled,dtype=torch.float32)).detach().numpy().flatten()
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    # Expected Improvement
    mu_sample = nn_model(torch.tensor(X_scaled,dtype=torch.float32)).detach().numpy().flatten() + gp.predict(X_scaled)
    mu_best = np.max(mu_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_best - xi
        Z = imp/sigma
        ei = imp*norm.cdf(Z) + sigma*norm.pdf(Z)
        ei[sigma==0.0]=0.0

    return X_candidates[np.argmax(ei)]

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
    # X_funcs[i] and y_funcs[i] are already correctly structured as numpy arrays
    # with the correct dimensions for a single observation.
    X = X_funcs[i]
    y = y_funcs[i]
    # func_id should be 1-indexed for get_hyperparams
    x_next = propose_next(X, y, i + 1)
    query = format_query(x_next)
    all_queries.append(query)
    print(f"Function {i+1} next query: {query}")

# =============================
# FINAL SUBMISSION
# =============================
print("\n===== SUBMIT THESE =====")
for i,q in enumerate(all_queries,1):
    print(f"Function {i}: {q}")