# ## Resources
#
# ### Data Preparation for Black-Box Optimization with CNNs
#
# To apply a Convolutional Neural Network (CNN) to the provided black-box function data, we first need to prepare the input values. CNNs typically require inputs of a consistent shape. Since your input arrays for each function have varying lengths, we'll pad them with zeros to the maximum observed length. The output values are scalar, indicating a regression task.
#
# We'll use `numpy` for numerical operations and `tensorflow.keras` to build and train our CNN model.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Provided input and output values
input_data = [
    [0.000000, 1.000000],
    [0.576828, 0.005590],
    [0.994414, 0.155934, 0.577175],
    [0.049333, 0.556906, 0.959505, 0.776867],
    [0.987195, 0.014819, 0.712247, 0.095264],
    [0.873108, 0.012525, 0.855522, 0.635516, 0.192955],
    [0.262767, 0.000976, 0.006989, 0.140990, 0.270575, 0.082587],
    [0.118005, 0.031473, 0.921936, 0.975123, 0.023717, 0.180215, 0.356369, 0.976873]
]

output_data = [
    0,
    -0.12835919922528194,
    -0.07032721401207534,
    -26.089357528789424,
    378.00694684013956,
    -1.2813517694628827,
    0.1198706087601757,
    6.9563111923996
]

# Find the maximum length of the input arrays for padding
max_input_length = max(len(arr) for arr in input_data)
print(f"Maximum input length: {max_input_length}")

# Pad input arrays with zeros to the max_input_length
padded_inputs = np.array([
    arr + [0.0] * (max_input_length - len(arr)) for arr in input_data
])

# Convert outputs to a numpy array
target_outputs = np.array(output_data)

print("\nShape of padded inputs:", padded_inputs.shape)
print("Shape of target outputs:", target_outputs.shape)
print("\nSample padded input (Function 1):", padded_inputs[0])
print("Sample target output (Function 1):", target_outputs[0])

# ### Building a 1D CNN Model for Regression
#
# Since our inputs are sequences of numerical values, a 1D CNN is appropriate. This model will treat each input array as a sequence and use convolutional layers to extract features. The final output layer will be a single neuron with a linear activation function, suitable for predicting continuous values (regression).
#
# Given the extremely small dataset (8 samples), this model will primarily serve as a conceptual demonstration. For real-world applications with such limited data, techniques like transfer learning, data augmentation (if applicable), or simpler models might be more robust.

# Define the 1D CNN model
model = keras.Sequential([
    layers.Input(shape=(max_input_length, 1)), # Input shape: (sequence_length, features)
    layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) # Output layer for regression (single continuous value)
])

# Compile the model
# Using 'mse' (Mean Squared Error) as the loss function for regression
# And 'adam' optimizer for its efficiency
model.compile(optimizer='adam', loss='mse')

# Display the model summary
model.summary()

# Reshape padded_inputs to be compatible with Conv1D (add a channel dimension)
X = padded_inputs.reshape(padded_inputs.shape[0], padded_inputs.shape[1], 1)
y = target_outputs

print("\nShape of X for CNN input:", X.shape)

# Train the model (with very few epochs for demonstration due to small dataset)
# In a real scenario, you'd need more data and careful validation.
history = model.fit(X, y, epochs=50, verbose=0) # verbose=0 to suppress output for brevity

print("\nModel training complete. Loss:", history.history['loss'][-1])

# Make a prediction for a sample input (e.g., Function 1)
sample_input_for_prediction = X[0:1]
predicted_output = model.predict(sample_input_for_prediction)[0][0]

print(f"\nActual output for Function 1: {y[0]:.4f}")
print(f"Predicted output for Function 1: {predicted_output:.4f}")

# You can now try to make predictions for other functions or new inputs
# For example, to predict for all inputs:
all_predictions = model.predict(X).flatten()
print("\nAll predictions:", all_predictions)
print("Actual outputs:", y)

# ### When are CNNs appropriate for solving business problems?
#
# Convolutional Neural Networks (CNNs) are particularly powerful for tasks involving data with a known grid-like topology, meaning data where spatial or temporal relationships between features are important. While often associated with image processing, their utility extends to other domains:
#
# 1.  **Image and Video Analysis**: This is the most common application. CNNs excel at:
#     *   **Object Detection and Recognition**: Identifying objects within images or video streams (e.g., product recognition in retail, anomaly detection in manufacturing inspection, facial recognition for security).
#     *   **Image Classification**: Categorizing images (e.g., classifying medical images for disease diagnosis, sorting products based on visual features, content moderation).
#     *   **Image Segmentation**: Dividing an image into segments to locate objects and boundaries (e.g., autonomous driving, medical imaging for tumor detection).
#     *   **Visual Search**: Finding similar images for e-commerce or intellectual property protection.
#
# 2.  **Time Series and Sequential Data Analysis**: 1D CNNs (like the one we just built) are effective for data where patterns occur sequentially, such as:
#     *   **Financial Market Prediction**: Analyzing stock prices, commodity trends, or other financial time series data.
#     *   **Sensor Data Analysis**: Processing data from IoT devices for predictive maintenance, anomaly detection, or system monitoring.
#     *   **Natural Language Processing (NLP)**: While Recurrent Neural Networks (RNNs) and Transformers are often preferred, CNNs can be used for tasks like sentiment analysis, text classification, and feature extraction from text by treating words or characters as a sequence.
#     *   **Audio Processing**: Analyzing speech, music, or other sound patterns for tasks like speaker recognition, voice assistants, or sound event detection.
#
# 3.  **Recommendation Systems**: CNNs can learn intricate patterns from user interaction data or item features, helping to provide more personalized recommendations.
#
# 4.  **Anomaly and Fraud Detection**: By learning normal patterns, CNNs can effectively flag unusual activities in financial transactions, network traffic, or operational data.
#
# ### Key characteristics that make CNNs a good fit:
#
# *   **Automatic Feature Extraction**: Unlike traditional machine learning models that require hand-crafted features, CNNs can automatically learn hierarchical features directly from raw data.
# *   **Parameter Sharing / Local Receptive Fields**: Convolutional layers apply the same filter across different parts of the input, making them efficient and able to detect patterns regardless of their position (translation invariance).
# *   **Hierarchical Pattern Learning**: Deeper CNNs can learn increasingly complex and abstract representations of the data.
# *   **Scalability**: With modern hardware (GPUs/TPUs) and frameworks, CNNs can handle very large datasets.
#
# In your specific black-box optimization challenge, if the input values represent some kind of sequential or structured parameter set, a 1D CNN could learn the relationship between these parameter configurations and the black-box function's output. The challenge here is the very limited number of samples, which is generally not ideal for training deep learning models.
#
# ### Adopting a Bayesian Optimization Strategy
#
# To refine the black-box optimization strategy, we will move towards a Bayesian Optimization approach, as suggested by the peer code. This involves building a surrogate model (a combination of a neural network and a Gaussian Process) to approximate the black-box function and an acquisition function (Expected Improvement) to guide the search for the next optimal point.
#
# First, we'll define some helper functions for data parsing, as the peer code uses `inputs.txt` and `outputs.txt`. Although we already have the `input_data` and `output_data` in Python lists, defining these functions will allow for future flexibility if you were to load data from text files.
#
# Note: The `CNNInspiredNN` in the peer code is a Multi-Layer Perceptron (MLP) whose architecture dynamically adjusts based on the input dimension, a principle that can be seen as 'inspired' by how CNN depths might vary with input complexity, but it does not use convolutional layers itself.

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
    # This function is adapted from the peer code but will be bypassed for now
    # as we already have input_data and output_data as Python lists.
    # It's included for completeness and future file-based loading.
    # For this current problem, we will use the directly provided data.
    print(f"Note: parse_data_file('{filepath}') is called but we are using pre-defined input_data/output_data for this task.")
    return np.array([]) # Return empty for now, as we'll use existing arrays

# Let's use the input_data and output_data already defined in the notebook.
# These are `input_data` and `output_data` from cell `ebdec231`

# We need to ensure X_raw matches the format expected by the Bayesian Optimization process.
# Assuming `padded_inputs` from our previous data preparation serves as X_raw.
X_raw = padded_inputs

# The peer code assumes output_data is a 2D array where each column corresponds to a function.
# Our `output_data` is a 1D array of outputs for the 8 functions. We need to re-structure.
# However, the Bayesian Optimization loop in the peer code iterates through 8 functions,
# and for each, it takes a specific `y` (y_funcs[i]) corresponding to that function.
# Given our current `output_data` is a flat list of 8 outputs for 8 distinct functions,
# we should treat each (X_raw, y_i) as a separate optimization problem where X_raw is the same for all.

# =============================
# GROUP BY FUNCTION (ADJUSTED for our current data structure)
# =============================
X_funcs, y_funcs = [], []
num_functions = 8

# Here, `X_raw` (padded_inputs) is the common set of input points
# and `output_data` is the list of corresponding outputs for *each* blackbox function
# assuming the i-th element of output_data corresponds to the output of function i+1
# for the *single* evaluation point it represents.

# The previous problem statement described this week's input/output values as a single evaluation per function.
# Example: Function 1: [0.0, 1.0], Output 1: 0.
# This means we have only ONE (input, output) pair per function for now.

# To make the Bayesian Optimization work, it usually expects a dataset of (X, y) pairs for each function.
# Since we only have one (X,y) pair per function from the prompt, we will use that as the starting 'observed' data.
# However, the peer code's structure for X_funcs and y_funcs implies multiple data points for each function.
# I will adapt this to represent the initial single observation for each function.

for i in range(num_functions):
    # Each function will start with a single observed (input, output) pair.
    # X_raw[i] is the padded input for Function i+1.
    # output_data[i] is the output for Function i+1.
    X_funcs.append(np.array([X_raw[i]]))
    y_funcs.append(np.array([output_data[i]]))

print("\nInitial X_funcs (first function):", X_funcs[0])
print("Initial y_funcs (first function):", y_funcs[0])

# =============================
# SETTINGS
# =============================
def get_settings(func_id):
    xi = 0.02
    noise = 1e-5
    global_n = 5000
    local_n = 2500
    # func_id is 1-indexed for functions
    if func_id == 2:  # noisy function
        noise = 1e-3
    if func_id == 5:  # unimodal
        xi = 0.001
    if func_id == 8:  # high-dimensional
        global_n = 7000
    return xi, noise, global_n, local_n

# ### CNN-Inspired Neural Network Surrogate
#
# This section defines the neural network architecture that will serve as part of the surrogate model in our Bayesian Optimization. As mentioned, it's an MLP with a hidden layer structure that scales with the input dimension.

class CNNInspiredNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Adjust depth based on input dimension (like CNN trade-offs)
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
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# ### Proposing the Next Point using Bayesian Optimization
#
# This is the core of the Bayesian Optimization algorithm. The `propose_next` function uses the trained neural network, a Gaussian Process Regressor, and the Expected Improvement acquisition function to suggest the next best input `x` to evaluate the black-box function. This process iteratively searches for the optimum.

def propose_next(X, y, func_id):
    dim = X.shape[1]
    xi, noise, global_n, local_n = get_settings(func_id)

    # ---- Standardize ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Train CNN-inspired surrogate ----
    nn_model = train_nn(X_scaled, y, epochs=600)

    # ---- GP on residuals ----
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_tensor).detach().numpy().flatten()
    residuals = y - nn_pred

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=noise)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_scaled, residuals)

    # ---- Candidate generation ----
    X_global = np.random.uniform(0, 1, size=(global_n, dim))
    X_global_scaled = scaler.transform(X_global)

    # best_x needs to be selected from the *observed* X, not from candidates in peer code.
    # For the initial step, this will be based on the single observed point.
    best_x_idx = np.argmax(y)
    best_x = X[best_x_idx]

    X_local = best_x + 0.05 * np.random.randn(local_n, dim)
    X_local = np.clip(X_local, 0, 1)
    X_local_scaled = scaler.transform(X_local)

    X_candidates_scaled = np.vstack([X_global_scaled, X_local_scaled])
    X_candidates = np.vstack([X_global, X_local])

    # ---- Predictions ----
    X_candidates_tensor = torch.tensor(X_candidates_scaled, dtype=torch.float32)
    nn_pred = nn_model(X_candidates_tensor).detach().numpy().flatten()
    gp_pred, gp_std = gp.predict(X_candidates_scaled, return_std=True)
    mu = nn_pred + gp_pred
    sigma = gp_std

    # ---- Expected Improvement ----
    # The calculation for mu_sample needs to consider both NN and GP predictions for observed points
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

# ### Executing the Optimization Loop
#
# Finally, we'll run the Bayesian Optimization loop for each of the 8 black-box functions. For each function, it will take the initial observed data (one point for each function, as provided), propose a new point using the `propose_next` function, and format the query for submission. If you were to run this iteratively, you would then evaluate the black-box function with `x_next` and add the result to your `X_funcs[i]` and `y_funcs[i]` for the next iteration.

# =============================
# FORMAT QUERY
# =============================
def format_query(x, original_dim):
    # Truncate x to its original dimension before formatting
    return "-".join([f"{xi:.6f}" for xi in x[:original_dim]])

# =============================
# MAIN LOOP SETUP
# =============================
all_queries = []

# Get original dimensions from the initial input_data
original_dims = [len(arr) for arr in input_data]
print(f"\nOriginal dimensions for each function: {original_dims}")

# This cell now contains the full main loop and the final submission printing.
for i in range(8):
    print(f"\n===== Function {i+1} ====")
    X = X_funcs[i] # Initial observed input for this function
    y = y_funcs[i] # Initial observed output for this function

    print("Data points:", len(y))

    x_next = propose_next(X, y, i+1)
    # Pass the original dimension for this function to format_query
    query = format_query(x_next, original_dims[i])
    all_queries.append(query)

    print("Next query:", query)

# =============================
# FINAL SUBMISSION
# =============================
# This part must be unindented to execute AFTER the loop completes
print("\n===== SUBMIT THESE ====")
for i, q in enumerate(all_queries, 1):
    print(f"Function {i}: {q}")

# ## Strategic Reflection: Bayesian Optimization & CNN Concepts
#
# This section provides a reflection on the adopted Bayesian Optimization (BO) strategy, drawing parallels with Convolutional Neural Network (CNN) concepts and explicitly linking to the implementation details within this notebook.
#
# ### 1. Progressive Feature Extraction & Hierarchical Understanding
#
# **CNN Analogy**: In CNNs, early layers detect simple features (edges, corners), and deeper layers combine these into more complex, abstract representations (objects).
#
# **BO Implementation**: Our BO strategy mirrors this by progressively refining its understanding of the black-box function landscape. Initially, with only one observed `(X, y)` pair per function (`X_funcs[i]`, `y_funcs[i]` initialized in `00e7693a`), the surrogate model starts with a broad approximation. As new query points (`x_next` from `propose_next` in `d6fa6a2a`) are generated and hypothetically evaluated (though not performed iteratively here), the surrogate model (combining `CNNInspiredNN` and `GaussianProcessRegressor`) would continuously update its 'features' (predictions and uncertainty estimates), leading to a more nuanced and hierarchical understanding of the function's optimal regions. The `StandardScaler` (`d6fa6a2a`) can also be seen as an initial normalization step, akin to feature scaling in deep learning inputs.
#
# ### 2. Breakthroughs vs. Incremental Improvements
#
# **CNN Analogy**: The development of new CNN architectures (e.g., ResNet, Transformer) represents breakthroughs, while each training epoch or new dataset contributes incremental improvements.
#
# **BO Implementation**: The adoption of the Bayesian Optimization framework itself, moving from a simplistic direct CNN model to a sophisticated surrogate-based search, was a significant *breakthrough* in strategy. Within this framework, each proposed `x_next` (`d6fa6a2a`, `08adb73e`) for a given function represents an *incremental improvement*. By evaluating these `x_next` values and feeding them back into the `X_funcs` and `y_funcs` arrays for subsequent iterations (a step not fully realized in this single-step demonstration), the system would incrementally converge towards better solutions.
#
# ### 3. Exploration-Exploitation Trade-offs
#
# **CNN Analogy**: In CNN training, balancing model complexity (depth, parameters) with generalization (avoiding overfitting) is an exploration-exploitation trade-off. Overfitting is exploiting the training data too much; underfitting is exploring too little of the feature space.
#
# **BO Implementation**: This trade-off is explicitly managed by the Expected Improvement (EI) acquisition function (`d6fa6a2a`). The `xi` parameter, configured via `get_settings(func_id)` (`00e7693a` and `d6fa6a2a`), directly controls this balance. A higher `xi` promotes exploration (searching uncertain regions), while a lower `xi` emphasizes exploitation (refining known good regions). The `global_n` and `local_n` parameters (`get_settings`) for candidate generation also contribute to this, by sampling globally for exploration and locally around the current best for exploitation.
#
# ### 4. CNN Components as Learning Mechanisms in BO
#
# **Convolutional Analogy**: The `GaussianProcessRegressor`'s kernel (`Matern`, `WhiteKernel` in `d6fa6a2a`) acts like a set of dynamic filters, akin to CNN convolutions. It models relationships and dependencies in the residual space, adapting its understanding of the function's smoothness and noise. The `CNNInspiredNN` (`a79fef5e`) uses `ReLU` activations, analogous to non-linear transformations in CNN layers, allowing the model to learn complex, non-linear relationships. The `nn.MSELoss()` in `train_nn` is directly analogous to a loss function in CNNs, guiding the surrogate model's learning process. Similarly, the Expected Improvement (EI) acquisition function, by maximizing potential gain, can be seen as a form of 'pooling' or 'attention' mechanism, focusing the search on the most promising areas of the parameter space.
#
# ### 5. Benchmarking Success: An Edge AI Analogy
#
# **Edge AI Analogy**: In Edge AI, success is often measured by efficiency within resource constraints (e.g., power, latency). The goal isn't always to find the absolute best model, but the best model given limited computational budget or data.
#
# **BO Implementation**: Our BO implementation, especially for functions with limited initial data, prioritizes "sample efficiency" – maximizing the improvement per query within a limited budget of function evaluations. This aligns with an Edge AI perspective where gathering more data from the black-box function (which could be expensive or time-consuming) is constrained. The `get_settings(func_id)` (`00e7693a`) further refines this by adapting `xi` (exploration) and candidate generation counts based on the `func_id`, acknowledging that different black-box functions might require different exploration-exploitation balances for efficient optimization.