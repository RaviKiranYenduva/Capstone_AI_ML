import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

# Suppress warnings that may arise from fitting models on extremely small datasets,
# as we are primarily demonstrating setup.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Provided input values for the 8 blackbox functions
inputs_raw = {
    1: [0.066859, 0.68851],
    2: [0.814405, 0.968605],
    3: [0.421662, 0.361859, 0.505996],
    4: [0.410884, 0.448224, 0.38438, 0.451683],
    5: [0.310243, 0.852426, 1.0, 0.971167],
    6: [0.323679, 0.300736, 0.482502, 0.805974, 0.052555],
    7: [0.015433, 0.415328, 0.316848, 0.162604, 0.338125, 0.745718],
    8: [0.0, 0.159996, 0.0, 0.012595, 0.88427, 0.305523, 0.0, 0.41985]
}

# Provided output values for the 8 blackbox functions
outputs_raw = {
    1: -1.4689782462102485e-102,
    2: 0.572886680168674,
    3: -0.10203907364088809,
    4: -4.02555701687789,
    5: 1088.8581918803375,
    6: -0.7139010209527445,
    7: 1.3649700705707237,
    8: 9.598482138169
}

# Determine the maximum number of features across all functions to ensure consistent columns
max_features = max(len(v) for v in inputs_raw.values())

# Prepare data for DataFrame creation, padding inputs with NaN
data_rows = []
for i in range(1, 9):
    inputs = inputs_raw[i]
    output = outputs_raw[i]

    padded_inputs = inputs + [np.nan] * (max_features - len(inputs)) # Pad with NaN
    row = {f'feature_{j+1}': padded_inputs[j] for j in range(max_features)}
    row['function_id'] = i # Identifier for the blackbox function
    row['target_value'] = output
    data_rows.append(row)

# Create the DataFrame
df = pd.DataFrame(data_rows)

# Fill NaN values for features. Using 0 assumes missing parameters have no effect,
# which is a common strategy when dealing with sparse or variable-length inputs in a fixed-size model.
df_processed = df.fillna(0)

# Create a binary target for logistic regression: 1 if output is positive, 0 otherwise.
df_processed['is_positive_output'] = (df_processed['target_value'] > 0).astype(int)

print("Processed DataFrame for Blackbox Function Analysis:")
display(df_processed)

print("\n--- Conceptual Model Setup ---")

# Prepare features (X) by dropping identifiers and target columns
X = df_processed.drop(['function_id', 'target_value', 'is_positive_output'], axis=1)

print("\n1. Linear Regression Setup:")
# Target for Linear Regression is the continuous 'target_value'
y_linear = df_processed['target_value']

# Initialize Linear Regression model
linear_model = LinearRegression()
print(f"Linear Regression model can be initialized as: {linear_model}")
print(f"Features (X) shape: {X.shape}, Continuous Target (y_linear) shape: {y_linear.shape}")
print("In a real scenario with sufficient data, you would fit the model using: linear_model.fit(X_train, y_train)")


print("\n2. Logistic Regression Setup:")
# Target for Logistic Regression is the binary 'is_positive_output'
y_logistic = df_processed['is_positive_output']

# Initialize Logistic Regression model
# 'solver' is specified for stability and convergence with small datasets; 'liblinear' is often a good choice.
logistic_model = LogisticRegression(solver='liblinear', random_state=42)
print(f"Logistic Regression model can be initialized as: {logistic_model}")
print(f"Features (X) shape: {X.shape}, Binary Target (y_logistic) shape: {y_logistic.shape}")
print("In a real scenario with sufficient data, you would fit the model using: logistic_model.fit(X_train, y_train)")

print("\nNote: Due to the extremely limited dataset (8 samples), training these models would result in severe overfitting and provide no meaningful or generalizable insights for evaluating the blackbox functions.")
print("The following explanation addresses the theoretical aspects of the assignment.")