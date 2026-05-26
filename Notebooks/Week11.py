import pandas as pd
import numpy as np

# This week's input values
inputs = {
    'Function 1': [0.029147, 0.998660],
    'Function 2': [0.979379, 0.003688],
    'Function 3': [0.062352, 0.057412, 0.985938],
    'Function 4': [0.991794, 0.024781, 0.303036, 0.940555],
    'Function 5': [0.297147, 0.052856, 0.994784, 0.032253],
    'Function 6': [0.996326, 0.454797, 0.547773, 0.513762, 0.813782],
    'Function 7': [0.976033, 0.139154, 0.245163, 0.642974, 0.961764, 0.016505],
    'Function 8': [0.032278, 0.002855, 0.479149, 0.084938, 0.295732, 0.931248, 0.914190, 0.042213]
}

# This week's output values
outputs = {
    'Function 1': 0,
    'Function 2': 0.023572116982283457,
    'Function 3': -0.4507615859121511,
    'Function 4': -33.139430805302865,
    'Function 5': 182.24407480442198,
    'Function 6': -1.772574044616747,
    'Function 7': 0.00013521996688955684,
    'Function 8': 8.2348623826071
}

# Create a DataFrame to store the data (optional for query generation, but good for context)
data = []
for func_name in inputs:
    data.append({
        'Function': func_name,
        'Input_Values': inputs[func_name],
        'Output_Value': outputs[func_name],
        'Dimension': len(inputs[func_name])
    })

df = pd.DataFrame(data)

# Display the DataFrame (optional, but shows data structure)
print("Current Black-Box Function Data:\n")
display(df)


print("\nGenerated New Query Points for the 11th Round:")
print("----------------------------------------------")

# Define a noise scale for perturbation
noise_scale = 0.05 # Adjust this value to control the perturbation magnitude

new_queries = {}
for func_name, current_input in inputs.items():
    new_query_values = []
    for val in current_input:
        # Generate a random perturbation within [-noise_scale, noise_scale]
        perturbed_val = val + np.random.uniform(-noise_scale, noise_scale)
        # Ensure values stay within the [0, 1] range
        new_query_values.append(np.clip(perturbed_val, 0.0, 1.0))
    new_queries[func_name] = new_query_values

formatted_queries = []
for func_name, query_values in new_queries.items():
    # Format each dimension to six decimal places and join with a hyphen
    formatted_dims = [f"{val:.6f}" for val in query_values]
    formatted_query = "-".join(formatted_dims)
    formatted_queries.append(f"{func_name}: {formatted_query}")

for query_str in formatted_queries:
    print(query_str)