# extract_reference_features.py
# Step 2: Reference Feature Extraction
# Using only NORMAL rows (Flag = R)
# Window sizes: 50,100, 500, 1000
# Features:
#   1. Mean
#   2. Standard Deviation
#   3. Entropy

import pandas as pd
import numpy as np
from collections import Counter

# -----------------------------------
# Load processed dataset from clean_data.py
# -----------------------------------
file_path = "processed_can_data.csv"

df = pd.read_csv(file_path)

print("Processed dataset loaded")
print("Shape:", df.shape)

# -----------------------------------
# Keep only NORMAL rows
# Using only R frames
# -----------------------------------
normal_df = df[df["Flag"] == "R"].copy()

print("\nOnly Normal (R) rows selected")
print("Shape:", normal_df.shape)

# -----------------------------------
# Function to calculate entropy
# -----------------------------------
def calculate_entropy(values):
    total = len(values)

    if total == 0:
        return 0

    counts = Counter(values)
    probabilities = [count / total for count in counts.values()]

    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    return entropy


# -----------------------------------
# Function to calculate
# Mean + SD + Entropy for each window
# -----------------------------------
def extract_window_features(data, window_size):
    results = []

    total_rows = len(data)

    for start in range(0, total_rows, window_size):
        end = start + window_size

        # Skip incomplete last window
        if end > total_rows:
            break

        window = data.iloc[start:end]

        can_ids = window["CAN_ID"].tolist()

        # Frequency of each CAN ID
        freq_counts = Counter(can_ids)
        freq_values = list(freq_counts.values())

        # Mean of frequency
        mean_value = np.mean(freq_values)

        # Standard deviation of frequency
        std_value = np.std(freq_values)

        # Entropy of CAN IDs
        entropy_value = calculate_entropy(can_ids)

        results.append([
            start,
            end,
            window_size,
            mean_value,
            std_value,
            entropy_value
        ])

    return pd.DataFrame(results, columns=[
        "Start_Row",
        "End_Row",
        "Window_Size",
        "Mean",
        "Standard_Deviation",
        "Entropy"
    ])


# -----------------------------------
# Window sizes to test
# -----------------------------------
window_sizes = [50,100, 500, 1000]

all_results = []

for w in window_sizes:
    print(f"\nProcessing Window Size = {w}")

    result_df = extract_window_features(normal_df, w)

    print(result_df.head())

    all_results.append(result_df)

# -----------------------------------
# Combine all reference datasets
# -----------------------------------
final_reference_df = pd.concat(all_results, ignore_index=True)

print("\nFinal Reference Dataset Created")
print(final_reference_df.head(20))

# -----------------------------------
# Save output
# -----------------------------------
output_file = "reference_features.csv"
final_reference_df.to_csv(output_file, index=False)

print(f"\nReference feature dataset saved as: {output_file}")