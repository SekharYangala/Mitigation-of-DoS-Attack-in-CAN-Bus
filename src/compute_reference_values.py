# compute_reference_values.py
# Step 3: Calculate Reference Values of Normality
# For each window size (100, 500, 1000)
# Calculate:
# 1. Average of Mean values        -> mu_a
# 2. SD of Mean values             -> sigma_a
# 3. Average of SD values          -> mu_s
# 4. SD of SD values               -> sigma_s
# 5. Average of Entropy values     -> mu_e
# 6. SD of Entropy values          -> sigma_e

import pandas as pd
import numpy as np

# -----------------------------------
# Load reference feature dataset
# created from extract_reference_features.py
# -----------------------------------
file_path = "reference_features.csv"

df = pd.read_csv(file_path)

print("Reference feature dataset loaded")
print("Shape:", df.shape)

# -----------------------------------
# Window sizes to process
# -----------------------------------
window_sizes = [50,100, 500, 1000]

final_results = []

# -----------------------------------
# Process each window size separately
# -----------------------------------
for w in window_sizes:
    print(f"\nProcessing Window Size = {w}")

    temp = df[df["Window_Size"] == w].copy()

    # -----------------------------
    # Mean statistics
    # -----------------------------
    mu_a = temp["Mean"].mean()
    sigma_a = temp["Mean"].std()

    # -----------------------------
    # Standard Deviation statistics
    # -----------------------------
    mu_s = temp["Standard_Deviation"].mean()
    sigma_s = temp["Standard_Deviation"].std()

    # -----------------------------
    # Entropy statistics
    # -----------------------------
    mu_e = temp["Entropy"].mean()
    sigma_e = temp["Entropy"].std()

    # -----------------------------
    # Print results clearly
    # -----------------------------
    print(f"\nWindow Size = {w}")

    print(f"Average of Means (mu_a) = {mu_a:.6f}")
    print(f"SD of Means (sigma_a) = {sigma_a:.6f}")

    print(f"Average of SDs (mu_s) = {mu_s:.6f}")
    print(f"SD of SDs (sigma_s) = {sigma_s:.6f}")

    print(f"Average of Entropy (mu_e) = {mu_e:.6f}")
    print(f"SD of Entropy (sigma_e) = {sigma_e:.6f}")

    # -----------------------------
    # Store results
    # -----------------------------
    final_results.append([
        w,
        mu_a,
        sigma_a,
        mu_s,
        sigma_s,
        mu_e,
        sigma_e
    ])

# -----------------------------------
# Final dataframe
# -----------------------------------
final_df = pd.DataFrame(final_results, columns=[
    "Window_Size",
    "mu_a",
    "sigma_a",
    "mu_s",
    "sigma_s",
    "mu_e",
    "sigma_e"
])

print("\nFinal Reference Normality Values")
print(final_df)

# -----------------------------------
# Save output
# -----------------------------------
output_file = "normal_reference_values.csv"
final_df.to_csv(output_file, index=False)

print(f"\nSaved as: {output_file}")