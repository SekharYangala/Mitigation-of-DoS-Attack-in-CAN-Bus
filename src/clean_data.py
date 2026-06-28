# clean_data.py
# Step 1: Data Loading + Basic Preprocessing
# Using only first 300,0000 rows
# Keeping only CAN_ID and last column (Flag: R/T)

import pandas as pd

# -----------------------------------
# Dataset path
# -----------------------------------
file_path = "DoS_dataset.csv"  

# -----------------------------------
# Column names based on your dataset
# -----------------------------------
columns = [
    "Timestamp",
    "CAN_ID",
    "DLC",
    "DATA_0",
    "DATA_1",
    "DATA_2",
    "DATA_3",
    "DATA_4",
    "DATA_5",
    "DATA_6",
    "DATA_7",
    "Flag"
]

# -----------------------------------
# Load first 3,500,000 rows only
# -----------------------------------
df = pd.read_csv(
    file_path,
    names=columns,
    nrows=3500000,
    header=None
)

print("Dataset loaded successfully")
print("Shape:", df.shape)

# -----------------------------------
# Fill missing values with 0
# -----------------------------------
df = df.fillna(0)

# -----------------------------------
# Keep only required columns for now
# CAN_ID + Flag
# -----------------------------------
df = df[["CAN_ID", "Flag"]]

# -----------------------------------
# Clean CAN_ID
# Convert to string and remove spaces
# Example: 0316 stays as "0316"
# -----------------------------------
df["CAN_ID"] = df["CAN_ID"].astype(str).str.strip()

# -----------------------------------
# Clean Flag column
# R = Received frame
# T = Transmitted frame
#
# Usually:
# R -> Normal observed CAN traffic
# T -> Injected / attack generated traffic
#
# (depends on dataset source)
# -----------------------------------
df["Flag"] = df["Flag"].astype(str).str.strip()

# -----------------------------------
# Replace empty values if any
# -----------------------------------
df["Flag"] = df["Flag"].replace("", "Unknown")

# -----------------------------------
# Basic checks
# -----------------------------------
print("\nUnique Flags Found:")
print(df["Flag"].unique())

print("\nSample Data:")
print(df.head(10))

# -----------------------------------
# Save cleaned output
# -----------------------------------
output_file = "processed_can_data.csv"
df.to_csv(output_file, index=False)

print(f"\nProcessed file saved as: {output_file}")