import pandas as pd
import re

# =========================================
# FILE PATHS
# =========================================
input_file = "DoS_attack_dataset.txt"
output_file = "processed_flag.csv"

# =========================================
# REGEX TO EXTRACT CAN ID
# =========================================
pattern = re.compile(r"ID:\s+([0-9A-Fa-f]+)")

# =========================================
# STORAGE
# =========================================
data = []

# =========================================
# READ FILE
# =========================================
with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            can_id_hex = match.group(1)
            can_id_int = int(can_id_hex, 16)

            # Flag logic
            flag = "T" if can_id_int == 0 else "R"

            data.append([can_id_int, flag])

# =========================================
# CREATE DATAFRAME
# =========================================
df = pd.DataFrame(data, columns=["can_id", "flag"])

# =========================================
# SAVE CSV
# =========================================
df.to_csv(output_file, index=False)

print("===================================")
print("File saved as:", output_file)
print("Total rows:", len(df))
print("Attack (T):", (df["flag"] == "T").sum())
print("Regular (R):", (df["flag"] == "R").sum())
print("===================================")