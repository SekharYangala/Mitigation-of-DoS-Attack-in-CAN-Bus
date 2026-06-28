
# run_detection.py

import pandas as pd
import numpy as np
from collections import Counter

# =====================================================
# USER OPTION
# =====================================================
# True  -> Run entropy fine tuning every time
# False -> Use fixed best gamma3 values

RUN_FINE_TUNING = False

# Previously found best gamma3 values
# fixed_gamma3 = {
#     50: 5.0,
#     100: 2.8,
#     500: 3.0,
#     1000: 3.6
    #for 2.5million
# }

fixed_gamma3= {
    50:4.4,
    100:2.8,
    500:3.2,
    1000:3.7
    #for 3.5 million
}

# =====================================================
# LOAD DATA
# =====================================================

data_file = "processed_can_data.csv"
ref_file = "normal_reference_values.csv"

df = pd.read_csv(data_file)
ref_df = pd.read_csv(ref_file)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# =====================================================
# CLEAN DATA
# =====================================================

df["CAN_ID"] = df["CAN_ID"].astype(str).str.strip()
df["Flag"] = df["Flag"].astype(str).str.strip()

# =====================================================
# PAPER THRESHOLDS
# =====================================================

paper_thresholds = {
    50: {"gamma1": 5.0, "gamma2": 5.0},
    100: {"gamma1": 3.0, "gamma2": 4.3},
    500: {"gamma1": 1.8, "gamma2": 2.5},
    1000: {"gamma1": 1.1, "gamma2": 1.6}
}

window_sizes = [50, 100, 500, 1000]

# =====================================================
# ATTACK RATIO THRESHOLD
# =====================================================

alpha = 0.02

# =====================================================
# ENTROPY FUNCTION
# =====================================================

def calculate_entropy(values):
    total = len(values)

    if total == 0:
        return 0

    counts = Counter(values)
    probabilities = [count / total for count in counts.values()]

    entropy = -sum(
        p * np.log2(p)
        for p in probabilities
        if p > 0
    )

    return entropy


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(window):
    can_ids = window["CAN_ID"].tolist()

    freq_counts = Counter(can_ids)
    freq_values = list(freq_counts.values())

    mean_value = np.mean(freq_values)
    std_value = np.std(freq_values)
    entropy_value = calculate_entropy(can_ids)

    return mean_value, std_value, entropy_value


# =====================================================
# WINDOW LABEL USING ATTACK RATIO
# =====================================================

def get_actual_label(window, window_size):
    attack_frames = (window["Flag"] == "T").sum()
    attack_ratio = attack_frames / window_size

    if attack_ratio >= alpha:
        return 1
    return 0


# =====================================================
# FINAL SUMMARY STORAGE
# =====================================================

final_summary = []

# =====================================================
# MAIN TESTING
# =====================================================

for w in window_sizes:

    print("\n" + "=" * 80)
    print(f"WINDOW SIZE = {w}")
    print("=" * 80)

    # -----------------------------------
    # Load reference values
    # -----------------------------------

    ref = ref_df[ref_df["Window_Size"] == w].iloc[0]

    mu_a = ref["mu_a"]
    sigma_a = ref["sigma_a"]

    mu_s = ref["mu_s"]
    sigma_s = ref["sigma_s"]

    mu_e = ref["mu_e"]
    sigma_e = ref["sigma_e"]

    gamma1 = paper_thresholds[w]["gamma1"]
    gamma2 = paper_thresholds[w]["gamma2"]

    print("\nPaper Thresholds:")
    print(f"gamma1 (Mean) = {gamma1}")
    print(f"gamma2 (SD)   = {gamma2}")
    print(f"Attack Ratio Threshold alpha = {alpha}")

    total_rows = len(df)
    total_windows = total_rows // w

    best_te = float("inf")
    best_gamma3 = None

    best_TP = 0
    best_TN = 0
    best_FP = 0
    best_FN = 0

    total_attack_windows = 0
    total_normal_windows = 0
    best_correct = 0

    # =================================================
    # OPTION 1: Fine Tuning
    # =================================================

    if RUN_FINE_TUNING:
        print("\nRunning Fine Tuning for gamma3...")

        gamma3_values = np.arange(1.0, 5.1, 0.2)

    # =================================================
    # OPTION 2: Use Fixed gamma3
    # =================================================

    else:
        print("\nUsing Fixed gamma3 Value...")

        gamma3_values = [fixed_gamma3[w]]

    # =================================================
    # Testing loop
    # =================================================

    for gamma3 in gamma3_values:

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        attack_windows = 0
        normal_windows = 0

        for start in range(0, total_rows, w):
            end = start + w

            if end > total_rows:
                break

            window = df.iloc[start:end]

            # ----------------------------
            # Actual label
            # ----------------------------

            actual = get_actual_label(window, w)

            if actual == 1:
                attack_windows += 1
            else:
                normal_windows += 1

            # ----------------------------
            # Feature extraction
            # ----------------------------

            mean_i, std_i, entropy_i = extract_features(window)

            # ----------------------------
            # Deviation scores
            # ----------------------------

            mean_score = abs(mean_i - mu_a) / (sigma_a + 1e-9)
            std_score = abs(std_i - mu_s) / (sigma_s + 1e-9)
            entropy_score = abs(entropy_i - mu_e) / (sigma_e + 1e-9)

            # ----------------------------
            # Prediction
            # ----------------------------

            if (
                mean_score > gamma1
                or std_score > gamma2
                  #entropy_score > gamma3
            ):
                predicted = 1
            else:
                predicted = 0

            # ----------------------------
            # Confusion Matrix
            # ----------------------------

            if actual == 1 and predicted == 1:
                TP += 1
            elif actual == 0 and predicted == 0:
                TN += 1
            elif actual == 0 and predicted == 1:
                FP += 1
            elif actual == 1 and predicted == 0:
                FN += 1

        # -----------------------------------
        # Total Error
        # -----------------------------------

        TE = FP + FN
        correct_predictions = TP + TN

        print(
            f"gamma3={gamma3:.1f} | "
            f"Total Windows={total_windows} | "
            f"Normal={normal_windows} | "
            f"Attack={attack_windows} | "
            f"Correct={correct_predictions} | "
            f"FP={FP} FN={FN} TP={TP} TN={TN} | "
            f"TE={TE}"
        )

        # -----------------------------------
        # Best Result Selection
        # -----------------------------------

        if TE < best_te:
            best_te = TE
            best_gamma3 = gamma3

            best_TP = TP
            best_TN = TN
            best_FP = FP
            best_FN = FN

            total_attack_windows = attack_windows
            total_normal_windows = normal_windows
            best_correct = correct_predictions

    # =================================================
    # PERFORMANCE METRICS
    # =================================================

    accuracy = (
        (best_TP + best_TN)
        / (best_TP + best_TN + best_FP + best_FN + 1e-9)
    ) * 100

    precision = (
        best_TP
        / (best_TP + best_FP + 1e-9)
    ) * 100

    recall = (
        best_TP
        / (best_TP + best_FN + 1e-9)
    ) * 100

    f1_score = (
        2 * precision * recall
        / (precision + recall + 1e-9)
    )

    false_positive_rate = (
        best_FP
        / (best_FP + best_TN + 1e-9)
    ) * 100

    detection_rate = recall

    # =================================================
    # FINAL RESULT
    # =================================================

    print("\nBEST RESULT")
    print(f"Window Size           = {w}")
    print(f"Total Windows         = {total_windows}")
    print(f"Normal Windows        = {total_normal_windows}")
    print(f"Attack Windows        = {total_attack_windows}")
    print(f"Correct Predictions   = {best_correct}")
    print(f"Best gamma3 (Entropy) = {best_gamma3}")
    print(f"Minimum TE            = {best_te}")

    print("\nPERFORMANCE METRICS")
    print(f"Accuracy              = {accuracy:.2f}%")
    print(f"Precision             = {precision:.2f}%")
    print(f"Recall (TPR)          = {recall:.2f}%")
    print(f"F1-Score              = {f1_score:.2f}")
    print(f"False Positive Rate   = {false_positive_rate:.2f}%")
    print(f"Detection Rate        = {detection_rate:.2f}%")

    # =================================================
    # Save Summary
    # =================================================

    final_summary.append([
        w,
        best_gamma3,
        best_te,
        accuracy,
        precision,
        recall,
        f1_score,
        false_positive_rate,
        detection_rate
    ])

# =====================================================
# FINAL SUMMARY TABLE
# =====================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY OF ALL WINDOW SIZES")
print("=" * 100)

summary_df = pd.DataFrame(final_summary, columns=[
    "Window_Size",
    "Best_gamma3",
    "Minimum_TE",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "False_Positive_Rate",
    "Detection_Rate"
])

print(summary_df)

summary_df.to_csv(
    "final_performance_summary.csv",
    index=False
)

print("\nSaved as final_performance_summary.csv")
print("\nProcess Completed Successfully")