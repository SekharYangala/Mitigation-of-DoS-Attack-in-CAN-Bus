# pipeline.py

import subprocess

print("Running clean_data.py")
subprocess.run(["python", "clean_data.py"])

print("Running extract_reference_features.py")
subprocess.run(["python", "extract_reference_features.py"])

print("Running compute_reference_values.py")
subprocess.run(["python", "compute_reference_values.py"])

print("Running run_detection.py")
subprocess.run(["python", "run_detection.py"])

print("Done")