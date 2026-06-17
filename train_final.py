import sys
import os
import pandas as pd
sys.path.append(os.path.abspath("Classification/DL-MLP"))
import myUtil
import CRF_model
import evaluate

# Read best parameters from Optuna
params = {}
with open("best_training_params.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if ":" in line and "Best Score" not in line:
            key, val = line.strip().split(":")
            params[key.strip()] = float(val.strip())

print("Training final model with 100% data using parameters:", params)

# Initialize and train on 100% data (dry_bean_train.csv)
model = CRF_model.CRF_model_package(
    from_file="",
    lr=params.get('lr', 0.005),
    cycles=int(params.get('cycles', 100)),
    label_smoothing=params.get('label_smoothing', 0.15),
    reconstruct_weight=params.get('reconstruct_weight', 0.2)
)
model.saveModelPackage("Classification\\DL-MLP\\best_CRF_model.pth")

# Inference on test set
model.load_test_data("dry_bean_test.csv")
mc_cycles = int(params.get('mc_cycles', 20))
mc_threshold = params.get('mc_threshold', 0.8)

print(f"Running inference on dry_bean_test.csv with threshold {mc_threshold}...")
all_data, classified, unclassified = model.testingMCDropout(cycles=mc_cycles, threshold=mc_threshold)

# Output results
myUtil.outputResultToFile(all_data, classified, unclassified)
print("Final unclassified_data.csv generated!")
