import optuna
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath("Classification/DL-MLP"))

import normal_model
import CRF_model
import normal_model_with_IsolationForest
import evaluate

# Pre-load models to save time
print("Loading models...")
models = {
    "normal": normal_model.normal_model_package(from_file="Classification\\DL-MLP\\normal_model.pth"),
    "CRF": CRF_model.CRF_model_package(from_file="Classification\\DL-MLP\\CRF_model.pth"),
    "Mix": normal_model_with_IsolationForest.mix_model_package(from_file="Classification\\DL-MLP\\mix_model.pth")
}

for m in models.values():
    m.load_test_data()
print("Models loaded successfully.")

def calculate_score(all_data, classified_data, unclassified_data):
    if len(classified_data) == 0:
        return 0.0

    # 1. Classified Accuracy
    # known labels: DERMASON, SIRA, SEKER, HOROZ, CALI
    acc = accuracy_score(classified_data['Class'], classified_data['Predict'])
    
    # 2. OOD F1-score
    p = evaluate.newClassPrecision(unclassified_data)
    r = evaluate.newClassRecall(all_data, unclassified_data)
    
    f1 = 0.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r) / (p + r)
        
    # Combined score (alpha = 0.5)
    combined_score = 0.5 * acc + 0.5 * f1
    return combined_score

def objective(trial):
    model_name = trial.suggest_categorical("model_name", ["normal", "CRF", "Mix"])
    method = trial.suggest_categorical("method", ["MCDropout", "Energy", "Temperature"])
    
    model = models[model_name]
    
    if method == "MCDropout":
        # Integer cycles, float threshold
        cycles = trial.suggest_int("mc_cycles", 5, 30)
        threshold = trial.suggest_float("mc_threshold", 0.5, 0.999)
        all_data, classified, unclassified = model.testingMCDropout(cycles=cycles, threshold=threshold)
        
    elif method == "Energy":
        T = trial.suggest_float("energy_T", 0.1, 10.0)
        threshold = trial.suggest_float("energy_threshold", -20.0, 5.0)
        all_data, classified, unclassified = model.testingEnergy(threshold=threshold, T=T)
        
    elif method == "Temperature":
        T = trial.suggest_float("temp_T", 0.1, 10.0)
        threshold = trial.suggest_float("temp_threshold", 0.5, 0.999)
        all_data, classified, unclassified = model.testingTemperature(T=T, threshold=threshold)
        
    score = calculate_score(all_data, classified, unclassified)
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("Starting optimization...")
    study.optimize(objective, n_trials=100)
    
    print("\n" + "="*40)
    print("Optimization finished.")
    print("Best combined score: ", study.best_value)
    print("Best parameters: ", study.best_params)
    
    # Save results to CSV
    df = study.trials_dataframe()
    df.to_csv("optuna_tuning_results.csv", index=False)
    print("Saved tuning results to optuna_tuning_results.csv")
