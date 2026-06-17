import optuna
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- 合法化防線：自動切分驗證集 ---
if not os.path.exists("dry_bean_train_sub.csv") or not os.path.exists("val_split.csv"):
    df_train = pd.read_csv("dry_bean_train.csv")
    train_sub, val_split = train_test_split(df_train, test_size=0.2, stratify=df_train['Class'], random_state=42)
    train_sub.to_csv("dry_bean_train_sub.csv", index=False)
    val_split.to_csv("val_split.csv", index=False)

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
    m.load_test_data(file_name="val_split.csv")
print("Models loaded successfully.")

def calculate_score(all_data, classified_data, unclassified_data):
    if len(classified_data) == 0:
        return 0.0

    # 1. Classified Accuracy
    # known labels: DERMASON, SIRA, SEKER, HOROZ, CALI
    # (由於驗證集中不包含未知品種，此階段直接最大化已知品種的 Accuracy)
    acc = accuracy_score(classified_data['Class'], classified_data['Predict'])
    
    return acc

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
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*40)
    print("Optimization finished.")
    print("Best validation accuracy: ", study.best_value)
    print("Best parameters: ", study.best_params)
    
    # Save results to CSV
    df = study.trials_dataframe()
    df.to_csv("optuna_tuning_results.csv", index=False)
    print("Saved tuning results to optuna_tuning_results.csv")
