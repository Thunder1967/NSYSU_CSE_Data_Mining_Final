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

# 將 DL-MLP 加入路徑以匯入模型 (不改動原檔案)
sys.path.append(os.path.abspath("Classification/DL-MLP"))
import myUtil

# --- 核心魔法：Monkey Patching 攔截訓練資料 ---
original_preprocess = myUtil.preprocess
def patched_preprocess(file_name="dry_bean_train_sub.csv"):
    return original_preprocess(file_name=file_name)
myUtil.preprocess = patched_preprocess

import normal_model
import CRF_model
import normal_model_with_IsolationForest
import evaluate

def calculate_score(all_data, classified_data, unclassified_data):
    if len(classified_data) == 0:
        return 0.0

    # 1. Classified Accuracy
    # (由於驗證集中不包含未知品種，此階段直接最大化已知品種的 Accuracy)
    acc = accuracy_score(classified_data['Class'], classified_data['Predict'])
    
    return acc

def objective(trial):
    # --- 1. 訓練參數 (Training Parameters) ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cycles = trial.suggest_int("cycles", 50, 200)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    reconstruct_weight = trial.suggest_float("reconstruct_weight", 0.05, 0.5)

    # 鎖定最佳模型 CRF_model，從頭訓練
    try:
        model = CRF_model.CRF_model_package(
            from_file="", lr=lr, cycles=cycles, label_smoothing=label_smoothing, reconstruct_weight=reconstruct_weight
        )
        model.load_test_data(file_name="val_split.csv")
    except Exception as e:
        return 0.0

    # --- 2. 推論參數 (Inference Parameters) ---
    # 鎖定最佳推論方法 MCDropout
    try:
        mc_cycles = trial.suggest_int("mc_cycles", 10, 30)
        mc_threshold = trial.suggest_float("mc_threshold", 0.5, 0.99)
        all_data, classified, unclassified = model.testingMCDropout(cycles=mc_cycles, threshold=mc_threshold)
            
        score = calculate_score(all_data, classified, unclassified)
    except Exception as e:
        return 0.0
        
    return score

if __name__ == "__main__":
    num_trials = 30 
    
    print(f"Starting Multi-Model Training-Inference Joint Optimization for {num_trials} trials...")
    print("This will train new models from scratch in every trial, which may take significant time.")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    
    print("\n" + "="*40)
    print("Optimization finished.")
    print("Best validation accuracy: ", study.best_value)
    print("Best parameters: ", study.best_params)
    
    # 儲存調參過程
    df = study.trials_dataframe()
    df.to_csv("optuna_train_results.csv", index=False)
    print("Saved tuning results to optuna_train_results.csv")
    
    with open("best_training_params.txt", "w") as f:
        f.write(f"Best Score: {study.best_value}\n\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")
