import sys
import os
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 將 Clustering 目錄加到 sys.path 方便載入你們手刻的 dbscan_model
sys.path.append(os.path.join(os.path.dirname(__file__), 'Clustering'))
import dbscan_model

def load_data():
    df = pd.read_csv("unclassified_data.csv")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col not in ['Class', 'Predict']]
    return df, feature_cols

df, feature_cols = load_data()
true_labels = df['Class'].values
X_raw = df[feature_cols].values

def objective(trial):
    # ===============================
    # 1. 資料前處理參數搜尋
    # ===============================
    scaler_type = trial.suggest_categorical("scaler", ["StandardScaler", "MinMaxScaler"])
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X_raw)
    
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    if use_pca:
        pca_components = trial.suggest_int("pca_components", 2, 10)
        pca = PCA(n_components=pca_components, random_state=42)
        X_processed = pca.fit_transform(X_scaled)
    else:
        X_processed = X_scaled

    # ===============================
    # 2. 手刻 DBSCAN 模型參數搜尋
    # ===============================
    eps = trial.suggest_float("eps", 0.05, 5.0)
    min_samples = trial.suggest_int("min_samples", 2, 20)
    
    # 初始化手刻 DBSCAN 模型
    dbscan = dbscan_model.DBSCAN_model(
        eps=eps, 
        min_samples=min_samples
    )
    
    # 執行分群 (雜訊點會被標記為 -1)
    cluster_labels = dbscan.fit_predict(X_processed)
    
    # ===============================
    # 3. 給分 (Silhouette Score)
    # ===============================
    # 如果全部分成同一類或是全部都是雜訊，給予極低分
    try:
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
            # 確保有至少兩群 (不含雜訊) 才算輪廓係數，避免作弊
            score = silhouette_score(X_processed, cluster_labels)
        else:
            score = -1
    except:
        score = -1
    
    return score

if __name__ == "__main__":
    print("Starting Optuna Bayesian Optimization for DBSCAN...")
    study = optuna.create_study(direction="maximize")
    # 執行 200 次大亂鬥
    study.optimize(objective, n_trials=200)
    
    study.trials_dataframe().to_csv("optuna_dbscan_results.csv", index=False)
    print("\n=== Best DBSCAN Parameters ===")
    print(f"Best Score (Silhouette Score): {study.best_value:.4f}")
    print("Best Parameter Combination:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    print("\nAll trial records have been saved to 'optuna_dbscan_results.csv'")
