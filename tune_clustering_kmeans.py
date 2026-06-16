import sys
import os
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# 將 Clustering 目錄加到 sys.path 方便載入你們手刻的 kmeans_model
sys.path.append(os.path.join(os.path.dirname(__file__), 'Clustering'))
import kmeans_model

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
    # 2. 手刻 K-Means 模型參數搜尋
    # ===============================
    n_clusters = trial.suggest_int("n_clusters", 2, 4)
    max_iters = trial.suggest_int("max_iters", 100, 1000)
    tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
    random_state = trial.suggest_int("random_state", 0, 1000)
    
    # 初始化手刻模型
    kmeans = kmeans_model.KMeans_model(
        n_clusters=n_clusters, 
        max_iters=max_iters, 
        tol=tol, 
        random_state=random_state
    )
    
    # 執行分群
    cluster_labels = kmeans.fit_predict(X_processed)
    
    # ===============================
    # 3. 給分 (Adjusted Rand Index)
    # ===============================
    # ARI 能精準比較「你們分出來的群」和「真實答案(Class)」的重疊度
    # 1.0 是完美切割，0 則是跟隨機亂猜一樣
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    return ari_score

if __name__ == "__main__":
    print("Starting Optuna Bayesian Optimization: searching for the best clustering and preprocessing parameters...")
    study = optuna.create_study(direction="maximize")
    # 執行 200 次大逃殺
    study.optimize(objective, n_trials=200)
    
    study.trials_dataframe().to_csv("optuna_clustering_results.csv", index=False)
    print("\n=== Best Clustering Parameters ===")
    print(f"Best Score (Adjusted Rand Index): {study.best_value:.4f}")
    print("Best Parameter Combination:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    print("\nAll trial records have been saved to 'optuna_clustering_results.csv'")
