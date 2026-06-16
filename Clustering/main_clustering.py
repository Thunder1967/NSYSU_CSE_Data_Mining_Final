import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import kmeans_model
import dbscan_model

def analyze_features(df, feature_cols, label_col, model_name):
    """
    特徵化分析：計算並比較各群集的特徵平均值
    """
    print(f"\n=== {model_name} 特徵化分析結果 ===")
    cluster_means = df.groupby(label_col)[feature_cols].mean()
    
    # 挑選幾個最具代表性的特徵印出來比較
    key_features = ['Area', 'Perimeter', 'MajorAxisLength', 'Eccentricity', 'roundness']
    print(cluster_means[key_features].T)
    
    return cluster_means

def plot_pca_clusters(X_scaled, labels, title, filename):
    """
    視覺化分析：使用 PCA 將資料降至 2D 並畫散佈圖
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8)
    
    plt.title(title, fontsize=14)
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Discovered Clusters')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" 已生成並儲存群集視覺化圖片：'{filename}'")
    plt.show()

def main():
    # 1. 讀取資料
    file_path = "unclassified_data.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"成功讀取資料，共 {df.shape[0]} 筆未知乾豆樣本。")
    except FileNotFoundError:
        print(f"找不到檔案 {file_path}，請確認路徑。")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col not in ['Class', 'Predict']]
    X = df[feature_cols].values

    # ==========================================
    # K-Means 最佳化模型
    # ==========================================
    print("\n" + "="*40)
    print(" 執行手刻 K-Means 分群 (Optuna最佳參數)")
    print("="*40)
    # K-Means 最佳前處理：MinMaxScaler, 無 PCA
    scaler_km = MinMaxScaler()
    X_scaled_km = scaler_km.fit_transform(X)
    
    kmeans = kmeans_model.KMeans_model(n_clusters=4, random_state=7, max_iters=572, tol=0.000644649539434322)
    labels_km = kmeans.fit_predict(X_scaled_km)
    df['Cluster_KMeans'] = labels_km
    
    sil_score_km = silhouette_score(X_scaled_km, labels_km)
    print(f" - 輪廓係數 (Silhouette Score): {sil_score_km:.4f}")
    
    analyze_features(df, feature_cols, 'Cluster_KMeans', 'K-Means')
    plot_pca_clusters(X_scaled_km, labels_km, 'PCA Projection (Custom K-Means)', 'Clustering/cluster_pca_kmeans.png')

    # ==========================================
    # DBSCAN 最佳化模型
    # ==========================================
    print("\n" + "="*40)
    print(" 執行手刻 DBSCAN 分群 (Optuna最佳參數)")
    print("="*40)
    # DBSCAN 最佳前處理：StandardScaler, 有 PCA(降至2維)
    scaler_db = StandardScaler()
    X_scaled_db = scaler_db.fit_transform(X)
    
    pca_db = PCA(n_components=2, random_state=42)
    X_processed_db = pca_db.fit_transform(X_scaled_db)
    
    # DBSCAN 最佳參數 (根據截圖)
    dbscan = dbscan_model.DBSCAN_model(eps=0.28276788939414094, min_samples=5)
    labels_db = dbscan.fit_predict(X_processed_db)
    df['Cluster_DBSCAN'] = labels_db
    
    # DBSCAN 可能會產生只有一類或全是雜訊的情況，需防呆
    if len(set(labels_db)) > 1:
        sil_score_db = silhouette_score(X_processed_db, labels_db)
        print(f" - 輪廓係數 (Silhouette Score): {sil_score_db:.4f}")
    else:
        print(" - 輪廓係數 (Silhouette Score): 無法計算 (全為同一群或全為雜訊)")

    analyze_features(df, feature_cols, 'Cluster_DBSCAN', 'DBSCAN')
    plot_pca_clusters(X_processed_db, labels_db, 'PCA Projection (Custom DBSCAN)', 'Clustering/cluster_pca_dbscan.png')

    # ==========================================
    # 輸出最終結果
    # ==========================================
    output_filename = "Clustering/clustered_new_beans.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n 分群與特徵分析完成！結果已存至 '{output_filename}'")

if __name__ == "__main__":
    main()