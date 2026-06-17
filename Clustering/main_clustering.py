import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

import kmeans_model
import dbscan_model

def analyze_features(df, feature_cols, label_col, model_name):
    """
    特徵化分析：計算並比較各群集的特徵平均值
    """
    print(f"\n=== {model_name} 特徵化分析結果 ===")
    
    # 計算並印出每一群有幾顆豆子
    cluster_counts = df[label_col].value_counts().sort_index()
    print(" 各群集資料筆數 (Cluster Sizes):")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Cluster {cluster_id} (雜訊/Noise): {count} 顆")
        else:
            print(f"  Cluster {cluster_id}: {count} 顆")
    print("-" * 40)
    
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

    # 🌟 修改處 1：把 StandardScaler 移到最上面，建立一個「共用畫布」
    # 這樣後面的 K-Means 和 DBSCAN 畫圖時都可以拿它來當底圖
    scaler_shared = StandardScaler()
    X_scaled_shared = scaler_shared.fit_transform(X)

    # ==========================================
    # K-Means 最佳化模型
    # ==========================================
    print("\n" + "="*40)
    print(" 執行手刻 K-Means 分群 (Optuna最佳參數)")
    print("="*40)
    # K-Means 訓練專用前處理：MinMaxScaler (不影響畫圖)
    scaler_km = MinMaxScaler()
    X_scaled_km = scaler_km.fit_transform(X)
    
    kmeans = kmeans_model.KMeans_model(n_clusters=4, random_state=7, max_iters=572, tol=0.000644649539434322)
    
    # K-Means 依然在 MinMaxScaler 的空間裡「算出標籤」
    labels_km = kmeans.fit_predict(X_scaled_km)
    df['Cluster_KMeans'] = labels_km
    
    sil_score_km = silhouette_score(X_scaled_km, labels_km)
    ari_km = adjusted_rand_score(df['Class'], labels_km)
    print(f" - 輪廓係數 (Silhouette Score): {sil_score_km:.4f}")
    print(f" - ARI 分數 (Adjusted Rand Index): {ari_km:.4f}")
    
    analyze_features(df, feature_cols, 'Cluster_KMeans', 'K-Means')
    
    # 🌟 修改處 2：畫圖時，傳入「共用畫布 (X_scaled_shared)」，而不是 X_scaled_km
    # 這樣點的位置就會跟 DBSCAN 完全對齊！
    plot_pca_clusters(X_scaled_shared, labels_km, 'PCA Projection (Custom K-Means Aligned)', 'Clustering/cluster_pca_kmeans.png')

    # ==========================================
    # DBSCAN 最佳化模型
    # ==========================================
    print("\n" + "="*40)
    print(" 執行手刻 DBSCAN 分群 (Optuna最佳參數)")
    print("="*40)
    
    # 🌟 修改處 3：DBSCAN 直接使用剛剛建好的共用畫布進行後續降維與訓練
    pca_db = PCA(n_components=2, random_state=42)
    X_processed_db = pca_db.fit_transform(X_scaled_shared)
    
    # DBSCAN 最佳參數
    dbscan = dbscan_model.DBSCAN_model(eps=0.28276788939414094, min_samples=5)
    labels_db = dbscan.fit_predict(X_processed_db)
    df['Cluster_DBSCAN'] = labels_db
    
    # DBSCAN 防呆與計分
    if len(set(labels_db)) > 1:
        sil_score_db = silhouette_score(X_processed_db, labels_db)
        print(f" - 輪廓係數 (Silhouette Score): {sil_score_db:.4f}")
    else:
        print(" - 輪廓係數 (Silhouette Score): 無法計算 (全為同一群或全為雜訊)")
        
    ari_db = adjusted_rand_score(df['Class'], labels_db)
    print(f" - ARI 分數 (Adjusted Rand Index): {ari_db:.4f}")

    analyze_features(df, feature_cols, 'Cluster_DBSCAN', 'DBSCAN')
    
    # 🌟 修改處 4：DBSCAN 畫圖時，同樣傳入共用畫布
    plot_pca_clusters(X_scaled_shared, labels_db, 'PCA Projection (Custom DBSCAN)', 'Clustering/cluster_pca_dbscan.png')

    # ==========================================
    # 輸出最終結果
    # ==========================================
    output_filename = "Clustering/clustered_new_beans.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n 分群與特徵分析完成！結果已存至 '{output_filename}'")

if __name__ == "__main__":
    main()