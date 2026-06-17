import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import kmeans_model, dbscan_model

def analyze_features(df, feature_cols, cluster_col, title_prefix):
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    print("\n=== " + title_prefix + " 各群特徵分析 ===")
    print("各群數量 (Cluster Sizes):")
    print(df[cluster_col].value_counts().sort_index())
    print("-" * 40)
    print(cluster_means.T)

def plot_pca_clusters(X_pca, labels, title, filename):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for l in unique_labels:
        mask = labels == l
        color = 'gray' if l == -1 else None
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {l}', alpha=0.7, c=color)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    if not os.path.exists('Clustering'):
        os.makedirs('Clustering')

    df = pd.read_csv('unclassified_data.csv')
    if df.empty:
        print("沒有未分類的資料可供分群。")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col not in ['Class', 'Predict']]
    X = df[feature_cols].values

    print("\n" + "="*40)
    print(" 執行 K-Means 分群 (Optuna最佳參數)")
    print("="*40)
    scaler_km = MinMaxScaler()
    X_scaled_km = scaler_km.fit_transform(X)
    pca_km = PCA(n_components=2, random_state=42)
    X_processed_km = pca_km.fit_transform(X_scaled_km)
    
    kmeans = kmeans_model.KMeans_model(n_clusters=5, random_state=244, max_iters=936, tol=2.563312135760017e-05)
    labels_km = kmeans.fit_predict(X_processed_km)
    df['Cluster_KMeans'] = labels_km
    
    sil_score_km = silhouette_score(X_processed_km, labels_km)
    print(f" - 輪廓係數 (Silhouette Score): {sil_score_km:.4f}")
    
    analyze_features(df, feature_cols, 'Cluster_KMeans', 'K-Means')
    plot_pca_clusters(X_processed_km, labels_km, 'PCA Projection (Custom K-Means)', 'Clustering/cluster_pca_kmeans.png')

    print("\n" + "="*40)
    print(" 執行 DBSCAN 分群 (Optuna最佳參數)")
    print("="*40)
    scaler_db = StandardScaler()
    X_scaled_db = scaler_db.fit_transform(X)
    pca_db = PCA(n_components=6, random_state=42)
    X_processed_db = pca_db.fit_transform(X_scaled_db)
    
    dbscan = dbscan_model.DBSCAN_model(eps=1.686262751166031, min_samples=16)
    labels_db = dbscan.fit_predict(X_processed_db)
    df['Cluster_DBSCAN'] = labels_db
    
    unique_labels = set(labels_db)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        sil_score_db = silhouette_score(X_processed_db, labels_db)
        print(f" - 輪廓係數 (Silhouette Score): {sil_score_db:.4f}")
    else:
        print(" - 輪廓係數 (Silhouette Score): 無法計算 (全為同一群或全為雜訊)")

    analyze_features(df, feature_cols, 'Cluster_DBSCAN', 'DBSCAN')
    
    # We can only plot 2D PCA, so let's make a 2D projection for plotting
    pca_plot = PCA(n_components=2, random_state=42)
    X_plot_db = pca_plot.fit_transform(X_scaled_db)
    plot_pca_clusters(X_plot_db, labels_db, 'PCA Projection (Custom DBSCAN)', 'Clustering/cluster_pca_dbscan.png')

    df.to_csv('Clustering/clustered_new_beans.csv', index=False)
    print("\n 分群特徵與結果已儲存至 'Clustering/clustered_new_beans.csv'")

if __name__ == "__main__":
    main()