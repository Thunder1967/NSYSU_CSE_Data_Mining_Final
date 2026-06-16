import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 特徵工程與分析可以用套件
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 引入你自己手刻的分群模型 (假設檔案名稱為 kmeans_model.py)
import kmeans_model 

def analyze_features(df, feature_cols):
    """
    特徵化分析：計算並比較各群集的特徵平均值
    """
    print("\n=== 🔍 特徵化分析結果 ===")
    cluster_means = df.groupby('Cluster_Label')[feature_cols].mean()
    
    # 挑選幾個最具代表性的特徵印出來比較
    key_features = ['Area', 'Perimeter', 'MajorAxisLength', 'Eccentricity', 'roundness']
    print(cluster_means[key_features].T)
    
    return cluster_means

def plot_pca_clusters(X_scaled, labels):
    """
    視覺化分析：使用 PCA 將高維度資料降至 2D 並畫散佈圖
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8)
    
    plt.title('PCA Projection of Unclassified Beans (Custom K-Means)', fontsize=14)
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Discovered Clusters')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 修改這裡：將圖片存入 Clustering 資料夾
    plt.savefig('Clustering/cluster_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("\n📊 已生成並儲存群集視覺化圖片：'Clustering/cluster_pca_visualization.png'")
    plt.show()

def main():
    # 1. 讀取資料
    file_path = "unclassified_data.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 成功讀取資料，共 {df.shape[0]} 筆未知乾豆樣本。")
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {file_path}，請確認路徑。")
        return

    # 2. 特徵前處理 (使用 sklearn 的 StandardScaler)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col not in ['Class', 'Predict']]
    
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 使用你手刻的 K-Means 模型！
    n_clusters = 2
    print(f"\n🚀 正在執行手刻 K-Means 分群 (K={n_clusters})...")
    kmeans = kmeans_model.KMeans_model(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 將結果存回 DataFrame
    df['Cluster_Label'] = cluster_labels
    
    # 4. 評估分群品質
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"\n📈 分群品質評估：")
    print(f" - 輪廓係數 (Silhouette Score): {sil_score:.4f}")

    # 5. 進行特徵化分析與視覺化
    analyze_features(df, feature_cols)
    plot_pca_clusters(X_scaled, cluster_labels)

    # 6. 輸出最終結果
    # 修改這裡：將 CSV 存入 Clustering 資料夾
    output_filename = "Clustering/clustered_new_beans.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n💾 分群與特徵分析完成！結果已存至 '{output_filename}'")

if __name__ == "__main__":
    main()