import numpy as np
import pandas as pd

class KMeans_model:
    def __init__(self, n_clusters=2, max_iters=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        np.random.seed(self.random_state)
        
        # 隨機從資料點中抽出 n_clusters 個點作為初始中心
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 計算距離與分派
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            # 找出每個點距離最近的群中心索引
            labels = np.argmin(distances, axis=1)

            # 更新群中心
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    # 計算平均值作為新中心
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = self.centroids[j]

            # 收斂判斷：如果所有中心的移動距離都小於tol，則提早結束
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"模型在第 {i+1} 次迭代時收斂。")
                break
                
            self.centroids = new_centroids

        return labels  
