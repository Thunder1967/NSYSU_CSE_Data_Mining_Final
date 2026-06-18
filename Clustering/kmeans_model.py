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
        
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # 隨機挑選「第 1 個」群集中心
        first_index = np.random.choice(X.shape[0])
        self.centroids[0] = X[first_index]
        
        # 挑選剩下的 K-1 個中心點
        for i in range(1, self.n_clusters):
    
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids[:i], axis=2)
            min_distances = np.min(distances, axis=1)
            
           
            probs = min_distances ** 2
            probs = probs / np.sum(probs) 
        
            next_index = np.random.choice(X.shape[0], p=probs)
            self.centroids[i] = X[next_index]
            
        for i in range(self.max_iters):
            # 計算距離
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
