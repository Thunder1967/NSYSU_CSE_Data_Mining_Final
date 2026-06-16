import numpy as np
import pandas as pd
from collections import deque

class DBSCAN_model:
    def __init__(self, eps=1.2, min_samples=5):
        self.eps = eps                 # 鄰居搜尋的半徑距離
        self.min_samples = min_samples # 為核心點所需要的最少鄰居數量（包含自己）

    def fit_predict(self, X):
        # 確保傳入的是 NumPy Array 確保矩陣運算速度
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=float)

        n_samples = X.shape[0]
        # 初始化所有點的標籤為 -1（DBSCAN 預設 -1 代表雜訊/噪音點）
        labels = np.full(n_samples, -1)
        # 用來記錄該點是否已經被主迴圈訪問過
        visited = np.zeros(n_samples, dtype=bool)
        
        # Shape 會是 (n_samples, n_samples)
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

        cluster_id = 0
        # 第一層：主掃描迴圈 (走訪每一顆豆子)
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            # 找出第 i 個點在 eps 半徑內的所有鄰居索引
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            # 判斷是否為核心點
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 暫時歸類為雜訊
            else:
                labels[i] = cluster_id  # 自己加入新群集
                # 啟動第二層：群集擴展迴圈
                self._expand_cluster(i, neighbors, labels, visited, dist_matrix, cluster_id)
                cluster_id += 1  # 這一群收編完畢，群集 ID 加 1

        return labels

    def _expand_cluster(self, core_idx, neighbors, labels, visited, dist_matrix, cluster_id):
        # 擴展密度相連的群集
        queue = deque(neighbors)

        while len(queue) > 0:
            current_point = queue.popleft()

            # 如果這個鄰居還沒被訪問過，就深入去挖它的鄰居
            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = np.where(dist_matrix[current_point] <= self.eps)[0]

                # 如果這個鄰居也是核心點，就把它的鄰居也塞進排隊清單 (Queue) 裡一起擴展
                if len(current_neighbors) >= self.min_samples:
                    # 加入 queue 前先檢查，把已經訪問過的鄰居直接擋在門外
                    new_neighbors = [n for n in current_neighbors if not visited[n]]
                    queue.extend(new_neighbors)

            # 如果這個點還沒被分配到任何群集 (或者是先前被邊緣化的暫時雜訊點)
            if labels[current_point] == -1:
                labels[current_point] = cluster_id