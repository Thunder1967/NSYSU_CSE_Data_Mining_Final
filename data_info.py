# following code made by gemini
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_combined_files(file_list,output_name="data_info.png"):
    all_counts = {}
    all_qualities = set()

    # 1. 收集所有檔案的數據
    for fileName in file_list:
        df = pd.read_csv(fileName)
        val, cnt = np.unique(df['Class'], return_counts=True)
        cnt = cnt/np.sum(cnt)
        all_counts[fileName] = dict(zip(val, cnt))
        all_qualities.update(val)

    # 排序所有的品質分數，作為 X 軸
    sorted_qualities = sorted(list(all_qualities))
    
    # 2. 設定繪圖參數
    x = np.arange(len(sorted_qualities))  # X 軸基礎位置
    width = 0.25  # 每個長條的寬度
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. 繪製每個檔案的長條
    for fileName, counts in all_counts.items():
        # 確保每個分數都有對應數值，若無則補 0
        measurement = [counts.get(q, 0) for q in sorted_qualities]
        
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=fileName)
        ax.bar_label(rects, padding=3,fmt='%.3f')
        multiplier += 1

    # 4. 圖表修飾
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution Comparison')
    ax.set_xticks(x + width, sorted_qualities) # 讓 X 軸刻度居中
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 5. 匯出與顯示
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)

plot_combined_files(["dry_bean_train.csv","dry_bean_test.csv"])