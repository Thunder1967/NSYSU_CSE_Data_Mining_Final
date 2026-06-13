import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dry_beans():
    # 1. 檢查檔案是否存在並載入
    train_path = "dry_bean_train.csv"
    test_path = "dry_bean_test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"❌ 錯誤：請確保 {train_path} 和 {test_path} 位於當前工作目錄下！")
        return

    print("📊 正在載入並合併資料集...")
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    
    # 合併兩者以取得最全面的特徵分佈
    df_all = pd.concat([train_set, test_set], axis=0).reset_index(drop=True)

    # 2. 篩選出指定的四個核心品種
    target_classes = ['BARBUNYA', 'CALI', 'SIRA', 'BOMBAY']
    df_filtered = df_all[df_all['Class'].isin(target_classes)].copy()
    
    # 提取 16 項特徵名稱 (排除 Class 欄位)
    features = [col for col in df_filtered.columns if col != 'Class']
    print(f"🔍 成功篩選目標品種！特徵數量：{len(features)} 項，資料總筆數：{len(df_filtered)} 筆")

    # 設定繪圖風格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial'] # 確保圖表英文標籤顯示正常

    # ==================== 圖一：16項特徵的品種差異分析 (Boxplots) ====================
    print("🎨 正在繪製特徵差異箱線圖 (1/2)...")
    # 建立 4x4 的畫布來容納 16 項特徵
    fig, axes = plt.subplots(4, 4, figsize=(20, 18), dpi=100)
    axes = axes.flatten()

    # 調色盤：為四個品種指定固定顏色，方便對比
    palette = {'BARBUNYA': '#3498db', 'CALI': '#2ecc71', 'SIRA': '#e67e22', 'BOMBAY': '#9b59b6'}

    for idx, feature in enumerate(features):
        sns.boxplot(
            data=df_filtered, 
            x='Class', 
            y=feature, 
            ax=axes[idx], 
            palette=palette,
            hue='Class',
            legend=False
        )
        axes[idx].set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        axes[idx].tick_params(axis='x', labelsize=9)

    plt.suptitle('Feature Differences Across 4 Dry Bean Classes', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 儲存與顯示圖一
    boxplot_filename = 'Classification\\bean_feature_differences.png'
    plt.savefig(boxplot_filename, bbox_inches='tight')
    plt.show()
    print(f"💾 差異分析圖已儲存至：{boxplot_filename}")

    # ==================== 圖二：16項特徵之間的關係分析 (Correlation Heatmap) ====================
    print("🎨 正在繪製特徵內在關係熱圖 (2/2)...")
    plt.figure(figsize=(14, 11), dpi=100)
    
    # 計算 16 項特徵之間的皮爾森相關係數 (Pearson Correlation)
    corr_matrix = df_filtered[features].corr()

    # 繪製熱圖
    sns.heatmap(
        corr_matrix, 
        annot=True,          # 顯示相關係數數值
        fmt=".2f",           # 顯示到小數點後兩位
        cmap='coolwarm',     # 藍紅冷暖色調（紅代表正相關，藍代表負相關）
        vmin=-1, vmax=1,     # 限制範圍在 -1 到 1
        linewidths=0.5,      # 格子線條寬度
        annot_kws={"size": 8} # 數值文字大小
    )

    plt.title('Correlation Matrix of 16 Features (Target Classes Only)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 儲存與顯示圖二
    heatmap_filename = 'Classification\\bean_feature_correlation.png'
    plt.savefig(heatmap_filename, bbox_inches='tight')
    # plt.show()
    print(f"💾 關係熱圖已儲存至：{heatmap_filename}")
    print("🏁 所有分析圖表生成完畢！")

if __name__ == "__main__":
    analyze_dry_beans()