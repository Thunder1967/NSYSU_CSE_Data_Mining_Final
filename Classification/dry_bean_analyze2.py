import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_individual_features():
    # 1. 載入並合併資料
    train_path = "dry_bean_train.csv"
    test_path = "dry_bean_test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"❌ 錯誤：請確保 {train_path} 和 {test_path} 位於當前目錄！")
        return

    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    df_all = pd.concat([train_set, test_set], axis=0).reset_index(drop=True)

    # 2. 篩選目標品種
    target_classes = ['BARBUNYA', 'CALI', 'SIRA', 'BOMBAY']
    df_filtered = df_all[df_all['Class'].isin(target_classes)].copy()
    
    # 提取 16 項特徵
    features = [col for col in df_filtered.columns if col != 'Class']

    # 3. 設定繪圖風格
    sns.set_theme(style="whitegrid")
    
    # 建立 4x4 的大畫布，總共 16 張獨立關係圖
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=100)
    axes = axes.flatten()

    # 為四個品種指定固定顏色（與前面保持一致，方便對照）
    palette = {'BARBUNYA': '#3498db', 'CALI': '#2ecc71', 'SIRA': '#e67e22', 'BOMBAY': '#9b59b6'}

    print("🎨 正在為 16 項特徵各自生成品種關係分佈圖...")
    
    # 4. 迭代每個特徵，單開頻道繪製關係圖
    for idx, feature in enumerate(features):
        # 繪製重疊的密度曲線圖 (KDE Plot)
        sns.kdeplot(
            data=df_filtered,
            x=feature,
            hue='Class',
            palette=palette,
            fill=True,         # 下方區域填滿顏色
            alpha=0.15,        # 透明度，讓重疊處看得清
            linewidth=2,
            ax=axes[idx],
            warn_singular=False
        )
        
        # 優化個別小圖表
        axes[idx].set_title(f'Feature: {feature}', fontsize=12, fontweight='bold', pad=8)
        axes[idx].set_xlabel('') # 移除下方的特徵標籤，靠標題識別即可
        axes[idx].set_ylabel('Density', fontsize=9)
        
        # 只在第一張圖顯示圖例(Legend)，避免 16 張圖都有圖例顯得太雜亂
        if idx != 0:
            axes[idx].get_legend().remove()
        else:
            axes[idx].get_legend().set_title("Bean Classes")

    # 5. 整體佈局調整與儲存
    plt.suptitle('Individual Feature Relationships & Similarity Across 4 Bean Classes', 
                 fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    output_filename = 'Classification\\bean_individual_feature_relations.png'
    plt.savefig(output_filename, bbox_inches='tight')
    # plt.show()
    print(f"💾 16項特徵的獨立關係圖已成功儲存至：{output_filename}")

if __name__ == "__main__":
    analyze_individual_features()