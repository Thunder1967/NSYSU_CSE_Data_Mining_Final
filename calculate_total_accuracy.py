import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=== 計算最終總準確率 (Overall Accuracy) ===")
    
    # 1. 讀取整合後的最終預測結果
    try:
        df = pd.read_csv("final_integrated_results.csv")
    except Exception as e:
        print("[Error] 無法讀取 final_integrated_results.csv:", e)
        return

    # 2. 處理分群標籤的對應關係 (Mapping)
    # 因為分群出來的標籤叫做 New_Species_0, 1... 
    # 我們需要找出這群豆子在真實世界裡「主要」是哪個品種 (BOMBAY 或 BARBUNYA)
    
    # 建立一個新的欄位來放「對應後的最終標籤」
    df['Mapped_Final_Predict'] = df['Final_Predict']
    
    # 找出所有包含 New_Species 的標籤
    cluster_labels = [label for label in df['Final_Predict'].unique() if isinstance(label, str) and label.startswith('New_Species_')]
    
    print("\n[Processing] 正在將分群標籤自動對應至真實品種...")
    mapping_dict = {}
    for cl in cluster_labels:
        # 取出被預測為該群的所有資料
        subset = df[df['Final_Predict'] == cl]
        # 找出這些資料中，真實標籤最多的是哪一個品種
        true_majority_class = subset['Class'].mode()[0]
        mapping_dict[cl] = true_majority_class
        print(f"  - {cl} 主要對應品種為: {true_majority_class} (該群共有 {len(subset)} 顆)")
        
    # 將 New_Species 替換成真實對應的品種名稱
    df['Mapped_Final_Predict'] = df['Mapped_Final_Predict'].replace(mapping_dict)
    
    # 3. 計算總準確率
    y_true = df['Class']
    y_pred = df['Mapped_Final_Predict']
    
    total_acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"[Result] 系統最終總準確率 (Total Accuracy): {total_acc * 100:.2f}%")
    print("="*40)
    
    print("\n[Report] 各品種詳細分類報告 (Classification Report):")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
