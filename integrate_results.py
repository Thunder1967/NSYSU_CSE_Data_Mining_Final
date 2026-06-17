import pandas as pd

def main():
    print("=== 開始整合分類與分群結果 ===")
    
    # 1. 讀取高信心度的分類結果
    try:
        df_classified = pd.read_csv("classified_data.csv")
        print(f"[OK] 讀取已分類資料: {len(df_classified)} 筆")
    except Exception as e:
        print("[Error] 找不到 classified_data.csv", e)
        return

    # 2. 讀取未標記但已完成分群的結果
    try:
        df_clustered = pd.read_csv("Clustering/clustered_new_beans.csv")
        print(f"[OK] 讀取已分群資料: {len(df_clustered)} 筆")
    except Exception as e:
        print("[Error] 找不到 Clustering/clustered_new_beans.csv", e)
        return

    # 3. 處理分群標籤 (這裡我們選用表現最好的 K-Means 分群結果)
    # 假設 K-Means 分了 4 群，我們將其重新命名為 New_Species_1, 2... 等
    # 這樣才能跟原本的 SEKER, DERMASON 等字串標籤格式統一
    df_clustered['Final_Predict'] = df_clustered['Cluster_KMeans'].apply(lambda x: f"New_Species_{x}")
    
    # 原本已分類的資料，直接沿用其神經網路的預測結果
    df_classified['Final_Predict'] = df_classified['Predict']

    # 4. 對齊兩個表格的欄位並合併 (整合)
    # 取出共通的特徵欄位與最終預測欄位
    common_cols = list(df_classified.columns)
    if 'Final_Predict' not in common_cols:
        common_cols.append('Final_Predict')
        
    df_final = pd.concat([
        df_classified,
        df_clustered
    ], ignore_index=True)

    # 5. 儲存最終結果
    output_name = "final_integrated_results.csv"
    df_final.to_csv(output_name, index=False)
    print(f"\n[Success] 整合大功告成！總共 {len(df_final)} 筆測試集資料，已儲存至 '{output_name}'")

if __name__ == "__main__":
    main()
