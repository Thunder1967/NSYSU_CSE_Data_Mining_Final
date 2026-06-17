import pandas as pd

def main():
    print("=== 統計 unclassified_data.csv 中各真實品種的數量 ===")
    try:
        df = pd.read_csv("unclassified_data.csv")
    except Exception as e:
        print("[Error] 找不到檔案:", e)
        return
    
    if 'Class' not in df.columns:
        print("[Error] 找不到 'Class' 欄位")
        return
        
    counts = df['Class'].value_counts()
    
    print(f"\n總共被剔除至未分類區 (Unclassified) 的豆子共有: {len(df)} 顆\n")
    print("各品種被剔除數量排行榜:")
    print("-" * 40)
    for idx, (bean_class, count) in enumerate(counts.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  第 {idx} 名 - {bean_class:<10}: {count:>4} 顆 ({percentage:>4.1f}%)")
    print("-" * 40)

if __name__ == "__main__":
    main()
