import pandas as pd 

# 分群不會用到 "信心指數和分類的預測結果"
unclassified_data = pd.read_csv("unclassified_data.csv")
unclassified_data = unclassified_data.drop(columns=['Trust_index','Predict'])