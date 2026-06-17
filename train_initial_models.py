import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 確保有驗證集
if not os.path.exists("dry_bean_train_sub.csv") or not os.path.exists("val_split.csv"):
    df_train = pd.read_csv("dry_bean_train.csv")
    train_sub, val_split = train_test_split(df_train, test_size=0.2, stratify=df_train['Class'], random_state=42)
    train_sub.to_csv("dry_bean_train_sub.csv", index=False)
    val_split.to_csv("val_split.csv", index=False)

sys.path.append(os.path.abspath("Classification/DL-MLP"))
import myUtil

# Monkey patch: 強制讀取 80% 子訓練集
original_preprocess = myUtil.preprocess
def patched_preprocess(file_name="dry_bean_train_sub.csv"):
    return original_preprocess(file_name=file_name)
myUtil.preprocess = patched_preprocess

import normal_model
import CRF_model
import normal_model_with_IsolationForest

print("Training Normal Model on 80% subset...")
m1 = normal_model.normal_model_package(from_file="")
m1.saveModelPackage("Classification\\DL-MLP\\normal_model.pth")

print("Training CRF Model on 80% subset...")
m2 = CRF_model.CRF_model_package(from_file="")
m2.saveModelPackage("Classification\\DL-MLP\\CRF_model.pth")

print("Training Mix Model on 80% subset...")
m3 = normal_model_with_IsolationForest.mix_model_package(from_file="")
m3.saveModelPackage("Classification\\DL-MLP\\mix_model.pth")

print("Initial models trained and saved!")
