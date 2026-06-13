import pandas as pd  
import torch
from sklearn.preprocessing import StandardScaler

def preprocess(file_name="dry_bean_train.csv"):
    train_set = pd.read_csv(file_name)  
    X_train_raw = train_set.drop(columns=['Class'])  
    Y_train_raw = train_set['Class']  

    # Standardize
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train_raw)  

    # mapping
    class_mapping = {label: idx for idx, label in enumerate(Y_train_raw.unique())}  
    Y_train_encoded = Y_train_raw.map(class_mapping).values  

    # transform to Tensor
    X_train_tensor = torch.FloatTensor(X_train_scaled)  
    Y_train_tensor = torch.LongTensor(Y_train_encoded)

    return X_train_tensor,Y_train_tensor,scaler,class_mapping

def outputResultToFile(all_data,classifiable_data,unclassifiable_data):
    all_data.to_csv("classification_result.csv", index=False)
    classifiable_data.to_csv("classified_data.csv", index=False)
    unclassifiable_data.to_csv("unclassified_data.csv", index=False)
    print("success")