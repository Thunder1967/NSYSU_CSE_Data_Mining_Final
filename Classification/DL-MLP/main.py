import pandas as pd  
import torch
from sklearn.preprocessing import StandardScaler
import trainMLP

# preprocess
train_set = pd.read_csv("dry_bean_train.csv")  
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

# test set preprocess
test_set = pd.read_csv("dry_bean_test.csv")  
X_test_raw = test_set.drop(columns=['Class'])
X_test_scaled = scaler.transform(X_test_raw)
X_test_tensor = torch.FloatTensor(X_test_scaled)


# train
network = trainMLP.getResultModle(X_train_tensor,Y_train_tensor)


# testing
# switch to evaluate mode
network.eval()
with torch.no_grad():
    # get current model predict result
    test_outputs = network(X_test_tensor)

    # transform original score to percentage with Softmax
    test_probabilities = torch.softmax(test_outputs, dim=1).numpy()
  

# Integrate results
test_results = test_set

# get max percentage
max_probability = test_probabilities.max(axis=1)  
predicted_idx = test_probabilities.argmax(axis=1)  
test_results['Trust_index'] = max_probability

# translate index to string label
inv_class_mapping = {v: k for k, v in class_mapping.items()}  
test_results['Predict'] = [inv_class_mapping[idx] for idx in predicted_idx]


# classifiable data
trust_threshold = 0.75  
classifiable_data_mask = test_results['Trust_index'] >= trust_threshold  
classifiable_data = test_results[classifiable_data_mask]

# unclassifiable data
unclassifiable_data = test_results[~classifiable_data_mask]

# output result CSV
classifiable_data.to_csv("classified_data.csv", index=False)
unclassifiable_data.to_csv("unclassified_data.csv", index=False)
