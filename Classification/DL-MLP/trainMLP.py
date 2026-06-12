import torch.nn as nn
import torch.optim as optim
import pandas as pd  
import torch
from sklearn.preprocessing import StandardScaler

class BeanClassifier(nn.Module):  
    def __init__(self):  
        super(BeanClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # input to hidden layer 1 
            # extend 16 feature to 128
            nn.Linear(16, 128),  
            # normalize layer
            nn.BatchNorm1d(128),  
            # add nolinear relation
            nn.ReLU(),  
            # forget some data
            nn.Dropout(0.3),  
            
            # hidden layer 1 to hidden layer 2 
            # shrink feature 128 to 64
            nn.Linear(128, 64),  
            # add nolinear relation
            nn.ReLU(),  
            
            # hidden layer 2 to ouput
            # shrink feature 65 to 5 type result
            nn.Linear(64, 5)  
        )  
        
    def forward(self, x):  
        return self.network(x)

def getResultModle(X_train_tensor,Y_train_tensor):
    network = BeanClassifier()  
    # cost function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  
    # optimize newwork weight & Learning Rate=0.005
    optimizer = optim.Adam(network.parameters(), lr=0.005)  

    # training
    for _ in range(100):
        # switch to train mode  
        network.train()
        # clear last time data
        optimizer.zero_grad()
        # get current model predict result
        outputs = network(X_train_tensor)
        # compare predict result with answer and count the cost
        loss = criterion(outputs, Y_train_tensor)
        # count the gradient with cost
        loss.backward()
        # update weight
        optimizer.step()
    
    return network

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

def getModelPackage():
    X_train_tensor,Y_train_tensor,scaler,class_mapping = preprocess()
    network = getResultModle(X_train_tensor,Y_train_tensor)
    model_package = {
        'model_state_dict': network.state_dict(),
        'scaler': scaler,
        'class_mapping': class_mapping
    }
    return model_package

if __name__=="__main__":
    # save overall model in file
    torch.save(getModelPackage(), "Classification\\DL-MLP\\MLP_classifier_model.pth")
    print("success")