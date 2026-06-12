import torch.nn as nn
import torch.optim as optim

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
    criterion = nn.CrossEntropyLoss()  
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