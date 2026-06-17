import torch.nn as nn
import torch.optim as optim
import torch
import myUtil
import pandas as pd
import numpy as np

class DOC_model(nn.Module):  
    def __init__(self):  
        super(DOC_model, self).__init__()
        
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

class DOC_model_package():
    def __init__(self,from_file="",lr=0.005,cycles=100,min_thresholds=0.5,std_multiplier=1.5) -> None:
        self.network = DOC_model()
        if from_file=="":
            X_train_tensor,Y_train_tensor,self.scaler,self.class_mapping = myUtil.preprocess()
            
            # training
            num_classes = len(self.class_mapping)
            Y_train_onehot = torch.nn.functional.one_hot(Y_train_tensor, num_classes=num_classes).float()
            # cost function (1-vs-rest)
            criterion = nn.BCEWithLogitsLoss()  
            # optimize newwork weight & Learning Rate=0.005
            optimizer = optim.Adam(self.network.parameters(), lr=lr)  

            for _ in range(cycles):
                # switch to train mode  
                self.network.train()
                # clear last time data
                optimizer.zero_grad()
                # get current model predict result
                outputs = self.network(X_train_tensor)
                # compare predict result with answer and count the cost
                loss = criterion(outputs, Y_train_onehot)
                # count the gradient with cost
                loss.backward()
                # update weight
                optimizer.step()

            self.thresholds = self._calculate_doc_thresholds(X_train_tensor, Y_train_tensor, num_classes,min_thresholds,std_multiplier)
        else:
            # read model package
            model_package = torch.load(from_file,weights_only=False)
            self.network.load_state_dict(model_package["model_state_dict"])
            self.scaler = model_package["scaler"]
            self.class_mapping = model_package['class_mapping']
            self.thresholds = model_package['thresholds']

    def _calculate_doc_thresholds(self, X_train, Y_train, num_classes,min_thresholds,std_multiplier):
        """
        Calculate a specific threshold for each known category.
        Using the training set samples, calculate the average and standard 
        deviation of the probability of correctly predicting each category, 
        while satisfying the constraints of a Gaussian distribution.
        """
        self.network.eval()
        with torch.no_grad():
            train_outputs = self.network(X_train)
            train_probs = torch.sigmoid(train_outputs).numpy() 
        
        Y_train_np = Y_train.numpy()
        thresholds = np.zeros(num_classes)
        
        for c in range(num_classes):
            # find all c class sample
            idx = np.where(Y_train_np == c)[0]
            if len(idx) == 0:
                thresholds[c] = min_thresholds # if c not exist
                continue
            
            # sample proba in model
            c_probs = train_probs[idx, c]
            
            mu = np.mean(c_probs)
            sigma = np.std(c_probs)
            
            # set threshold
            thresholds[c] = max(min_thresholds, mu - std_multiplier * sigma)
            print(thresholds[c])
            
        return thresholds

    def testingDOC(self):
        self.network.eval()
        with torch.no_grad():
            test_outputs = self.network(self.X_test_tensor)
            # Independent probabilities obtained using Sigmoid transformation
            test_probabilities = torch.sigmoid(test_outputs).numpy()

        num_samples = test_probabilities.shape[0]
        predicted_idx = test_probabilities.argmax(axis=1)
        
        test_results = self.test_set.copy()
        
        max_probability = test_probabilities.max(axis=1)
        test_results['Trust_index'] = max_probability

        inv_class_mapping = {v: k for k, v in self.class_mapping.items()}  
        
        # classified or unclassified
        predict_labels = []
        is_classified = []
        
        for i in range(num_samples):
            best_class = predicted_idx[i]
            best_prob = test_probabilities[i, best_class]
            
            # check highest prob > thresholds
            if best_prob >= self.thresholds[best_class]:
                predict_labels.append(inv_class_mapping[best_class])
                is_classified.append(True)
            else:
                predict_labels.append("Unseen")
                is_classified.append(False)

        test_results['Predict'] = predict_labels
        
        # Split Data Set
        classified_data_mask = np.array(is_classified)
        
        classified_data = test_results[classified_data_mask].copy()
        classified_data = classified_data.drop(columns=['Trust_index'])

        unclassified_data = test_results[~classified_data_mask].copy()
        unclassified_data = unclassified_data.drop(columns=['Trust_index','Predict'])

        return test_results, classified_data, unclassified_data

    def load_test_data(self,file_name="dry_bean_test.csv"):
        # test set preprocess
        self.test_set = pd.read_csv(file_name)  
        X_test_raw = self.test_set.drop(columns=['Class'])
        X_test_scaled = self.scaler.transform(X_test_raw)
        self.X_test_tensor = torch.FloatTensor(X_test_scaled)

    def saveModelPackage(self,file_name="Classification\\DL-MLP\\DOC_model.pth"):
        # save overall model in file
        model_package = {
            'model_state_dict': self.network.state_dict(),
            'scaler': self.scaler,
            'class_mapping': self.class_mapping,
            'thresholds': self.thresholds
        }
        torch.save(model_package,file_name)
        print("success")
        return model_package

    
if __name__=="__main__":
    DOC_model_package().saveModelPackage()