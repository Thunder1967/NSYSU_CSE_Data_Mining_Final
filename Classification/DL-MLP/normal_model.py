import torch.nn as nn
import torch.optim as optim
import torch
import myUtil
import pandas as pd

class normal_model(nn.Module):  
    def __init__(self):  
        super(normal_model, self).__init__()
        
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

class normal_model_package():
    def __init__(self,from_file="",label_smoothing=0.15,lr=0.005,cycles=100) -> None:
        self.network = normal_model()
        if from_file=="":
            X_train_tensor,Y_train_tensor,self.scaler,self.class_mapping = myUtil.preprocess()
            
            # training
            # cost function
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  
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
                loss = criterion(outputs, Y_train_tensor)
                # count the gradient with cost
                loss.backward()
                # update weight
                optimizer.step()
        else:
            # read model package
            model_package = torch.load(from_file,weights_only=False)
            self.network.load_state_dict(model_package["model_state_dict"])
            self.scaler = model_package["scaler"]
            self.class_mapping = model_package['class_mapping']

    def load_test_data(self,file_name="dry_bean_test.csv"):
        # test set preprocess
        self.test_set = pd.read_csv(file_name)  
        X_test_raw = self.test_set.drop(columns=['Class'])
        X_test_scaled = self.scaler.transform(X_test_raw)
        self.X_test_tensor = torch.FloatTensor(X_test_scaled)

    def testingTemperature(self,T=1,threshold=0.8):
        # switch to evaluate mode
        self.network.eval()
        with torch.no_grad():
            # get current model predict result
            test_outputs = self.network(self.X_test_tensor)

            # transform original score to percentage with Softmax
            test_probabilities = torch.softmax(test_outputs/T, dim=1).numpy()
        return self.combineResult(test_probabilities,threshold)

    def testingMCDropout(self,cycles=10,threshold=0.8):
        # switch to evaluate mode
        self.network.eval()
        # open Dropout
        for module in self.network.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        all_probs = []
        for _ in range(cycles):
            with torch.no_grad():
                # get current model predict result
                test_outputs = self.network(self.X_test_tensor)

                # transform original score to percentage with Softmax
                all_probs.append(torch.softmax(test_outputs, dim=1))
        test_probabilities = torch.stack(all_probs, dim=0).mean(dim=0).numpy()
        return self.combineResult(test_probabilities,threshold)

    def combineResult(self,test_probabilities,threshold):
        # get max percentage
        max_probability = test_probabilities.max(axis=1)  
        predicted_idx = test_probabilities.argmax(axis=1)
        test_results = self.test_set.copy()
        test_results['Trust_index'] = max_probability

        # translate index to string label
        inv_class_mapping = {v: k for k, v in self.class_mapping.items()}  
        test_results['Predict'] = [inv_class_mapping[idx] for idx in predicted_idx]

        # classified data
        classified_data_mask = test_results['Trust_index'] >= threshold  
        classified_data = test_results[classified_data_mask].copy()
        classified_data = classified_data.drop(columns=['Trust_index'])

        # unclassified data
        unclassified_data = test_results[~classified_data_mask].copy()
        unclassified_data = unclassified_data.drop(columns=['Trust_index','Predict'])

        return test_results,classified_data,unclassified_data
    
    def testingEnergy(self,threshold=-3.0,T=2):
        # switch to evaluate mode
        self.network.eval()
        with torch.no_grad():
            # get current model predict result
            test_outputs = self.network(self.X_test_tensor)

            # energy formula
            energy_scores = -(T * torch.logsumexp(test_outputs/T, dim=1)).numpy()

            predicted_idx = test_outputs.argmax(dim=1).numpy()
        
        # combine result  
        test_results = self.test_set.copy()
        test_results['Energy_Score'] = energy_scores

        inv_class_mapping = {v: k for k, v in self.class_mapping.items()}  
        test_results['Predict'] = [inv_class_mapping[idx] for idx in predicted_idx]

        classified_data_mask = test_results['Energy_Score'] < threshold

        classified_data = test_results[classified_data_mask].copy()
        classified_data = classified_data.drop(columns=['Energy_Score'])
        unclassified_data = test_results[~classified_data_mask].copy()
        unclassified_data = unclassified_data.drop(columns=['Energy_Score','Predict'])
        return test_results,classified_data,unclassified_data

    def saveModelPackage(self,file_name="Classification\\DL-MLP\\normal_model.pth"):
        # save overall model in file
        model_package = {
            'model_state_dict': self.network.state_dict(),
            'scaler': self.scaler,
            'class_mapping': self.class_mapping
        }
        torch.save(model_package,file_name)
        print("success")
        return model_package

    
if __name__=="__main__":
    normal_model_package().saveModelPackage()