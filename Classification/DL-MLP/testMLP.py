import pandas as pd  
import torch
import trainMLP

def testing(network,X_test_tensor):
    # switch to evaluate mode
    network.eval()
    with torch.no_grad():
        # get current model predict result
        test_outputs = network(X_test_tensor)

        # transform original score to percentage with Softmax
        test_probabilities = torch.softmax(test_outputs, dim=1).numpy()
    return test_probabilities

def testingTemperature(network,X_test_tensor,T=1):
    # switch to evaluate mode
    network.eval()
    with torch.no_grad():
        # get current model predict result
        test_outputs = network(X_test_tensor)

        # transform original score to percentage with Softmax
        test_probabilities = torch.softmax(test_outputs/T, dim=1).numpy()
    return test_probabilities

def testingMCDropout(network,X_test_tensor,times=10):
    # switch to evaluate mode
    network.eval()
    # open Dropout
    for module in network.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    all_result = []
    for _ in range(times):
        with torch.no_grad():
            # get current model predict result
            test_outputs = network(X_test_tensor)

            # transform original score to percentage with Softmax
            all_result.append(test_outputs)
    return torch.softmax(torch.stack(all_result, dim=0).mean(dim=0), dim=1).numpy()



def readData(file_name="Classification\\DL-MLP\\MLP_classifier_model.pth"):
    # read model package
    model_package = torch.load(file_name,weights_only=False)
    network = trainMLP.BeanClassifier()
    network.load_state_dict(model_package["model_state_dict"])
    scaler = model_package["scaler"]
    class_mapping = model_package['class_mapping']
    # test set preprocess
    test_set = pd.read_csv("dry_bean_test.csv")  
    X_test_raw = test_set.drop(columns=['Class'])
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    return network,class_mapping,X_test_tensor,test_set

def combineResult(test_results,class_mapping,test_probabilities):
    # get max percentage
    max_probability = test_probabilities.max(axis=1)  
    predicted_idx = test_probabilities.argmax(axis=1)  
    test_results['Trust_index'] = max_probability

    # translate index to string label
    inv_class_mapping = {v: k for k, v in class_mapping.items()}  
    test_results['Predict'] = [inv_class_mapping[idx] for idx in predicted_idx]
    return test_results

def classifiableOrNot(test_results,trust_threshold=0.8):
    # classifiable data
    classifiable_data_mask = test_results['Trust_index'] >= trust_threshold  
    classifiable_data = test_results[classifiable_data_mask]

    # unclassifiable data
    unclassifiable_data = test_results[~classifiable_data_mask]

    return classifiable_data,unclassifiable_data

def outputResultToFile(classifiable_data,unclassifiable_data):
    classifiable_data.to_csv("classified_data.csv", index=False)
    unclassifiable_data.to_csv("unclassified_data.csv", index=False)
    print("success")

if __name__=="__main__":
    network,class_mapping,X_test_tensor,test_set = readData()
    test_probabilities = testingTemperature(network,X_test_tensor,T=1)
    test_results = combineResult(test_set,class_mapping,test_probabilities)
    classifiable_data,unclassifiable_data = classifiableOrNot(test_results,trust_threshold=0.8)
    outputResultToFile(classifiable_data,unclassifiable_data)