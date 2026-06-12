import testMLP
import evaluate

if __name__=="__main__":
    network,class_mapping,X_test_tensor,test_set = testMLP.readData()
    test_probabilities = testMLP.testingMCDropout(network,X_test_tensor,times=10)
    test_results = testMLP.combineResult(test_set,class_mapping,test_probabilities)
    classified_data,unclassified_data = testMLP.classifiableOrNot(test_results,trust_threshold=0.8)
    
    evaluate.printAllResult(test_results,classified_data,unclassified_data)