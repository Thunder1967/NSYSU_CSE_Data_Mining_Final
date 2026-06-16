import normal_model
import DOC_model
import normal_model_with_IsolationForest
import CRF_model
import evaluate
import myUtil

def case1():
    model = normal_model.normal_model_package(from_file="Classification\\DL-MLP\\normal_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingMCDropout()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case2():
    model = DOC_model.DOC_model_package(from_file="Classification\\DL-MLP\\DOC_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingDOC()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case3():
    model = normal_model_with_IsolationForest.mix_model_package(from_file="Classification\\DL-MLP\\mix_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingMCDropout()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case4():
    model = CRF_model.CRF_model_package(from_file="Classification\\DL-MLP\\CRF_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingMCDropout(threshold=0.3)
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

if __name__=="__main__":
    # normal_model_with_IsolationForest.mix_model_package().saveModelPackage()
    # myUtil.outputResultToFile(*case4())
    case4()