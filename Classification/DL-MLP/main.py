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
    all_data,classified_data,unclassified_data = model.testingEnergy()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case4():
    model = CRF_model.CRF_model_package(from_file="Classification\\DL-MLP\\CRF_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingEnergy()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case5():
    model = CRF_model.CRF_model_package(from_file="Classification\\DL-MLP\\CRF_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingMCDropout(cycles=23, threshold=0.736)
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case6():
    # 使用 Optuna 找出的最佳訓練參數，直接重新訓練一個終極模型
    model = CRF_model.CRF_model_package(
        from_file="",
        lr=0.0026822027285217983,
        cycles=63,
        label_smoothing=0.07568634172575536,
        reconstruct_weight=0.44976661562055753
    )
    # 儲存這個終極模型
    model.saveModelPackage("Classification\\DL-MLP\\best_CRF_model.pth")
    model.load_test_data()
    
    # 使用最佳推論參數
    all_data,classified_data,unclassified_data = model.testingMCDropout(cycles=28, threshold=0.6551976059164604)
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

if __name__=="__main__":
    # normal_model_with_IsolationForest.mix_model_package().saveModelPackage()
    myUtil.outputResultToFile(*case4())