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
    all_data,classified_data,unclassified_data = model.testingMCDropout(cycles=13, threshold=0.7598985455458621)
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

def case6():
    # 使用 Optuna 找出的最佳訓練參數，直接重新訓練一個終極模型
    model = CRF_model.CRF_model_package(
        from_file="",
        lr=0.0032544716701542937,
        cycles=52,
        label_smoothing=0.16369619152356668,
        reconstruct_weight=0.25849666161387097
    )
    # 儲存這個終極模型
    model.saveModelPackage("Classification\\DL-MLP\\best_CRF_model.pth")
    model.load_test_data()
    
    # 使用最佳推論參數
    all_data,classified_data,unclassified_data = model.testingMCDropout(cycles=24, threshold=0.6218427239525507)
    evaluate.printAllResult(all_data,classified_data,unclassified_data)
    return all_data,classified_data,unclassified_data

if __name__=="__main__":
    # normal_model_with_IsolationForest.mix_model_package().saveModelPackage()
    myUtil.outputResultToFile(*case1())
    myUtil.outputResultToFile(*case2())
    myUtil.outputResultToFile(*case3())
    myUtil.outputResultToFile(*case4())
    myUtil.outputResultToFile(*case5())
    myUtil.outputResultToFile(*case6())