import normal_model
import evaluate

if __name__=="__main__":
    model = normal_model.normal_model_package(from_file="Classification\\DL-MLP\\normal_model.pth")
    model.load_test_data()
    all_data,classified_data,unclassified_data = model.testingMCDropout()
    evaluate.printAllResult(all_data,classified_data,unclassified_data)