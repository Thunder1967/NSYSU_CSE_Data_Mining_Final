from sklearn.metrics import classification_report
import pandas as pd

def getClassificationReport(data):
    return classification_report(data['Class'], data['Predict'])

def unclassifiedRate(all_data,unclassified_data):
    return len(unclassified_data)/len(all_data)

def NDIUD(unclassified_data):
    # new data in unclassified data
    missing_in_train = ['BARBUNYA', 'BOMBAY']
    new_class_count = unclassified_data['Class'].isin(missing_in_train).sum()
    total_unclassified = len(unclassified_data)
    return new_class_count/total_unclassified

def printAllResult(all_data,classified_data,unclassified_data):
    print("=== Classified Data Report ===")
    print(getClassificationReport(classified_data))
    print("=== All Data Report ===")
    print(getClassificationReport(all_data))
    print("=== unclassified data rate ===")
    print(unclassifiedRate(all_data,unclassified_data))
    print("=== new data in unclassified data ===")
    print(NDIUD(unclassified_data))
    print("=== new data in unclassified data * unclassified data rate ===")
    print(NDIUD(unclassified_data)*unclassifiedRate(all_data,unclassified_data))

if __name__=="__main__":
    classified_data = pd.read_csv("classified_data.csv")
    unclassified_data = pd.read_csv("unclassified_data.csv")
    all_data = pd.concat([classified_data, unclassified_data], axis=0, ignore_index=True)
    
    printAllResult(all_data,classified_data,unclassified_data)