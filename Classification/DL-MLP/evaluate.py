from sklearn.metrics import classification_report
import pandas as pd

def getClassificationReport(data):
    return classification_report(data['Class'], data['Predict'])

def unclassifiedRate(all_data,unclassified_data):
    return len(unclassified_data)/len(all_data)

def newClassPrecision(unclassified_data):
    # new data in unclassified data
    missing_in_train = ['BARBUNYA', 'BOMBAY']
    new_class_count = unclassified_data['Class'].isin(missing_in_train).sum()
    total_unclassified = len(unclassified_data)
    return new_class_count/total_unclassified

def newClassRecall(all_data,unclassified_data):
    missing_in_train = ['BARBUNYA', 'BOMBAY']
    new_class_in_unclassified = unclassified_data['Class'].isin(missing_in_train).sum()
    total_new_class = all_data['Class'].isin(missing_in_train).sum()
    return new_class_in_unclassified/total_new_class

def printAllResult(all_data,classified_data,unclassified_data):
    print("=== All Data Report ===")
    print(getClassificationReport(all_data))
    print("=== Classified Data Report ===")
    print(getClassificationReport(classified_data))
    print("=== unclassified data rate ===")
    print(unclassifiedRate(all_data,unclassified_data))
    print("=== # of unclassified data ===")
    print(len(unclassified_data))
    print("=== new Class Precision ===")
    print(a:=newClassPrecision(unclassified_data))
    print("=== new Class Recall  ===")
    print(b:=newClassRecall(all_data,unclassified_data))
    print("=== new class F1-score  ===")
    print(2*(a*b)/(a+b))