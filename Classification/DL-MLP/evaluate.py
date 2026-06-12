from sklearn.metrics import classification_report
import pandas as pd

# classified_data
classified_data = pd.read_csv("classified_data.csv")
Y_true_classified = classified_data['Class']
Y_pred_classified = classified_data['Predict']
total_classified_data = len(classified_data)

print("=== Classified Data Report ===")
print(classification_report(Y_true_classified, Y_pred_classified))


# unclassified_data
train_set = pd.read_csv("dry_bean_train.csv")
test_set = pd.read_csv("dry_bean_test.csv")
train_classes = set(train_set['Class'].unique())
test_classes = set(test_set['Class'].unique())
missing_in_train = test_classes - train_classes

unclassified_data = pd.read_csv("unclassified_data.csv")
new_class_count = unclassified_data['Class'].isin(missing_in_train).sum()
total_unclassified = len(unclassified_data)

print("=== unclassified data rate ===")
print(total_unclassified / (total_classified_data+total_unclassified) * 100)
print("=== new data in unclassified data ===")
print(missing_in_train)
print(new_class_count / total_unclassified * 100)