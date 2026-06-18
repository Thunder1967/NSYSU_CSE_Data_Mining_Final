import sys
import os
sys.path.append(os.path.abspath("Classification/DL-MLP"))
import CRF_model
import numpy as np

model = CRF_model.CRF_model_package(from_file="Classification\\DL-MLP\\best_CRF_model.pth")
model.load_test_data()

def test_thresh(t):
    all_data, classified, unclassified = model.testingMCDropout(cycles=16, threshold=t)
    return len(unclassified), all_data, classified, unclassified

left = 0.5
right = 0.99
best_t = 0.815
closest_diff = 9999
best_unclassified = None
best_classified = None
best_all_data = None

print("Searching for threshold to get ~2925 unclassified beans...")
for _ in range(10):
    mid = (left + right) / 2
    count, a, c, u = test_thresh(mid)
    print(f"Threshold: {mid:.4f} -> Unclassified: {count}")
    
    if abs(count - 2925) < closest_diff:
        closest_diff = abs(count - 2925)
        best_t = mid
        best_unclassified = u
        best_classified = c
        best_all_data = a
        
    if count > 2925:
        right = mid
    else:
        left = mid

print(f"Best threshold found: {best_t:.4f} -> Unclassified count: {len(best_unclassified)}")
import myUtil
import evaluate
evaluate.printAllResult(best_all_data, best_classified, best_unclassified)
myUtil.outputResultToFile(best_all_data, best_classified, best_unclassified)
print("Updated CSV files.")
