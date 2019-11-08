from sklearn.metrics import roc_curve
import numpy as np

y = np.array([1,1,2,2,])
scores = np.array([0.1, 0.4, 0.35, 0.8])

fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
print(fpr, tpr, thresholds)