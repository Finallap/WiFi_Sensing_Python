from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as  plt

np.random.seed(0)
iris = load_iris()
x, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
x, y = x[indices], y[indices]

train_scores, validation_scores = validation_curve(Ridge(), x, y, 'alpha', np.logspace(-7, 3, 3),cv=5)
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), x, y, train_sizes=[50, 80, 110], cv=5)
print(train_scores)
print(validation_scores)

plt.plot(train_scores)
plt.plot(validation_scores)
plt.show()