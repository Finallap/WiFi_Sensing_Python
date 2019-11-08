from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

iris = datasets.load_iris()
print(iris.data.shape)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
print(clf)
print(clf.score(x_test,y_test))

# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf,iris.data, iris.target,cv=10)
# print(scores)

y_predict = clf.predict(x_test)

print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print(metrics.precision_recall_fscore_support(y_test,y_predict))