import sklearn.datasets
import sklearn.neighbors
import sklearn.svm
from sklearn.model_selection import train_test_split
import numpy as np

digits = sklearn.datasets.load_digits()

clf = sklearn.svm.SVC(gamma=0.001)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)


clf.fit(X_train, y_train)
print(clf.predict(X_test))

