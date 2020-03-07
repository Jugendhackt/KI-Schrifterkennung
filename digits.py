import sklearn.datasets
import sklearn.neighbors
import sklearn.svm
import sklearn.neural_network

digits = sklearn.datasets.load_digits()

clf_svc = sklearn.svm.SVC(gamma=0.001)
clf_mlp = sklearn.neural_network.MLPClassifier()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X = data[:500]
Y = digits.target[:500]

X_test = data[1000:1036]
Y_test = digits.target[1000:1036]

clf_svc.fit(X, Y)
clf_mlp.fit(X, Y)
print(clf_svc.predict(X_test))
print(clf_mlp.predict(X_test))
print(Y_test)
