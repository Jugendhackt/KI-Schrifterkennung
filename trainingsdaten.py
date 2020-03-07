from sklearn import datasets, neighbors
import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Display the first digit



# Fraegt Bild ab
def get_imag(nr):
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[nr], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# Fraegt Zahl ab
def get_target(nr):
    print(digits.target[nr])





n_neighbors = 15
X = digits.data[:, :2]
Y =digits.target

clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, X)



ts.get_imag(3)
ts.get_target(3)