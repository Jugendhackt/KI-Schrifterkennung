from sklearn import datasets, neighbors
import matplotlib.pyplot as plt


# Fraegt Bild ab
def get_imag(nr):
    plt.figure(1, figsize=(3, 3), frameon=False)
    #plt.imshow(digits.images[nr], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.savefig('img2.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

# Load the digits dataset
digits = datasets.load_digits()
get_imag(1001)
# Display the first digit



# Fraegt Zahl ab
def get_target(nr):
    print(digits.target[nr])





#n_neighbors = 15
#X = digits.data[:, :2]
# Y =digits.target
#
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     clf.fit(X, X)
#
#
#
# ts.get_imag(3)
# ts.get_target(3)
