import matplotlib.pyplot as plt
import numpy as np

import sklearn.datasets
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm


def load_images():
    images = 11
    image_array = np.zeros((images, 64))

    for i in range(images):
        im = plt.imread("img/%02d.jpg" % i)
        im = np.array(im).flatten()
        image_array[i] = im
        #plt.imshow(im, cmap='Greys_r')
        # plt.show()

    return image_array


def train_classifier():

    digits = sklearn.datasets.load_digits()

    clf_svc = sklearn.svm.SVC(gamma=0.001)
    clf_svc = sklearn.neural_network.MLPClassifier()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    fig, ax = plt.subplots(1,10)
    for i in range(10):
        ax[i].imshow(data[i].reshape((8,8)), cmap='Greys')
    plt.show()
    #print(data[0])

    X = data
    Y = digits.target

    clf_svc.fit(X, Y)

    return clf_svc

if __name__ == '__main__':
    our_images = load_images()
    our_images = (255 - our_images) / 255. * 15.
    our_images = np.round(our_images)
    #plt.imshow(our_images[0].reshape((8,8)), cmap='Greys')
    #plt.show()

    #print(our_images[0])
    classifier = train_classifier()
    print(classifier.predict(our_images))
    print(np.array([0, 1, 4, 5, 8, 9, 2, 3, 1, 7, 3]))
