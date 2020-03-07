import matplotlib.pyplot as plt
import numpy as np


def load_images():
    images = 1
    image_array = np.zeros((images,64))

    for i in range(images):
        im = plt.imread("img/%02d.jpg" % i)
        im = np.array(im).flatten()
        image_array[i] = im
        #plt.imshow(im, cmap='Greys_r')
        #plt.show()

    return image_array

if __name__ == '__main__':
    print(load_images())
