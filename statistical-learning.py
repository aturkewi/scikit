from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
data.shape

digits = datasets.load_digits()
digits.images.shape

import matplotlib.pyplot as plt
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
plt.show()

data = digits.images.reshape((digits.images.shape[0], -1))
