from sklearn import datasets, neighbors, linear_model
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

ten_percent = round(len(X_digits)/10)
np.random.seed(0)
indices = np.random.permutation(len(X_digits))

# Get data sets
digits_X_train = X_digits[indices[:-ten_percent]]
digits_y_train = y_digits[indices[:-ten_percent]]
digits_X_test = X_digits[indices[-ten_percent:]]
digits_y_test = y_digits[indices[-ten_percent:]]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(digits_X_train, digits_y_train)

knn.predict(digits_X_test)

digits_y_test

log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial', max_iter=1000)
log.fit(digits_X_train, digits_y_train)
log.predict(digits_X_test)
