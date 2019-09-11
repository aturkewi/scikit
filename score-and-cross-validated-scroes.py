from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
score = svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

# Print the score
print(score)


# Create three models with different subsets (folds) of the data
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print("manually creating folds and then scoring:")
print(scores)

# KFold automatically creates these folds for us and returns indicies
print("==============================")
print("Using KFold to make the folds:")
from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))

# Once we have the indicies it's trivial to run the test
scores = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])\
for train, test in k_fold.split(X_digits)]

print("manually running scoring for each fold:")
print(scores)

# OORRR we can just pass in a model, the data, and the folds to have this auto calculated
# n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
print("cross_val_score: ")
scores = cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
print(scores)

print("cross_val_score with precision_macro: ")
scores = cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')
print(scores)

# Exercise
# Solution: https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#sphx-glr-auto-examples-exercises-plot-cv-digits-py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)
k_fold = KFold(n_splits=3)

for c in C_s:
    svc.C = c
    scores = cross_val_score(svc, X, y, cv=3, n_jobs=-1)
    print(scores)

# Grid Search
