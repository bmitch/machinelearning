import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Indexes to remove for testing
test_idx = [0, 50, 100]

# Remove 3 indexes from training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Use those same 3 indexes to create test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)