from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

# STEP 1 load the iris
iris_data = load_iris()

# STEP 2 test data and train data

# Test data 0,50,100
test_idx = [0, 50, 100]

# test data
test_features = iris_data.data[test_idx]
test_targets = iris_data.target[test_idx]

# training data
train_features = np.delete(iris_data.data, test_idx, axis=0)
train_targets = np.delete(iris_data.target, test_idx)

# STEP 3: create classifier
classifier = tree.DecisionTreeClassifier()
# STEP 4: train the classifier
classifier.fit(train_features, train_targets)

print test_targets
result = classifier.predict(test_features)
print result

import graphviz

dot_data = tree.export_graphviz(classifier, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
