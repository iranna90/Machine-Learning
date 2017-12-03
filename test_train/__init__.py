from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

# F(X)=y -> X is input features and y is label
X = iris.data
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.5)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)

print "Excepted y {}".format(Y_test)
print "Predictions {}".format(predictions)
print "accuracy {}".format(accuracy)
