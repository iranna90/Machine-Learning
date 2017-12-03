from sklearn import tree

apple = "Apple"
orange = "Orange"
fruits = {0: apple, 1: orange}

smooth = "smooth"
bumpy = "bumpy"
look = {smooth: 0, bumpy: 1}
# 0 -> smooth and 1 -> bumpy
features = [[140, look.get(smooth)], [130, look.get(smooth)], [150, look.get(bumpy)], [170, look.get(bumpy)]]
# 0 -> apple and 1 -> orange
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)

test = [[180, 0]]
result = classifier.decision_path(test)
print(result)
result = classifier.predict(test)
print(result)
print("resulting fruit is {}".format(fruits.get(result[0])))
