from scipy.spatial import distance


def distance_between(x1, x2):
    return distance.euclidean(x1, x2)


class KNNImpl:
    def closest(self, test):
        """
        Finds the nearest element for the given test record in the train set
        And returns its label as the test record label/target

        how it working :
        consider first element from train data as nearest and iterate through all the train data
        if any of the train data is nearest to the assumed first update the nearest value and continue till
        we iterate through all the elements in train data
        return label of the nearest element as predicted label of this test record

        Example :
        train data
        [
          [ 1, 2]
          [3, 4]
          [5, 6]
        ]

        test data : [2, 4]
        then the nearest distance is [3, 4]
            -> sqrt(sqr(2-1)+sqr(4-4)) is smallest compared with other training data

        """
        nearest = distance_between(test, self.X[0])
        index = 0
        for i in range(1, len(self.y)):
            dist = distance_between(test, self.X[i])
            if dist < nearest:
                nearest = dist
                index = i
        return self.y[index]

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        predictions = []
        for test in X_test:
            label = self.closest(test)
            predictions.append(label)
        return predictions
