from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import random

iris = load_iris()
bt_sample = 50
avg = 0

for i in range(bt_sample):
    xtrain = []
    xtest = []
    ytrain = []
    ytest = []
    shift = [0 for k in range(150)]  # First 50 rows of iris data set
    for j in range(150):
        xtest.append(iris.data[j])
        ytest.append(iris.target[j])
    for j in range(150):
        r = random.randint(0, 150 - 1)
        xtrain.append(iris.data[r])
        ytrain.append(iris.target[r])
        if shift[r] != -1:
            xtest.pop(r - shift[r])
            ytest.pop(r - shift[r])
            shift[r] = -1
            for k in range(r + 1, 150):
                if shift[k] != -1:
                    shift[k] += 1
    clf = DecisionTreeClassifier()
    clf.fit(xtrain, ytrain)
    avg += clf.score(xtest, ytest)

print("Decision tree Score: ", avg / bt_sample)
