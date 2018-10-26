from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics

iris = datasets.load_iris()
data = pd.DataFrame({'sepal length': iris.data[:, 0],
                     'sepal width': iris.data[:, 1],
                     'petal length': iris.data[:, 2],
                     'petal width': iris.data[:, 3],
                     'species': iris.target})

x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = rfc(n_estimators=100)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))