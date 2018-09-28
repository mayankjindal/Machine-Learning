# author : Mayank Jindal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("/home/mayank/Downloads/Bike-Sharing-Dataset/day.csv")

y = data.pop("cnt")
x = pd.DataFrame(data, columns=["temp", "hum"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)

print(y_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test['temp'], x_test['hum'], y_test, c='blue', marker='o', alpha=0.5)
ax.plot(x_test["temp"], x_test['hum'], linreg.predict(x_test))
plt.show()

