# author : Mayank Jindal

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("/home/mayank/Downloads/Bike-Sharing-Dataset/day.csv")

x = pd.DataFrame(data["temp"])
y = pd.DataFrame(data["cnt"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)

print(y_pred)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, linreg.predict(x_test), color='blue')
plt.title('Test Set')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()