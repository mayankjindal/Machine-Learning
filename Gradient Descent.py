import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def gradient_descent(x, y, alpha=0.01, ep=0.00001, max_it=1000):
    theta = np.zeros([2, 1])
    i = 0
    while i <= max_it:
        temp = theta.copy()
        j = sum((np.dot(x, temp) - y)**2)/(2*x.size)
        theta[0][0] = theta[0][0] - (alpha/len(x)*sum(np.dot(x, temp) - y))
        theta[1][0] = theta[1][0] - (alpha/len(x)*sum((np.dot(x, temp) - y)*(x[:, 1].reshape(len(x), 1))))
        e = sum((np.dot(x, theta) - y)**2)/(2*x.size)
        if abs(j - e) < ep:
            print("Stopping loop at: ", i)
            return theta
        i += 1
    print("Iteration Over")
    return theta


def predict(x, theta):
    return np.dot(x, theta)


def normalize_data(data):
    data = np.array(data)
    mu = data.mean()
    std = data.std()
    return (data - mu)/std


df = pd.read_csv("/home/mayank/Downloads/kc_house_data.csv")

x_data = normalize_data(df['sqft_living'][:500])
y_data = normalize_data(df['price'][:500])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)

# Vectorizing train data for Gradient Descent
X = np.append(np.ones(len(x_train)).reshape(len(x_train), 1), x_train.reshape(len(x_train), 1), axis=1)
Y = y_train.reshape(len(y_train), 1)

Theta = gradient_descent(X, Y)
print("Theta : ", Theta)

# Vectorizing test data for Gradient Descent
X = np.append(np.ones(len(x_test)).reshape(len(x_test), 1), np.array(x_test).reshape(len(x_test), 1), axis=1)
y_predicted = predict(X, Theta)
# print(y_predicted)

mse = mean_squared_error(y_test, y_predicted)
print("Mean Square Error: ", mse)