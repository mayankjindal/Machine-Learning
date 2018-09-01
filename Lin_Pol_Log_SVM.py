import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

# Predicting grades through Linear Regression as well as Polynomial Regression
# Predicting school through Logistic Regression as well as SVM

file_location = "/home/mayank/Downloads/"


# Formatting the data to make it fit for the model. Data is saved in another csv file in the required format.
def data_preprocessing(categorical_variables, binary_variables, name):
    org = pd.read_csv(file_location+"Data.csv", sep=';')
    x = pd.DataFrame(org, index=list(range(0, len(org))))

    for v in categorical_variables:
        dummies_train = pd.get_dummies(x[v], prefix=v)
        x = pd.concat([x, dummies_train], axis=1)
        x.drop([v], axis=1, inplace=True)

    for v in binary_variables:
        for i in range(0, len(x)):
            if x.ix[i, v] == 'yes':
                x.ix[i, v] = 1
            else:
                x.ix[i, v] = 0

    x.to_csv(file_location+name, index=False)


# Retrieving the formatted data
def retrieve_data(v, name, train=False):
    data = pd.read_csv(file_location+name)
    if train:
        res = data.pop(v)
        return data, res
    else:
        data.drop([v], axis=1, inplace=True)
        return data


# Applying Linear Regression
def linear_regression(xtrain, ytrain):
    linreg = LinearRegression()
    linreg.fit(xtrain, ytrain)
    return linreg


# Applying Polynomial Regression
def polynomial_regression(xtrain, ytrain, xtest, d):
    poly = PolynomialFeatures(degree=d)
    poly_train_data = poly.fit_transform(xtrain)
    poly_test_data_i = poly.fit_transform(xtest)
    poly_reg_i = LinearRegression()
    poly_reg_i.fit(poly_train_data, ytrain)

    return poly_reg_i, poly_test_data_i


# Applying Logistic Regression
def logistic_regression(xtrain, ytrain):
    logreg = LogisticRegression()
    logreg.fit(xtrain, ytrain)
    return logreg


def support_vector_model(xtrain, ytrain):
    svm_i = SVC()
    svm_i.fit(xtrain, ytrain)
    return svm_i


categorical_variables = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]
binary_variables = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

data_preprocessing(categorical_variables, binary_variables, "Data1.csv")
train_data, sample_results = retrieve_data('G3', 'Data1.csv', True)
test_data = retrieve_data('G3', 'Data1.csv')

print("Predicting grades through Linear Regression")
lin_reg = linear_regression(train_data, sample_results)  # Calling Custom Function
lin_predict = lin_reg.predict(test_data)
print(lin_predict)
lin_rms = mean_squared_error(sample_results, lin_predict)
print("Mean Squared Error with Linear Regression = ", lin_rms)

print("Predicting grades through Polynomial Regression, Degree = 2")
poly_reg, poly_test_data = polynomial_regression(train_data, sample_results, test_data, 2)
poly_predict_raw = poly_reg.predict(poly_test_data)
poly_predict = []
for i in range(0, len(poly_predict_raw)):
    poly_predict.append(int(poly_predict_raw[i]))

poly_rms = mean_squared_error(sample_results, poly_predict)
print("Mean Squared Error with Polynomial Regression with degree 2= ", poly_rms)

print("Predicting grades through Polynomial Regression, Degree = 3")
poly_reg, poly_test_data = polynomial_regression(train_data, sample_results, test_data, 3)
poly_predict_raw = poly_reg.predict(poly_test_data)
poly_predict = []
for i in range(0, len(poly_predict_raw)):
    poly_predict.append(int(poly_predict_raw[i]))

poly_rms = mean_squared_error(sample_results, poly_predict)
print("Mean Squared Error with Polynomial Regression with degree 3= ", poly_rms)

categorical_variables = ["sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]

data_preprocessing(categorical_variables, binary_variables, "Data2.csv")
train_data, sample_results = retrieve_data('school', 'Data2.csv', True)
test_data = retrieve_data('school', 'Data2.csv')

print("Predicting school through Logistic Regression")
log_reg = logistic_regression(train_data, sample_results)
log_predict = (log_reg.predict(test_data))
tn, fp, fn, tp = confusion_matrix(sample_results, log_predict).ravel()
print(tn, fp, fn, tp)
log_true = tn + tp
log_false = fp + fn
accuracy = log_true/(log_false+log_true)*100
print("True predictions by Logistic Regression = ", log_true, "False predictions by Logistic Regression = ", log_false, "accuracy = ", accuracy)

print("Predicting school through Support Vector Machine")
svm = support_vector_model(train_data, sample_results)
svm_predict = svm.predict(test_data)
tn, fp, fn, tp = confusion_matrix(sample_results, svm_predict).ravel()
print(tn, fp, fn, tp)
svm_true = tn + tp
svm_false = fn + fp
accuracy = svm_true/(svm_false+svm_true)*100
print("True predictions by SVM = ", svm_true, "False predictions by SVM = ", svm_false, "accuracy = ", accuracy)
