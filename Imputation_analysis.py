#This code uses various impuatation methods to calculate the grades of the students and then compares those methods.

import pandas as pd
from fancyimpute import KNN
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.read_excel("/home/mayank/Downloads/Data for Missing values.xlsx")

# Grade Calculation
def grade_calc(sum):
    if sum < 7.5:
        return 'D'
    elif sum >= 7.5 and sum < 15:
        return 'D+'
    elif sum >= 15 and sum < 22.5:
        return 'C'
    elif sum >= 22.5 and sum < 30:
        return 'C+'
    elif sum >= 30 and sum < 37.5:
        return 'B'
    elif sum >= 37.5 and sum < 45:
        return 'B+'
    elif sum >= 45 and sum < 52.5:
        return 'A'
    elif sum >= 52.5:
        return 'A+'

del data['Unnamed: 0']
del data['Unnamed: 1']
data.columns.values[0] = 'ID'
data.columns.values[1] = 'First_Name'
data.columns.values[2] = 'Last_Name'
data.columns.values[3] = 'IC1'
data.columns.values[4] = 'IC2'
data.columns.values[5] = 'IC3'
data.columns.values[6] = 'IC4'
data.columns.values[7] = 'Mid_Sem'
data['grade'] = ['Grade' for i in range (0, len(data))]
data['marks'] = ['0' for i in range (0, len(data))]

#Imputation using dropna

temp = data.copy()
temp.dropna(axis=0, how='any', inplace = True)
index = list(range(0, len(temp)))
temp.index = index
for i in range(0, len(temp)):
    sum = temp['IC1'].ix[i] + temp['IC2'].ix[i] + temp['IC3'].ix[i] + temp['IC4'].ix[i] + temp['Mid_Sem'].ix[i]
    temp['grade'].ix[i] = grade_calc(sum)
    temp['marks'].ix[i] = sum
print("Imputation using dropna")
print(temp['grade'].value_counts())
org_mean = temp['marks'].mean()
org_std = temp['marks'].std()
print("Average marks: ", org_mean, "    Standard deviation: ", org_std)

#Imputation using global mean
print("Imputation using mean")
temp = data.copy()
temp['IC1'].fillna(temp['IC1'].mean(), inplace = True)
temp['IC2'].fillna(temp['IC2'].mean(), inplace = True)
temp['IC3'].fillna(temp['IC3'].mean(), inplace = True)
temp['IC4'].fillna(temp['IC4'].mean(), inplace = True)
temp['Mid_Sem'].fillna(temp['Mid_Sem'].mean(), inplace = True)
for i in range(0, len(temp)):
    sum = temp['IC1'].ix[i] + temp['IC2'].ix[i] + temp['IC3'].ix[i] + temp['IC4'].ix[i] + temp['Mid_Sem'].ix[i]
    temp['grade'].ix[i] = grade_calc(sum)
    temp['marks'].ix[i] = sum
print(temp['grade'].value_counts())
gmean_mean = temp['marks'].mean()
gmean_std = temp['marks'].std()
print("Average marks: ", gmean_mean, "    Standard deviation: ", gmean_std)

#Imputation using mode
print("Imputation using mode")
temp = data.copy()
for i in range(0, len(temp)):
    temp['IC1'].fillna(temp['IC1'].mode()[0], inplace = True)
    temp['IC2'].fillna(temp['IC2'].mode()[0], inplace = True)
    temp['IC3'].fillna(temp['IC3'].mode()[0], inplace = True)
    temp['IC4'].fillna(temp['IC4'].mode()[0], inplace = True)
    temp['Mid_Sem'].fillna(temp['Mid_Sem'].mode()[0], inplace = True)
for i in range(0, len(temp)):
    sum = temp['IC1'].ix[i] + temp['IC2'].ix[i] + temp['IC3'].ix[i] + temp['IC4'].ix[i] + temp['Mid_Sem'].ix[i]
    temp['grade'].ix[i] = grade_calc(sum)
    temp['marks'].ix[i] = sum
print(temp['grade'].value_counts())
mode_mean = temp['marks'].mean()
mode_std = temp['marks'].std()
print("Average marks: ", mode_mean, "    Standard deviation: ", mode_std)

#Imputation using zero
print("Imputation using zero")
temp = data.copy()
temp['IC1'].fillna(0, inplace = True)
temp['IC2'].fillna(0, inplace = True)
temp['IC3'].fillna(0, inplace = True)
temp['IC4'].fillna(0, inplace = True)
temp['Mid_Sem'].fillna(0, inplace = True)
for i in range(0, len(temp)):
    sum = temp['IC1'].ix[i] + temp['IC2'].ix[i] + temp['IC3'].ix[i] + temp['IC4'].ix[i] + temp['Mid_Sem'].ix[i]
    temp['grade'].ix[i] = grade_calc(sum)
    temp['marks'].ix[i] = sum
print(temp['grade'].value_counts())
zero_mean = temp['marks'].mean()
zero_std = temp['marks'].std()
print("Average marks: ", zero_mean, "    Standard deviation: ", zero_std)

#Imputation using KNN (K nearest neighbours, sampled using mean square differences)
print("Imputation using KNN")
temp_data = data.copy()
temp = pd.DataFrame(columns = ['IC1', 'IC2', 'IC3', 'IC4', 'Mid_Sem'])
temp['IC1'] = data['IC1'].copy()
temp['IC2'] = data['IC2'].copy()
temp['IC3'] = data['IC3'].copy()
temp['IC4'] = data['IC4'].copy()
temp['Mid_Sem'] = data['Mid_Sem'].copy()
temp = KNN(k=3).complete(temp)

for i in range(0, len(temp_data)):
    temp_data['IC1'].ix[i] = temp[i][0]
    temp_data['IC2'].ix[i] = temp[i][1]
    temp_data['IC3'].ix[i] = temp[i][2]
    temp_data['IC4'].ix[i] = temp[i][3]
    temp_data['Mid_Sem'].ix[i] = temp[i][4]

for i in range(0, len(temp_data)):
    sum = temp_data['IC1'].ix[i] + temp_data['IC2'].ix[i] + temp_data['IC3'].ix[i] + temp_data['IC4'].ix[i] + temp_data['Mid_Sem'].ix[i]
    temp_data['grade'].ix[i] = grade_calc(sum)
    temp_data['marks'].ix[i] = sum
print(temp_data['grade'].value_counts())
knn_mean = temp_data['marks'].mean()
knn_std = temp_data['marks'].std()
print("Average marks: ", knn_mean, "    Standard deviation: ", knn_std)

#Imputation using rolling mean (Local Mean)
print("Imputation using rolling mean")
temp = data.copy()
temp['IC1'].fillna(pd.rolling_mean(temp['IC1'], 3, min_periods=2, center=True))
temp['IC2'].fillna(pd.rolling_mean(temp['IC2'], 3, min_periods=2, center=True))
temp['IC3'].fillna(pd.rolling_mean(temp['IC3'], 3, min_periods=2, center=True))
temp['IC4'].fillna(pd.rolling_mean(temp['IC4'], 3, min_periods=2, center=True))
temp['Mid_Sem'].fillna(pd.rolling_mean(temp['Mid_Sem'], 3, min_periods=2, center=True))
for i in range(0, len(temp)):
    sum = temp['IC1'].ix[i] + temp['IC2'].ix[i] + temp['IC3'].ix[i] + temp['IC4'].ix[i] + temp['Mid_Sem'].ix[i]
    temp['grade'].ix[i] = grade_calc(sum)
    temp['marks'].ix[i] = sum
print(temp['grade'].value_counts())
rolling_mean = temp['marks'].mean()
rolling_std = temp['marks'].std()
print("Average marks: ", rolling_mean, "    Standard deviation: ", rolling_std)
