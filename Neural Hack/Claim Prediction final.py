import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Test Data
Data = pd.read_csv("Test_InPatient.csv")

cols = ['SEGMENT', 'NCH_PRMRY_PYR_CLM_PD_AMT', 'CLM_PASS_THRU_PER_DIEM_AMT', 'NCH_BENE_IP_DDCTBL_AMT', 'NCH_BENE_PTA_COINSRNC_LBLTY_AM', 'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM', 'CLM_UTLZTN_DAY_CNT', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'Total_mons']
X = pd.DataFrame(Data[:20537], columns=cols)

# Filling Missing Data
for c in cols:
    if X[c].isnull().values.any():
        X[c].fillna(int(X[c].mean()), inplace=True)

# Handling Categorical Variables
categorical_var = ['BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']
for v in categorical_var:
    dummies = pd.get_dummies(X[v], prefix=v)
    X = pd.concat([X, dummies], axis=1)
    X.drop([v], axis=1, inplace=True)

# Importing Linear Regression Model
with open("Model", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X)

final_data = pd.DataFrame(Data, columns=['DESYNPUF_ID'])
final_data['Predicted PMT AMT'] = pred
final_data.rename({'DESYNPUF_ID':'Patient ID'}, inplace=True)
final_data.to_csv("Output.csv", index=False)

# Visualization
cols = ['CLM_FROM_DT', 'SEGMENT', 'NCH_PRMRY_PYR_CLM_PD_AMT', 'CLM_PASS_THRU_PER_DIEM_AMT', 'NCH_BENE_IP_DDCTBL_AMT', 'NCH_BENE_PTA_COINSRNC_LBLTY_AM', 'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM', 'CLM_UTLZTN_DAY_CNT', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'Total_mons']
X = pd.DataFrame(Data[:20537], columns=cols)
X['Result'] = pred
gender = list(X['Result'].groupby(X['BENE_SEX_IDENT_CD']))
race = list(X['Result'].groupby(X['BENE_RACE_CD']))

sex_label = ["Male", "Female"]
sex_count = [len(list(gender[0][1])), len(list(gender[1][1]))]

race_label = [str(i) for i in set(X['BENE_RACE_CD'])]
race_count = [len(race[i][1]) for i in range(len(race_label))]

plt.pie(sex_count, labels=sex_label, autopct='%1.1f%%')
plt.title('Genders')
plt.show()

plt.pie(race_count, labels=race_label, autopct='%1.1f%%')
plt.title('Race')
plt.show()


# Time Series Graphs
for i in range(0, len(X)):
    X.iloc[i, 0] = str(X.iloc[i, 0])[:4]

X.sort_values(by=['CLM_FROM_DT'], inplace=True)
X.dropna(axis=0, how='any', inplace=True)

# Graph with Claim Amount

plt.scatter(X.iloc[:, 0], X.iloc[:, -1], color='blue')
plt.title('Time Series')
plt.xlabel('Years')
plt.ylabel('Claim Amount')
plt.show()

# Graph with Claim Utilization Day Count
plt.scatter(X.iloc[:, 0], X.iloc[:, 7], color='blue')
plt.title('Time Series')
plt.xlabel('Years')
plt.ylabel('Claim Utilization Day Count')
plt.show()
