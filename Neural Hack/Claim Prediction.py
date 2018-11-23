import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Training Data
Data = pd.read_csv("Train_InPatient.csv")

cols = ['SEGMENT', 'NCH_PRMRY_PYR_CLM_PD_AMT', 'CLM_PASS_THRU_PER_DIEM_AMT', 'NCH_BENE_IP_DDCTBL_AMT', 'NCH_BENE_PTA_COINSRNC_LBLTY_AM', 'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM', 'CLM_UTLZTN_DAY_CNT', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'Total_mons']
X = pd.DataFrame(Data[:44998], columns=cols)
Y = Data[:44998]['CLM_PMT_AMT']

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

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=101)

lin_reg = LinearRegression()
lin_reg.fit(xtrain, ytrain)
pred = lin_reg.predict(xtest)
print(mean_squared_error(ytest, pred))
print(r2_score(ytest, pred))

# Saving the Linear Regression Model
with open("Model", 'wb') as f:
    pickle.dump(lin_reg, f)

