import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
uploaded=files.upload()
train=pd.read_csv("train.csv")
train.head()
test=pd.read_csv("test.csv")
test.head()
train.columns
print(train.shape, test.shape)
train['Loan_Status'].value_counts().plot.bar()

train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_employee')
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')
train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
train['ApplicantIncome'].value_counts(normalize=True).plot.bar(title='ApplicantIncome')

train['CoapplicantIncome'].plot.box()
train['LoanAmount'].plot.box()

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))

matrix=train.corr()
f,ax=plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu",annot=True)
train_encoded = pd.get_dummies(train)
# Generate the correlation matrix
matrix = train_encoded.corr()
train_encoded = pd.get_dummies(train)

# Generate the correlation matrix
matrix = train_encoded.corr()

# Plot the heatmap
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu", annot=True)
plt.show()
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
print(train['Gender'])
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
print(train['Married'])

train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
print(train['Dependents'])
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
print(train['Self_Employed'])
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
print(train['Self_Employed'])
train['LoanAmount'].fillna(train['LoanAmount'].mode()[0],inplace=True)
print(train['LoanAmount'])
train['LoanAmount'].fillna(train['LoanAmount'].mode()[0],inplace=True)
print(train['LoanAmount'])
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace=True)
print(train['LoanAmount'])
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace=True)
print(train['LoanAmount'])
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace=True)
print(train['LoanAmount'])
train['Credit_History'].fillna(train['Credit_History'].mean(),inplace=True)
print(train['Credit_History'])
train.isnull().sum()
X=train.drop('Loan_Status',1)

X=X.drop('Loan_ID',axis=1)
Y=train['Loan_Status']

X=pd.get_dummies(X)

print(X)
print(Y)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
clf=model.fit(X,Y)
df=pd.read_csv('test.csv')
print(df)

df.isnull().sum()
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
#df['Loan_Amount_Term'].fillna(df[Loan_Amount_Term].mean(),inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)
df.isnull().sum()
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
df.isnull().sum()
df=df.drop('Loan_ID',axis=1)
df=pd.get_dummies(df)
print(X)
print(df)

clf.predict(df)

