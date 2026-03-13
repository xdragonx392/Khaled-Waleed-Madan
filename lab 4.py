import pandas as pd

df = pd.read_csv("Chocolate_Sales.csv")

df['Date'] = pd.to_datetime(df['Date'])

df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)

df['Amount'].fillna(df['Amount'].median(), inplace=True)

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Amount'] >= lower) & (df['Amount'] <= upper)]

print(df.head())
