import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

df = pd.read_csv('Ecommerce Customers')

print("FIRST 5 ROWS:")
print(df.head())

print("\nDATA INFO:")
print(df.info())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nCOLUMNS:")
print(df.columns)

print("\nMISSING VALUES:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
sns.pairplot(df)

plt.figure(figsize=(8,5))
sns.histplot(df['Yearly Amount Spent'], bins=30)
plt.title('Distribution of Yearly Amount Spent')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
X = df[['Avg. Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership']]

y = df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

print("\nINTERCEPT:")
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

print("\nCOEFFICIENTS:")
print(coeff_df)
predictions = lm.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot((y_test - predictions), bins=30)
plt.title("Residuals Distribution")
plt.show()

from sklearn import metrics

print("\nMODEL EVALUATION:")

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2 Score:', metrics.r2_score(y_test, predictions))
