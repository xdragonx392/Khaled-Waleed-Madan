import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

ad_data = pd.read_csv('advertising.csv')

print("First 5 rows:")
print(ad_data.head())

print("\nDataset info:")
print(ad_data.info())

print("\nStatistical summary:")
print(ad_data.describe())

sns.histplot(ad_data['Age'], bins=30)
plt.title("Age Distribution")
plt.show()
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde')
plt.show()
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)
plt.show()

sns.pairplot(ad_data, hue='Clicked on Ad')
plt.show()
X = ad_data.drop(['Clicked on Ad', 'Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
