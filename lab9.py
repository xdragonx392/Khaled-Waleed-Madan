
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('your_dataset.csv')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

df = pd.get_dummies(df, drop_first=True)


X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

pred_dt = dtree.predict(X_test)

print("\n==============================")
print("Decision Tree Results")
print("==============================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, pred_dt))

print("Accuracy:", accuracy_score(y_test, pred_dt))
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

pred_rf = rfc.predict(X_test)

print("\n==============================")
print("Random Forest Results")
print("==============================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

print("Accuracy:", accuracy_score(y_test, pred_rf))
print("\n==============================")
print("Model Comparison")
print("==============================")

dt_acc = accuracy_score(y_test, pred_dt)
rf_acc = accuracy_score(y_test, pred_rf)

print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

if rf_acc > dt_acc:
    print("Random Forest performed better.")
elif dt_acc > rf_acc:
    print("Decision Tree performed better.")
else:
    print("Both models performed equally.")

importances = pd.Series(rfc.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', figsize=(8,6))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()
