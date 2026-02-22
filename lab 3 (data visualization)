import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline

# Load dataset
tips = sns.load_dataset('tips')

# Distribution plot
sns.histplot(tips['total_bill'], bins=30)
plt.show()

# KDE plot
sns.kdeplot(tips['total_bill'])
plt.show()

# Joint plot
sns.jointplot(x='total_bill', y='tip', data=tips, kind='scatter')

# Pairplot
sns.pairplot(tips)

# Barplot
sns.barplot(x='sex', y='total_bill', data=tips)
plt.show()

# Countplot
sns.countplot(x='sex', data=tips)
plt.show()

# Boxplot
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()

# Violinplot
sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()

# Heatmap (Correlation)
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.show()

# Regression plot
sns.lmplot(x='total_bill', y='tip', data=tips)
