sns.set()

df = pd.read_csv("Chocolate_Sales.csv")
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
df['Amount'] = pd.to_numeric(df['Amount'])

df.dtypes

print(df.isna().sum())
print("Duplicates:", df.duplicated().sum())
print("Shape:", df.shape)
df.describe(include='all')


plt.figure(figsize=(8,5))
sns.histplot(df['Boxes Shipped'], bins=20)
plt.title("Distribution of Boxes Shipped")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], bins=20)
plt.title("Distribution of Revenue")
plt.show()
