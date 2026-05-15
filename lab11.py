import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv('CC_GENERAL.csv')
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

df.drop('CUST_ID', axis=1, inplace=True)
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)
print("\nMissing Values After Handling:")
print(df.isnull().sum())
df.hist(figsize=(18, 12), bins=20)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(df['BALANCE'], df['PURCHASES'])
plt.xlabel('BALANCE')
plt.ylabel('PURCHASES')
plt.title('BALANCE vs PURCHASES')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df['BALANCE'], df['CASH_ADVANCE'])
plt.xlabel('BALANCE')
plt.ylabel('CASH_ADVANCE')
plt.title('BALANCE vs CASH_ADVANCE')
plt.show()

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(8,6))
plt.plot(range(1,11), inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(scaled_data)

    score = silhouette_score(
        scaled_data,
        cluster_labels
    )

    silhouette_scores.append(score)

plt.figure(figsize=(8,6))
plt.plot(range(2,11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
plt.show()

silhouette_table = pd.DataFrame({
    'K': range(2,11),
    'Silhouette Score': silhouette_scores
})

print("\nSilhouette Scores Table:")
print(silhouette_table)
final_k = 4

final_kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

clusters = final_kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters

print("\nData with Cluster Labels:")
print(df.head())

cluster_summary = df.groupby('Cluster').mean()

print("\nCluster Summary:")
print(cluster_summary)

print("\nCustomers per Cluster:")
print(df['Cluster'].value_counts())

pca = PCA(n_components=2)

pca_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    data=pca_components,
    columns=['PCA1', 'PCA2']
)

pca_df['Cluster'] = clusters
plt.figure(figsize=(10,8))

sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=pca_df,
    palette='Set1'
)

plt.title('Customer Segments (PCA Visualization)')
plt.show()

print("\nFinal Questions Answers:\n")

print("1. This is an unsupervised learning problem because the dataset does not contain target labels.")
print("\n2. We removed the CUST_ID column because it is only an identifier and does not describe customer behavior.")
print("\n3. Some columns contained missing values such as MINIMUM_PAYMENTS and CREDIT_LIMIT.")
print("\n4. Missing values were handled using mean imputation.")
print("\n5. Feature scaling is important because K-Means uses distance calculations and features have different ranges.")
print("\n6. The elbow method helps estimate a good value for K by observing inertia.")
print("\n7. The silhouette score measures how well the clusters are separated.")
print("\n8. PCA was used to reduce the data into 2 dimensions for visualization.")
