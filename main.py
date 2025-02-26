import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv
data = pd.read_csv("./03_Clustering_Marketing.csv")

def preprocess():
    # Preprocessing
    # print(data.isnull().sum()) # Lots of nan for gender and age
    summary = data.describe()
    
    # Populate the age with average age
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    mean_age = data['age'].mean()
    data['age'].fillna(mean_age, inplace=True)

    # We drop all rows with a non-disclosed gender
    data.dropna(inplace=True)

    # We binarize gender, assigning 1 to female and 2 to male
    data['gender'] = data['gender'].apply(lambda gender: 1 if gender == 'F' else 2)

    # Perform a PCA dimensional reduction on the dataset for all keywords
    pca = PCA(n_components=7)
    features = data.iloc[:,4:].values
    reducedData = pca.fit_transform(features)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reducedData)
    scaled_data = pd.DataFrame(scaled_data)

    # Concatenate the PCA transformed data with the original data
    reducedData = pd.DataFrame(reducedData, columns=["PC" + str(i) for i in range(7)])
    reducedData = pd.concat([data.iloc[:,:4].reset_index(drop=True), reducedData.reset_index(drop=True)], axis=1)
    return reducedData
    
def k_means_test(reducedData):
    # We use the elbow method to find the best amount of clusters
    elbow_k_means(reducedData)
    
    # We use the optimized cluster value for our k-means model
    optimized_cluster_amount = 5
    kmeans = KMeans(n_clusters=optimized_cluster_amount, random_state=69)
    kmeans.fit(reducedData)
    
    plot_results(kmeans, "kmeans")
    return kmeans

def bisecting_k_means(reducedData):
    # Create a bisecting k_means model
    bisecting_kmeans = BisectingKMeans(n_clusters=5, random_state=69)
    bisecting_kmeans.fit(reducedData)
    
    plot_results(bisecting_kmeans, "bisecting kmeans")
    return bisecting_kmeans

def hierarchical_test(reducedData):
    hierarchical = AgglomerativeClustering(n_clusters=5).fit(reducedData)
    plot_dendrogram(reducedData)
    return hierarchical

def analysis(model, name):
    # Assign the data cluster labels according to the current model analyzed
    data['cluster'] = model.labels_
    
    # Create a column of average principal components for graphing
    data['avg_principal_component'] = reducedData[['PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']].mean(axis=1)
    cluster_counts = data['cluster'].value_counts()
    
    # Plot a graph of genders and cluster
    sns.scatterplot(data=data, x='NumberOffriends', y='avg_principal_component', hue='cluster', palette='deep')
    plt.title(f"Friends vs. Average Principal Component for {name}")
    plt.show()
    
    # Plot gender against average principal component
    sns.scatterplot(data=data, x='gender', y='avg_principal_component', hue='cluster', palette='deep')
    plt.title(f"Gender vs. Average Principal Component for {name}")
    plt.show()
    
    # Plot age against average principal component
    sns.scatterplot(data=data, x='age', y='avg_principal_component', hue='cluster', palette='deep')
    plt.title(f"Age vs. Average Principal Component for {name}")
    plt.show()
    
    # Amount of people in each cluster
    people = list(data.groupby(['cluster']).count()['age'].values)
    print(people)
    
    # Male gender distribution among the clusters
    percentage_male = data[data['gender'] == 2].groupby('cluster').size() / data.groupby('cluster').size() * 100
    
    print(f"Analysis for {name}")
    
    # Average age among each cluster
    average_age = data.groupby("cluster")['age'].mean()
    for cluster, age in average_age.items():
        print(f"Cluster {cluster}: Average {age} years")
    
    # Print gender proportions for clusters
    for cluster, proportion in percentage_male.items():
        print(f"Cluster {cluster}: {proportion}% male")
    
    # Average number of friends among each cluster
    average_friends = data.groupby("cluster")['NumberOffriends'].mean()
    for cluster, friends in average_friends.items():
        print(f"Cluster {cluster}: Average {friends:.0f} friends")
    

def plot_results(results, modelType):
    data['cluster'] = results.labels_
    cluster_counts = data['cluster'].value_counts()

    # We plot each cluster in a bar graph
    plt.bar(cluster_counts.index, cluster_counts.values, color=['red', 'orange', 'yellow', 'green', 'blue'])
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title(f'Number of Data Points in Each Cluster for {modelType}')
    plt.xticks(cluster_counts.index)
    plt.show()
    
def elbow_k_means(reducedData):
    sse = []
    for k in range(1,30):
        kmeans = KMeans(n_clusters=k, random_state=69)
        kmeans.fit(reducedData)
        sse.append(kmeans.inertia_)
    
    plt.plot(range(1,30),sse) 
    plt.title('The Elbow Curve for K-means')
    plt.xlabel('Number of Clusters')
    plt.ylabel("Sum of Square Intra-cluster Distances")
    plt.show()
    
def plot_dendrogram(reducedData):
    z = linkage(reducedData, method="ward")
    plt.figure(figsize=(10, 7))
    dendrogram(z)
    plt.title('Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.show()
        

reducedData = preprocess()

ktest = k_means_test(reducedData)
bi_k_test = bisecting_k_means(reducedData)
hierarchical = hierarchical_test(reducedData)

analysis(ktest, "kmeans")
analysis(bi_k_test, "bi kmeans")
analysis(hierarchical, "hierarchical")


