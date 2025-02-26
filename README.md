# Clustering Marketing Data

This is a program that clusters data scraped from teenagers' social media pages into meaningful groups. The data is clustered through various machine learning algorithms, such as K-means, bisecting K-means, hierarchical agglomerative clustering, and DBSCAN.

The dataset used was titled “Students’ Social Network Profile Clustering”. It was composed of 15 000 records of highschool students who maintained social profiles on a popular social network from
2006-2009. Features included things such as graduation year, gender, age, sex, and number of friends. It contains the counts of the 37 most dominant words found in the profiles, such as “football”, “drugs”, and so on.

# Methods

Prior to clustering the data, it is important to engage in data preprocessing. Continuous data such as gender was binarized. For unavailable age values, they were filled in with the average age from the dataset. This was because it was important to keep as many samples as possible to derive meaningful results. Data normalization was also done to avoid certain attributes from being over represented. Principal component analysis (PCA) was used in order to figure out the 7 most representative attributes for the dataset at large.

Following this, we trained various clustering algorithms. The amount of clusters was determined using the elbow method, graphing the number of clusters against square intra cluster distance. The elbow of the graph is the optimal amount. This was a total of five clusters. This value was used for both k-means and bisecting k-means

Agglomerative hierarchical clustering was used to determine relevant hierarchical data, like possible social cliques, friend groups, and so on.

# Results

When graphing the results of the clustering data, we had the following results. There was a relationship between the number of friends a person may have, and the average of the associated principal components. This may suggest that people with more friends form a distinct social clique, different from people with less friends. We also conclude that there are some clusters with a higher proportion of females than would be expected given the dataset. This suggests that women may be more likely to have more female-centric social networks. From a marketing perspective, these clusters should be marketed with more female-centric marketing. The same thing follows for males. The average age of a cluster did not vary, and suggests that age has less of an impact in forming a social clique.
