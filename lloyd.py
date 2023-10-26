import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

np.random.seed(42)

rawDF = pd.read_csv('Dry_Bean.csv')
mainDF = pd.get_dummies(rawDF, columns=['Class'])

pcaOBJ = PCA(n_components=2)
twoDDF = pcaOBJ.fit_transform(mainDF)
# twoDDF : two dimentional DataFrame
K = 3
centroids = []
def lloydAlgo(DF):
    #finding random centroids
    random_indices = np.random.choice(len(DF),K, replace=False)
    global centroids 
    centroids =  DF[random_indices]

    while True:
        # finding nearest centroid for every data
        labels = Clustering(DF)

        # updating centroid based on mean of each cluster
        new_centroids = update_centroids(DF, labels)

        #if the centroids converged, break
        if np.all(centroids - new_centroids <= 0.01):
            break

        centroids = new_centroids

def Clustering(DF):
    labels = []
    for x in DF:
        # Calculate distances to centroids
        distances = [np.linalg.norm(x - centroid) for centroid in centroids]

        # Assign cluster label of the nearest centroid
        label = np.argmin(distances)
        labels.append(label)

    return np.array(labels)

def update_centroids(DF, labels):
    new_centroids = []
    for i in range(K):
        cluster_points = DF[labels == i]
        
        # Calculate mean of data points in the cluster
        centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(centroid)

    return np.array(new_centroids)


lloydAlgo(twoDDF)

#plot
plt.scatter(twoDDF[:, 0], twoDDF[:, 1], c=Clustering(twoDDF), cmap='viridis')
plt.scatter(centroids[:, 0],centroids[:, 1], marker='x', color="red")
plt.show()
