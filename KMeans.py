import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def Kmeans(X, k = 5, epochs = 100):
    #Number of observations in the data
    m = X.shape[0]
    #Number of features in the data
    n = X.shape[1]

    #Radomly assign data points as cluster centers
    centers = np.array([]).reshape(n, 0)
    for i in range(k):
        rand = random.randrange(0, m)
        centers = np.c_[centers, X[rand]]

    Clusters = {}
    #Train for number of iterations specified
    for epoch in range(epochs):
        #Initialise distance array. It will be a m * k array.
        #Having distance of each point from every cluster center
        dist = np.array([]).reshape(m, 0)
        for i in range(k):
            diff = (X - centers[:, i].T) ** 2
            t_dist = np.sum(diff, axis = 1)
            dist = np.c_[dist, t_dist]
        #For every point i, find the closest cluster center C[i]
        C = np.argmin(dist, axis = 1)

        #Reassign every point to the closest cluster based on distance
        clusters = {}
        for i in range(k):
            clusters[i] = np.array([]).reshape(n, 0)
        for i in range(m):
            clusters[C[i]] = np.c_[clusters[C[i]], X[i]]
        for i in range(k):
            clusters[i] = clusters[i].T
            #Calculate new cluster center after reassignment of points.
            centers[:, i] = np.mean(clusters[i], axis = 0)
        Clusters = clusters

    #Return clusters and cluster centers
    return Clusters, centers.T
    
if __name__ == '__main__':
    df = pd.read_csv('mall.csv')
    print(df)
    #Taking Annual Income (k$) & Spending Score (1-100) as input for simplicity
    df = df.iloc[:, [3, 4]]

    #Elbow Curve - WCSS
    WCSS = np.array([])
    for k in range(1, 11):
        clusters, centers = Kmeans(df.to_numpy(), k)
        wcss = 0
        for i in range(k):
            wcss += np.sum((clusters[i] - centers[i, :]) ** 2)
        WCSS = np.append(WCSS, wcss)
    #Plot
    plt.plot(np.array(range(1, 11)), WCSS)
    plt.xlabel('Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Curve')
    plt.show()

    k = 5
    clusters, centers = Kmeans(df.to_numpy(), k = k)
    cmap = ['r', 'g', 'b', 'c', 'm']
    #Visualise the data
    plt.subplot(1, 2, 1)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('DATA')
    #Visualise after clustering
    plt.subplot(1, 2, 2)
    for i in range(5):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], c = cmap[i])
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('CLUSTERS')
    plt.show()
