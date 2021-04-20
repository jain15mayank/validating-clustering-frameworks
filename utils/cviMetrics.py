# -*- coding: utf-8 -*-
"""
Created on Sat May 30 07:46:00 2020

@author: Mayank Jain
"""
from warnings import warn
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import distance_metrics
from utils.utils import calc_cluster_centroids

def intercluster_dist(data, labelsMat, mode='min', dist_metric = 'euclidean'):
    """
    Computes the intercluster distance between each pair of clusters from the 
    given dataset and clustering labels.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labelsMat: ndarray of shape (n_samples, n_clusteringAlgorithms)
            - labels assigned by each clustering algorithms stored in columns
            - assigned labels are in the range [0, n_clusters]
        mode: Either of the three defined below
            - 'min' :: Minimum distance between any pair of points (1 from
              each cluster)
            - 'max' :: Maximum distance between any pair of points (1 from
              each cluster)
            - 'centroid' :: Distance between centroids of the 2 clusters
        dist_metric: string, can be one of the following:
            - 'manhattan' or 'l1'
            - 'euclidean' or 'l2'
            - 'cosine'
            - 'haversine'
    Returns:
        interClusterDist: list[ndarray (n_clusters, n_clusters)] of length n_clusteringAlgorithms
            - every element in the list is a numpy array containing intercluster
              distance matrix for the clusters defined by the a clustering algorithm
            - list elements arranges according to the labelsMat
    """
    # Change labelsMat shape if only one clustering algorithm is used
    if len(labelsMat.shape)==1:
        labelsMat = np.expand_dims(labelsMat, 1)
    interClusterDist = []
    if mode == 'centroid':
        centersMat = calc_cluster_centroids(data, labelsMat)
    for j in range(labelsMat.shape[1]): #Iterate over n_clusteringAlgorithms
        nClusters = np.max(labelsMat[:,j]) + 1 #Find number of clusters corresponding to each algorithm
        clusterDistMat = np.zeros((nClusters, nClusters))
        if mode == 'min' or mode =='max':
            clusters = dict()
            for i in range(data.shape[0]): #Iterate over n_samples
                clusters.setdefault(labelsMat[i,j], []).append(data[i,:])
        for m in range(nClusters):
            for n in range(m+1, nClusters):
                if mode == 'min':
                    clusterDistMat[m][n] = np.min(distance_metrics()[dist_metric]
                                        (clusters[m],clusters[n]))
                    clusterDistMat[n][m] = clusterDistMat[m][n]
                elif mode == 'max':
                    clusterDistMat[m][n] = np.max(distance_metrics()[dist_metric]
                                        (clusters[m],clusters[n]))
                    clusterDistMat[n][m] = clusterDistMat[m][n]
                elif mode == 'centroid':
                    clusterDistMat[m][n] = np.max(distance_metrics()[dist_metric]
                                        ((centersMat[j][m], centersMat[j][n])))
                    clusterDistMat[n][m] = clusterDistMat[m][n]
                else:
                    raise Exception("Unsupported MODE to calculate intercluster distance")
        interClusterDist.append(clusterDistMat)
    return interClusterDist

def cluster_diameter(data, labelsMat, mode='max', dist_metric = 'euclidean'):
    """
    Computes the diameter for each cluster from the given dataset and 
    clustering labels.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labelsMat: ndarray of shape (n_samples, n_clusteringAlgorithms)
            - labels assigned by each clustering algorithms stored in columns
            - assigned labels are in the range [0, n_clusters]
        mode: Either of the four defined below
            - 'max' :: Maximum distance between any 2 points of a cluster
            - 'avg' :: Mean distance between all pairs within the cluster
            - 'avg_centroid' :: Twice the mean distance of every point from 
              cluster centroid
            - 'far_centroid' :: Twice the distance between the centroid and the
              farthest point from it within the same cluster
        dist_metric: string, can be one of the following:
            - 'manhattan' or 'l1'
            - 'euclidean' or 'l2'
            - 'cosine'
            - 'haversine'
    Returns:
        diamCluster: list[ndarray (n_clusters,)] of length n_clusteringAlgorithms
            - every element in the list is a numpy array containing diameter of
              the clusters defined by the a clustering algorithm
            - list elements arranges according to the labelsMat
    """
    # Change labelsMat shape if only one clustering algorithm is used
    if len(labelsMat.shape)==1:
        labelsMat = np.expand_dims(labelsMat, 1)
    diamCluster = []
    for j in range(labelsMat.shape[1]): #Iterate over n_clusteringAlgorithms
        nClusters = np.max(labelsMat[:,j]) + 1 #Find number of clusters corresponding to each algorithm
        clusterDiameter = np.zeros((nClusters))
        clusters = dict() # Dictionary of clusters with labels as key and list of numpy vectors (samples corresponding to that label) as values
        for i in range(data.shape[0]): #Iterate over n_samples
            clusters.setdefault(labelsMat[i,j], []).append(data[i,:])
        for k in range(nClusters):
            if mode == 'max':
                if len(clusters[k]) == 1:
                    clusterDiameter[k] = 0
                else:
                    clusterDiameter[k] = np.max(distance_metrics()['euclidean']
                                        (clusters[k]))
            elif mode == 'avg':
                if len(clusters[k]) == 1:
                    clusterDiameter[k] = 0
                else:
                    clusterDiameter[k] = np.sum(distance_metrics()['euclidean']
                                        (clusters[k]))/(2*len(clusters[k]))
            elif mode == 'avg_centroid':
                if len(clusters[k]) == 1:
                    clusterDiameter[k] = 0
                else:
                    centersMat = calc_cluster_centroids(data, labelsMat)
                    clusterDiameter[k] = 2*np.sum(distance_metrics()['euclidean']
                                        (np.array(clusters[k]),centersMat[j][k].reshape(1,-1)))/(len
                                        (clusters[k]))
            elif mode == 'far_centroid':
                if len(clusters[k]) == 1:
                    clusterDiameter[k] = 0
                else:
                    centersMat = calc_cluster_centroids(data, labelsMat)
                    clusterDiameter[k] = 2*np.max(distance_metrics()['euclidean']
                                        (np.array(clusters[k]),centersMat[j][k].reshape(1,-1)))
            else:
                raise Exception("Unsupported MODE to calculate cluster diameter")
        diamCluster.append(clusterDiameter)
    return diamCluster

def dunn_score(data, labelsMat, dist_metric = 'euclidean'):
    """
    Computes the Dunn index score for each clustering algorithm whose corresponding
    labels are defined as columns of the labelsMat.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labelsMat: ndarray of shape (n_samples, n_clusteringAlgorithms)
            - labels assigned by each clustering algorithms stored in columns
        dist_metric: string, can be one of the following:
            - 'manhattan' or 'l1'
            - 'euclidean' or 'l2'
            - 'cosine'
            - 'haversine'
    Returns:
        scoreArray: ndarray of shape (n_clusteringAlgorithms)
            - The resulting Dunn Index score correpsonding to each clustering
              algorithm in the order defined in labelsMat
    """
    # Change labelsMat shape if only one clustering algorithm is used
    if len(labelsMat.shape)==1:
        labelsMat = np.expand_dims(labelsMat, 1)
    
    # Set calculation mode
    interClusterDistanceMode = 'min'
    clusterDiameterMode = 'max'
    
    # Determine cluster diameter and intercluster distances
    diamCluster = cluster_diameter(data, labelsMat, mode=clusterDiameterMode, dist_metric = dist_metric)
    interClusterDist = intercluster_dist(data, labelsMat, mode=interClusterDistanceMode, dist_metric = dist_metric)
    
    scoreArray = np.zeros((labelsMat.shape[1]))
    for i in range(labelsMat.shape[1]): #Iterate over n_clusteringAlgorithms
        # Setting diagonal elements to infinity to ignore them
        np.fill_diagonal(interClusterDist[i], np.inf)
        if np.max(diamCluster[i]) == 0:
            scoreArray[i] = np.inf
        else:
            scoreArray[i] = np.min(interClusterDist[i])/np.max(diamCluster[i])
    return scoreArray

def xie_beni_score(data, labels=None, fuzzyMembershipMat=None, fuzzifier=2):
    """
    Computes the Xie-Beni index score for the clustering algorithm whose
    corresponding labels are defined in the vector 'labels' or the membership
    of each sample point is defined in the fuzzy membership matrix (i.e.
    'fuzzyMembershipMat'). If the value of 'fuzzyMembershipMat' remains 'None'
    upon call to this function, 'labels' vector is used to define membership of
    each sample to the labelled cluster as 1 while others as 0. In that case,
    cluster center will also be computed as the mean of all the points belonging
    to that cluster.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labels: ndarray of shape (n_samples,)
            - labels assigned by the clustering algorithms stored in columns
            - 'None' signifies that the provided algorithm is Fuzzy C-Means
        fuzzyMembershipMat: ndarray of shape (n_clusters, n_samples)
            - matrix defining the fuzzy membership of each sample point to the
              generated clusters
            - 'None' signifies that the provided algorithm is NOT Fuzzy C-Means
        fuzzifier: float in range (1,2]
            - fuzzifier used in Fuzzy C-Means Algorithm
            - Only considered if 'fuzzyMembershipMat' is not None
    Returns:
        score: float
            - The resulting Xie-Beni Index score correpsonding to the given
              clustering algorithm.
    """
    # Check if all input parameters are sepcified correctly
    if fuzzyMembershipMat is None:
        if labels is None:
            raise Exception("Neither fuzzy membership matrix, nor fixed labels provided.")
        else: # Create fuzzyMembershipMat based on labels
            nClusters = np.max(labels) + 1 # Find number of clusters
            fuzzyMembershipMat = np.zeros((nClusters, data.shape[0]))
            for i in range(data.shape[0]):
                fuzzyMembershipMat[labels[i],i] = 1
    else:
        nClusters = fuzzyMembershipMat.shape[0] # Find number of clusters
        if labels is not None:
            warn("Since both labels and fuzzy membership matrix are provided, latter one is used.")
    if fuzzifier<=1 or fuzzifier>2:
        raise Exception("Value of fuzzifier must lie in range: (1,2]")
    
    # Compute Fuzzy Centroids (n_clusters, n_features)
    fuzzyCentroids = np.zeros((nClusters, data.shape[1]))
    for i in range(nClusters):
        fuzzyCentroids[i] = np.matmul(np.power(fuzzyMembershipMat[i],fuzzifier), 
                            data)/np.sum(np.power(fuzzyMembershipMat[i],fuzzifier))
    
    # Compute Separation
    interClusterDist = distance_metrics()['euclidean'](fuzzyCentroids)
    np.fill_diagonal(interClusterDist, np.inf)
    separation = np.power(np.min(interClusterDist), 2)
    
    # Compute Total Variance
    sigma = 0
    for i in range(nClusters):
        sigma += np.sum(np.multiply(np.power(fuzzyMembershipMat[i], fuzzifier),
                                    np.power(distance_metrics()['euclidean'](
                                            np.expand_dims(fuzzyCentroids[i],
                                                           axis=0), data), 2)))
    
    # Compute Compactness
    compactness = sigma/data.shape[0]
    
    # Compute Score
    score = compactness/separation
    return score

def compute_cvi(data, labelsMat=None, cvi_metric = 'silhouette', dist_metric = 'euclidean', fuzzyMemMat=None, fuzzifier=2):
    """
    Computes the validation score and gives recommendation for the best clustering
    algorithm based on the defined Cluster Validation Index.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labelsMat: ndarray of shape (n_samples, n_clusteringAlgorithms)
            - labels assigned by each clustering algorithms stored in columns
            - Can be None if only Fuzzy C-Means algorithms is employed
        cvi_metric: 
        dist_metric: string, can be one of the following:
            - 'manhattan' or 'l1'
            - 'euclidean' or 'l2'
            - 'cosine'
            - 'haversine'
        fuzzyMemMat: ndarray of shape (n_clusters, n_samples)
            - matrix defining the fuzzy membership of each sample point to the
              generated clusters
            - 'None' signifies that no Fuzzy C-Means algorithm is employed
            - Note: There are no labels in labelsMat corresponding to the Fuzzy
              C-Means Algorithm
        fuzzifier: float in range (1,np.inf); generally less than 2
            - fuzzifier used in Fuzzy C-Means Algorithm
            - Only considered if 'fuzzyMembershipMat' is not None
    Returns:
        bestAlgorithmIdx: list
            - indices, corresponding to the columns in 'labelsMat', of the 
              algorithms which are deemed best according to the cvi_matric
            - if the Fuzzy C-Means algorithm is declared the best, the value #
              will be -1
        scoreArray: ndarray of shape (n_clusteringAlgorithms,)
            - complete scoring array containing scores corresponding to the
              clustering algorithms - scores given in order defined in labelsMat
    """
    fuzzyFlag = 0
    if labelsMat is None:
        if fuzzyMemMat is None:
            raise Exception("Neither hard labels, nor fuzzy membership matrix is provided.")
        if not cvi_metric == 'XBscore':
            # Convert fuzzy membership matrix into hard labels and store it in 
            # labelsMat
            labelsMat = np.expand_dims(np.argmax(fuzzyMemMat, axis=0), 1)
            n_clusteringAlgorithms = labelsMat.shape[1]
            fuzzyFlag = 1
    else:
        # Change labelsMat shape if only one clustering algorithm is used
        if len(labelsMat.shape)==1:
            labelsMat = np.expand_dims(labelsMat, 1)
        if (fuzzyMemMat is not None) and (not cvi_metric == 'XBscore'):
            # Convert fuzzy membership matrix into hard labels and store it in 
            # labelsMat
            labelsMat = np.concatenate(
                    (labelsMat, np.expand_dims(np.argmax(fuzzyMemMat, axis=0),
                                               1)), axis=1)
            fuzzyFlag = 1
        n_clusteringAlgorithms = labelsMat.shape[1]
    if cvi_metric == 'silhouette':
        '''#Silhouette Score#'''
        #Range: [-1, 1]; -1: Incorrect; 0: Overlapping; 1:Perfect
        scoreArray = np.zeros((n_clusteringAlgorithms))
        bestAlgorithmIdx = []
        bestScore = -2
        for i in range(n_clusteringAlgorithms):
            scoreArray[i] = silhouette_score(data, labelsMat[:,i], metric=dist_metric)
            if scoreArray[i] > bestScore:
                bestAlgorithmIdx = [i]
                bestScore = scoreArray[i]
            elif scoreArray[i] == bestScore:
                bestAlgorithmIdx.append(i)
        if fuzzyFlag==1:
            try:
                bestAlgorithmIdx[bestAlgorithmIdx.index(
                                                n_clusteringAlgorithms-1)] = -1
            except ValueError:
                pass
    elif cvi_metric == 'CHscore':
        '''#Calinski-Harabasz Index#'''
        #Range: not defined; Higher => better
        scoreArray = np.zeros((n_clusteringAlgorithms))
        bestAlgorithmIdx = []
        bestScore = -np.inf
        for i in range(n_clusteringAlgorithms):
            scoreArray[i] = calinski_harabasz_score(data, labelsMat[:,i])
            if scoreArray[i] > bestScore:
                bestAlgorithmIdx = [i]
                bestScore = scoreArray[i]
            elif scoreArray[i] == bestScore:
                bestAlgorithmIdx.append(i)
        if fuzzyFlag==1:
            try:
                bestAlgorithmIdx[bestAlgorithmIdx.index(
                                                n_clusteringAlgorithms-1)] = -1
            except ValueError:
                pass
    elif cvi_metric == 'DBscore':
        '''#Davies-Bouldin Index#'''
        #Range: [0, np.inf); Lower => better; 0 being technically unrealistic
        scoreArray = np.zeros((n_clusteringAlgorithms))
        bestAlgorithmIdx = []
        bestScore = np.inf
        for i in range(n_clusteringAlgorithms):
            scoreArray[i] = davies_bouldin_score(data, labelsMat[:,i])
            if scoreArray[i] < bestScore:
                bestAlgorithmIdx = [i]
                bestScore = scoreArray[i]
            elif scoreArray[i] == bestScore:
                bestAlgorithmIdx.append(i)
        if fuzzyFlag==1:
            try:
                bestAlgorithmIdx[bestAlgorithmIdx.index(
                                                n_clusteringAlgorithms-1)] = -1
            except ValueError:
                pass
    elif cvi_metric == 'DIscore':
        '''#Dunn Index#'''
        #Range: [0, np.inf); higher => better; np.inf being technically unrealistic
        scoreArray = dunn_score(data, labelsMat, dist_metric = dist_metric)
        bestAlgorithmIdx = []
        bestScore = -1
        for i in range(n_clusteringAlgorithms):
            if scoreArray[i] > bestScore:
                bestAlgorithmIdx = [i]
                bestScore = scoreArray[i]
            elif scoreArray[i] == bestScore:
                bestAlgorithmIdx.append(i)
        if fuzzyFlag==1:
            try:
                bestAlgorithmIdx[bestAlgorithmIdx.index(
                                                n_clusteringAlgorithms-1)] = -1
            except ValueError:
                pass
    elif cvi_metric == 'XBscore':
        '''#Xie-Beni Index#'''
        #Range: [0, np.inf); lower => better
        if labelsMat is None:
            scoreArray = np.array([xie_beni_score(data, fuzzyMembershipMat=fuzzyMemMat, fuzzifier=fuzzifier)])
            bestAlgorithmIdx = [-1]
        elif fuzzyMemMat is None:
            scoreArray = np.zeros((n_clusteringAlgorithms))
            bestAlgorithmIdx = []
            bestScore = np.inf
            for i in range(n_clusteringAlgorithms):
                scoreArray[i] = xie_beni_score(data, labels=labelsMat[:,i])
                if scoreArray[i] < bestScore:
                    bestAlgorithmIdx = [i]
                    bestScore = scoreArray[i]
                elif scoreArray[i] == bestScore:
                    bestAlgorithmIdx.append(i)
        else:
            scoreArray = np.zeros((n_clusteringAlgorithms))
            bestAlgorithmIdx = []
            bestScore = np.inf
            for i in range(n_clusteringAlgorithms):
                scoreArray[i] = xie_beni_score(data, labels=labelsMat[:,i])
                if scoreArray[i] < bestScore:
                    bestAlgorithmIdx = [i]
                    bestScore = scoreArray[i]
                elif scoreArray[i] == bestScore:
                    bestAlgorithmIdx.append(i)
            scoreFuzzy = xie_beni_score(data, fuzzyMembershipMat=fuzzyMemMat, fuzzifier=fuzzifier)
            scoreArray = np.concatenate((scoreArray, np.array([scoreFuzzy])))
            if scoreFuzzy < bestScore:
                bestAlgorithmIdx = [-1]
            elif scoreFuzzy == bestScore:
                bestAlgorithmIdx.append(-1)
    else:
        raise Exception("Unsupported CVI Metric Encountered")
    return bestAlgorithmIdx, scoreArray