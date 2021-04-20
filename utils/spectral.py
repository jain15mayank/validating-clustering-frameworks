# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:20:37 2020

@author: Mayank Jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import preProcessing_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from utils.objectiveValidation import shuffle_partition

def spectralClustering_KM_KNN_Euc(dataset, nClusters, mode="unnormalized", knnType="undirected", nNeighbors=5, randomSeed=51, nEigenVectors=None):
    # use the nearest neighbor graph as our adjacency matrix
    A = kneighbors_graph(dataset, n_neighbors=nNeighbors, mode='distance').toarray()
    if knnType=="undirected":
        A1 = np.maximum(A, A.T) # undirected k-nearest neighbors
    elif knnType=="mutual":
        A1 = np.minimum(A, A.T) # mutual k-nearest neighbors
    else:
        raise ValueError("knnType should either be 'undirected' or 'mutual' - found something else.")
    # create the graph laplacian
    D = np.diag(A1.sum(axis=1))
    if mode=="unnormalized":
        L = D-A1
    elif mode=="normalizedShi":
        L = D-A1
        try:
            L = np.linalg.inv(D)*L
        except ZeroDivisionError:
            print("ERROR finding inverse of diagonal matrix. Switching to 'unnormalized' mode.")
            mode = "unnormalized"
            L = D-A1
    elif mode=="normalizedNg":
        L = D-A1
        try:
            L = np.linalg.inv(np.sqrt(D))*L*np.linalg.inv(np.sqrt(D))
        except ZeroDivisionError:
            print("ERROR finding inverse of sqrt(diagonal matrix). Switching to 'unnormalized' mode.")
            mode = "unnormalized"
            L = D-A1
    else:
        raise ValueError("knnType should either be 'unnormalized' or 'normalizedShi' or 'normalizedNg' - found something else.")
    # find the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)
    #vals = vals.real
    #vecs = vecs.real
    # sort
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]
    # kmeans on first k (nClusters) vectors [with nonzero eigenvalues ???]
    # take only those eigenvectors with eigenvalues less than the minimum value in the diagonal of D
    if mode=="unnormalized":
        if nEigenVectors==None:
            temp = D
            temp[temp==0]=np.inf
            U = vecs[:,vals<np.amin(temp)]
        else:
            U = vecs[:,:nEigenVectors]
    else:
        U = vecs[:,vals>0.05][:,0:nClusters]
        if mode=="normalizedNg":
            U = U / U.sum(axis=1).reshape(len(U.sum(axis=1)),1)
    np.random.seed(randomSeed)
    kmeans = KMeans(n_clusters=nClusters)
    kmeans.fit(U.real)
    nEigenVectors = U.shape[1]
    return kmeans, U, nEigenVectors

def optimalKspectral(data, nrefs=3, maxClusters=15, randomSeed = 51, plotting=False, figPath=None):
    """
    Calculates spectral optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
        randomSeed: seed to random number generator
        plotting: True to plot the gap statistic results
        figPath: Only considered if plotting is True. If None, figure is displayed
                 in console instead
    Returns: optimalK
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            #km = KMeans(k)
            #km.fit(randomReference)
            km, _, _ = spectralClustering_KM_KNN_Euc(randomReference, k, randomSeed=randomSeed)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        #km = KMeans(k)
        #km.fit(data)
        km, _, _ = spectralClustering_KM_KNN_Euc(data, k, randomSeed=randomSeed)
        
        origDisp = km.inertia_
        
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    
    k = gaps.argmax() + 1 # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    gapdf = resultsdf
    if plotting:
        plt.rc('font', size=21)         # controls default text sizes
        plt.rc('axes', titlesize=22)    # fontsize of the axes title
        plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
        plt.rc('legend', fontsize=21)   # legend fontsize
        plt.rc('figure', titlesize=22)  # fontsize of the figure title
        colorBar = ['blue' for i in range(len(gapdf.clusterCount))]
        colorBar[int(gapdf[gapdf.clusterCount == k].clusterCount.values[0]-1)] = 'red'
        plt.figure(figsize=(10, 6))
        #plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
        #plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
        plt.bar(gapdf.clusterCount, gapdf.gap, align='center', alpha=0.6, color=colorBar)
        #plt.grid(True)
        low = min(gapdf.gap.values)
        high = max(gapdf.gap.values)
        plt.ylim([max(0,low-0.8*(high-low)), min(high+0.4*high, high+0.4*(high-low))])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Gap Value')
        if figPath is not None:
            plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
    return k

def validate_spectral_clusters(clusterCenters, labels, originalData, nEigenVectors, partitions=4, dimRedMethod=None, trials=100):
    """
    Computes the cluster centroids from the given dataset and clustering labels.
    Params:
        clusterCenters: ndarray of shape (n_clusters, n_features)
            - cluster centers as assigned by the algorithm which needs to be
              validated
        labels: ndarray of shape (n_samples,)
            - labels assigned by the clustering algorithm to each household
        originalData: ndarray (shape determined by the problem)
            - original data which was used for pre-processing followed by
              dimensionality reduction before passing onto for final clustering
            - this will be passed straight to the methods:
              - preProcessing_clustering:: to get ndarray of shape (n_samples, n_features)
              - shuffle_partition:: to get list of arrays similar to originalData
        nEigenVectors: int
            - number of eigenvectors used during spectral clustering
        partitions: int (>1)
            - number of partitions (of the original data) to be studied
        dimRedMethod: 'FA' or 'PCA' or None
            - Dimensionality reduction method which was used post pre-processing
        trials: int (>=1)
            - Number of times partitioning is done before averaging out the results
    Returns:
        totalCases: int
            - total number of cases for which match/mis-match is calculated
        nMatchAvg: float
            - average number of matches across trials
        nMisMatchAvg: float
            - average number of mis-matches across trials
        percentMatch: float
            - match% obtained across trials
        percentMisMatch: float
            - mis-match% obtained across trials
        sampleMisMatchFreq: ndarray of shape (n_samples,)
            - Average number of mis-matches obatined for each sample after all
              trials
            - Note: In each trial, number of times a match/mis-match is calculated
              for a particular sample is equal to the number of partitions studied
              during the validation
    """
    nClusters = clusterCenters.shape[0]
    nComponents = clusterCenters.shape[1]
    sampleMisMatchFreq = np.zeros((len(labels)))
    nMatchAvg = 0
    nMisMatchAvg = 0
    for trial in range(trials):
        # Shuffle and Partition the data
        partitionedData = shuffle_partition(originalData, partitions)
        nMatch = 0
        nMisMatch = 0
        for i in range(len(partitionedData)):
            # Perform pre-processing routines on each partitions
            # This variables shape must be (n_samples, n_features)
            processedData = preProcessing_clustering(partitionedData[i])
            # Perform dimensionality reduction on pre-processed partitions
            if dimRedMethod==None:
                processedData_reduced = processedData
            elif dimRedMethod=='PCA':
                if nComponents==None:
                    raise ValueError("nComponents cannot be None when dimRedMethod is not None.")
                pca = PCA(n_components=nComponents)
                processedData_reduced = pca.fit_transform(processedData)
            elif dimRedMethod=='FA':
                if nComponents==None:
                    raise ValueError("nComponents cannot be None when dimRedMethod is not None.")
                agglo = FeatureAgglomeration(n_clusters=nComponents)
                processedData_reduced = agglo.fit_transform(processedData)
            else:
                raise ValueError("dimRedMethod should either be 'PCA' or 'FA' or None - found something else.")
            # Perform spectral clustering's internal processing steps before
            # internal implementation of K-Means algorithm
            _, U, _ = spectralClustering_KM_KNN_Euc(processedData_reduced, nClusters, nEigenVectors=nEigenVectors)
            # Check which cluster is nearest to the newly obtained vector
            # representaions of the same sample. Note: corresponding to each
            # sample, each partition specifies a new representation of the sample.
            # In other words, original sample is divided into n (n = number of 
            # partitions) representations of itself.
            for sample in range(len(processedData_reduced)):
                temp = np.argmin(np.linalg.norm(clusterCenters - U[sample].reshape(1,-1), axis=1))
                if temp==labels[sample]:
                    nMatch+=1
                else:
                    nMisMatch+=1
                    sampleMisMatchFreq[sample]+=1
        nMatchAvg = ((nMatchAvg*trial) + nMatch)/(trial+1)
        nMisMatchAvg = ((nMisMatchAvg*trial) + nMisMatch)/(trial+1)
    totalCases = int(np.round(nMatchAvg + nMisMatchAvg))
    percentMatch = (nMatchAvg*100)/totalCases
    percentMisMatch = (nMisMatchAvg*100)/totalCases
    return totalCases, nMatchAvg, nMisMatchAvg, percentMatch, percentMisMatch, sampleMisMatchFreq