# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:55:47 2020

@author: Mayank Jain
"""

import numpy as np
from utils.utils import preProcessing_clustering
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

def shuffle_partition(originalData, partitions=4):
    """
    Shuffles and partitions the given data for validation. Highly specific to
    the problem at hand.
    Params:
        originalData: ndarray of shape (n_samples, n_days, n_hoursPerDay)
            - original data which needs to be shuffled and partitioned to make
              it ready for the objective validation
            - In this case: complete_day_series
    Returns:
        partitionedList: list
            - list of partitioned arrays
    """
    # Swap axes in order to shuffle the data w.r.t. axis=1 at later stage
    series = np.swapaxes(originalData, 0, 1)
    # Generate new random seed from the CPU clock cycle to effect maximum
    # randomization
    np.random.seed()
    # Shuffle the original data
    shuffled_series = np.random.permutation(series)
    # Swap axes back to restore orignal configuration
    shuffled_series = np.swapaxes(shuffled_series, 0, 1)
    # Split the shuffled data in said number of partitions w.r.t. axis=1
    partitionedList = np.array_split(shuffled_series, partitions, axis=1)
    return partitionedList

def validate_clusters(clusterCenters, labels, originalData, partitions=4, dimRedMethod=None, trials=100):
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
    #nClusters = clusterCenters.shape[0]
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
            # Check which cluster is nearest to the newly obtained vector
            # representaions of the same sample. Note: corresponding to each
            # sample, each partition specifies a new representation of the sample.
            # In other words, original sample is divided into n (n = number of 
            # partitions) representations of itself.
            for sample in range(len(processedData_reduced)):
                temp = np.argmin(np.linalg.norm(clusterCenters - processedData_reduced[sample].reshape(1,-1), axis=1))
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
