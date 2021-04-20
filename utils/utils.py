# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:21:03 2020

@author: Mayank Jain
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def preProcessing_clustering(originalData):
    """
    Performs the pre-processing steps on the original data to make it ready for
    clustering framework. Highly specific to the problem at hand.
    Params:
        originalData: ndarray of shape (n_samples, n_days, n_hoursPerDay)
            - original data which needs to be pre-processed to make it ready for
              the clustering framework
            - In this case: complete_day_series
    Returns:
        processedData: ndarray of shape (n_samples, n_features)
            - data obtained after performing the pre-processing steps on the
              original data
            - Note: n_features represents original features and not the ones
              obtained after dimensionality reduction
    """
    median_series = np.nanmedian(originalData, axis=1)
    median_series_norm = median_series / (np.linalg.norm(median_series, axis=1)[:,None])
    processedData = median_series_norm
    return processedData

def calc_cluster_centroids(data, labelsMat):
    """
    Computes the cluster centroids from the given dataset and clustering labels.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data elements which were clustered
        labelsMat: ndarray of shape (n_samples, n_clusteringAlgorithms)
            - labels assigned by each clustering algorithms stored in columns
            - assigned labels are in the range [0, n_clusters)
    Returns:
        centersMat: list[ndarray (n_clusters, n_features)] of length n_clusteringAlgorithms
            - every element in the list is a numpy array containing positions of
              cluster centroids for the clusters defined by a clustering algorithm
            - list elements arranges according to the labelsMat
    """
    # Change labelsMat shape if only one clustering algorithm is used
    if len(labelsMat.shape)==1:
        labelsMat = np.expand_dims(labelsMat, 1)
    centersMat = []
    for i in range(labelsMat.shape[1]): #Iterate over n_clusteringAlgorithms
        nClusters = np.max(labelsMat[:,i]) + 1 #Find number of clusters corresponding to each algorithm
        clusterCenter = np.zeros((nClusters, data.shape[1]))
        count = np.zeros((nClusters, data.shape[1]))
        for j in range(data.shape[0]): #Iterate over n_samples
            clusterCenter[labelsMat[j,i]] += data[j]
            count[labelsMat[j,i], :] += 1
        centersMat.append(clusterCenter/count)
    return centersMat

def plotClusters(data, labels, savePath):
    """
    Plots all the households' pre-processed median curves (before dimensionality
    reduction) grouped together according to the assigned cluster labels.
    Params:
        data: ndarray of shape (n_samples, n_originalFeatures)
            - original data elements which were passed to the clustering
              framework (dimensionality reduction + clustering algorithm)
        labels: ndarray of shape (n_samples,)
            - labels assigned to each sample
            - assigned labels are in the range [0, nClusters)
        savePath: string
            - path to the directory where the resulting plots will be stored
            - the path will only be appended by '<clusterId>.pdf'
            - hence, make sure the path also contains any prefix to the file
              name (if required)
    Returns:
        None
    """
    nClusters = np.max(labels)+1
    
    plt.rc('font', size=21)         # controls default text sizes
    plt.rc('axes', titlesize=22)    # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=17)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)   # fontsize of the tick labels
    plt.rc('legend', fontsize=19)   # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    
    for clusterId in range(nClusters):
        plt.figure(figsize=(15, 7))
        count = 0
        for houseIdx in range(len(labels)):
            if labels[houseIdx]==clusterId:
                plt.plot(data[houseIdx], label="Daily profile of House ID " + str(houseIdx))
                count+=1
        plt.xlim(-0.5, 23.5)
        plt.ylim(0, 0.6)
        
        #locs, labels = plt.xticks()
        locs = np.arange(data.shape[-1])
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-02')
        t = np.linspace(start.value, end.value, data.shape[-1], endpoint=False)
        t = pd.to_datetime(t)
        newT = []
        for j in range(len(t)):
            newT.append(datetime.datetime.strptime(str(t[j]), '%Y-%m-%d %H:%M:%S').strftime('%H:%M'))
        plt.xticks(locs, newT)
        plt.xticks(rotation=30)
        
        plt.xlabel("Time Of The Day")
        plt.ylabel("Load Consumption (Watts) - Normalized")
        if count>5:
            plt.legend(ncol=2)
        else:
            plt.legend()
        #plt.legend()
        plt.savefig(savePath+str(clusterId)+'.png', bbox_inches = 'tight', pad_inches = 0.05)
        plt.savefig(savePath+str(clusterId)+'.pdf', bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()