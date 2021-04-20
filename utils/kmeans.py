# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 01:14:39 2020

@author: Mayank Jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans(data, nClusters, randomSeed=51):
    """
    Uses SkLearn implementation of K-Means++ algorithm to determine cluster
    centers and labels for each sample point in the dataset
    Params:
        data: ndarry of shape (n_samples, n_features)
        nClusters: number of clusters to be made
        randomSeed: seed to random number generator
    """
    np.random.seed(randomSeed)
    km = KMeans(nClusters)
    km.fit(data)
    return km.cluster_centers_, km.labels_

def optimalK(data, nrefs=3, maxClusters=15, randomSeed=51, plotting=False, figPath=None):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
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
            np.random.seed(randomSeed)
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        np.random.seed(randomSeed)
        km = KMeans(k)
        km.fit(data)
        
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
