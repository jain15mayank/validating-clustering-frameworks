# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:41:15 2020

@author: Mayank Jain
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.utils import calc_cluster_centroids
import matplotlib.pyplot as plt

def agglomerativeClustering(data, distance_threshold=0.5, n_clusters=None):
    """
    Params:
        data: ndarry of shape (n_samples, n_features)
        distance_threshold: Optimal threshold value for similarity measure
        n_clusters: Number of clusters (if known a priori)
    Returns: (clusterCenters, labels)
    """
    if n_clusters is None:
        agglo = AgglomerativeClustering(n_clusters=None,distance_threshold=distance_threshold)
    else:
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
    clustering = agglo.fit(data)
    # clusterCenters.shape = (nClusters, nFeatures)
    clusterCenters = calc_cluster_centroids(data, clustering.labels_.reshape(-1,1))[0]
    return clusterCenters, clustering.labels_

def elbowHeuristic_Agglo(data, markX=None, markY=None, annotX=None, annotY=None, figPath=None):
    """
    Given the data on which PCA needs to be implemented, this function will
    plot the curve for elbow heuristics. Pass annotation parameters after manual
    inspection for final graph.
    Params:
        data: ndarry of shape (n_samples, n_features)
            - data on which FA needs to be performed
        markX: float
            - x-coordinate of the elbow point
        markY: float
            - y-coordinate of the elbow point
        annotX: float
            - x-coordinate where annotation text needs to be placed
        annotY: float
            - y-coordinate where annotation text needs to be placed
        figPath: string
            - Path where figure needs to be stored. If None, the figure is
              plotted in console itself.
    Returns: None
    """
    distThr = []
    nClusters = []
    for i in np.arange(0,1,0.05):
        distThr.append(i)
        agglo = AgglomerativeClustering(n_clusters=None,distance_threshold=i)
        clustering = agglo.fit(data)
        nClusters.append(clustering.n_clusters_)
    plt.rc('font', size=21)         # controls default text sizes
    plt.rc('axes', titlesize=22)    # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
    plt.rc('legend', fontsize=21)   # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    plt.figure(figsize=(10,6))
    plt.plot(distThr, nClusters)
    if markX is not None and markY is not None:
        plt.hlines(markY, 0, markX, linestyles='dashed')
        plt.vlines(markX, 0, markY, linestyles='dashed')
        plt.scatter([markX],[markY], c='r')
        if annotX is not None and annotY is not None:
            if annotY>markY:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY+0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"));
            else:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY-0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"));
    plt.xlim(0, 1)
    plt.ylim(0, 27)
    plt.xlabel("Distance Threshold")
    plt.ylabel("Number of Clusters")
    if figPath is not None:
        plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()


'''
data = day_median_consumption_reduced
temp = []
temp2 = []
for i in np.arange(0,1,0.05):
    temp.append(i)
    agglo = AgglomerativeClustering(n_clusters=None,distance_threshold=i)
    clustering = agglo.fit(data)
    temp2.append(clustering.n_clusters_)
plt.rc('font', size=15)         # controls default text sizes
plt.rc('axes', titlesize=17)    # fontsize of the axes title
plt.rc('axes', labelsize=17)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)   # fontsize of the tick labels
plt.rc('ytick', labelsize=15)   # fontsize of the tick labels
plt.rc('legend', fontsize=15)   # legend fontsize
plt.rc('figure', titlesize=17)  # fontsize of the figure title
plt.figure(figsize=(10,6))
plt.plot(temp, temp2)
plt.hlines(4, 0, 0.5, linestyles='dashed')
plt.vlines(0.5, 0, 4, linestyles='dashed')
plt.annotate('Elbow point\n(0.5, 4)', xy=(0.5, 4.2),
             xytext=(0.70, 12), arrowprops=dict(arrowstyle="->",
             connectionstyle="angle3,angleA=0,angleB=-90"));
plt.scatter([0.5],[4], c='r')
plt.xlim(0, 1)
plt.ylim(0, 27)
plt.xlabel("Distance Threshold")
plt.ylabel("Number of Clusters")
plt.savefig('./Elbow_Agglomerative.pdf', bbox_inches = 'tight', pad_inches = 0.05)
plt.close()


data = day_median_consumption_reducedFA
temp = []
temp2 = []
for i in np.arange(0,1,0.05):
    temp.append(i)
    agglo = AgglomerativeClustering(n_clusters=None,distance_threshold=i)
    clustering = agglo.fit(data)
    temp2.append(clustering.n_clusters_)
plt.rc('font', size=15)         # controls default text sizes
plt.rc('axes', titlesize=17)    # fontsize of the axes title
plt.rc('axes', labelsize=17)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)   # fontsize of the tick labels
plt.rc('ytick', labelsize=15)   # fontsize of the tick labels
plt.rc('legend', fontsize=15)   # legend fontsize
plt.rc('figure', titlesize=17)  # fontsize of the figure title
plt.figure(figsize=(10,6))
plt.plot(temp, temp2)
plt.hlines(4, 0, 0.3, linestyles='dashed')
plt.vlines(0.3, 0, 4, linestyles='dashed')
plt.annotate('Elbow point\n(0.3, 4)', xy=(0.3, 4.2),
             xytext=(0.70, 12), arrowprops=dict(arrowstyle="->",
             connectionstyle="angle3,angleA=0,angleB=-90"));
plt.scatter([0.3],[4], c='r')
plt.xlim(0, 1)
plt.ylim(0, 27)
plt.xlabel("Distance Threshold")
plt.ylabel("Number of Clusters")
plt.savefig('./Elbow_AgglomerativeFA.pdf', bbox_inches = 'tight', pad_inches = 0.05)
plt.close()
'''