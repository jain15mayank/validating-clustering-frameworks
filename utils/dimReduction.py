# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:24:11 2020

@author: Mayank Jain
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

def dim_reduction_PCA(data, cumulative_explained_variance_ratio_threshold=0.96):
    """
    Given the data and 'explained variance ratio' threshold value, returns the
    performs the dimensionality reduction using PCA (sklearn.decomposition.PCA)
    by first finding the required number of components according to the threshold
    Params:
        data: ndarry of shape (n_samples, n_features)
        cumulative_explained_variance_ratio_threshold: Threshold for EVR
    Returns: (reducedDimData, nReducedComponents)
    """
    pca1 = PCA()
    pca1.fit(data)
    cumulative_EVR = 0
    for i in range(pca1.n_components_):
        cumulative_EVR += pca1.explained_variance_ratio_[i]
        if cumulative_EVR>=cumulative_explained_variance_ratio_threshold:
            break
    pca2 = PCA(n_components=i+1)
    reducedDimData = pca2.fit_transform(data)
    return reducedDimData, i+1

def elbowHeuristic_PCA(data, markX=None, markY=None, annotX=None, annotY=None, figPath=None):
    """
    Given the data on which PCA needs to be implemented, this function will
    plot the curve for elbow heuristics. Pass annotation parameters after manual
    inspection for final graph.
    Params:
        data: ndarry of shape (n_samples, n_features)
            - data on which PCA needs to be performed
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
    CEVR = []
    nComponents = []
    pca1 = PCA()
    pca1.fit(data)
    for comp in range(pca1.n_components_):
        nComponents.append(comp+1)
        temp = pca1.explained_variance_ratio_[comp]
        if comp>0:
            CEVR.append(CEVR[-1]+temp)
        else:
            CEVR.append(temp)
    plt.rc('font', size=21)         # controls default text sizes
    plt.rc('axes', titlesize=22)    # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
    plt.rc('legend', fontsize=21)   # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    plt.figure(figsize=(10,6))
    plt.plot(CEVR, nComponents)
    if markX is not None and markY is not None:
        plt.hlines(markY, 0, markX, linestyles='dashed')
        plt.vlines(markX, 0, markY, linestyles='dashed')
        plt.scatter([markX],[markY], c='r')
        if annotX is not None and annotY is not None:
            if annotY>markY:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY+0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"),
                             fontsize=21);
            else:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY-0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"),
                             fontsize=21);
    plt.xlim(0.4, 1)
    plt.ylim(0, data.shape[1])
    plt.xlabel("Cumulative Explained Variance Ratio")
    plt.ylabel("Reduced Dimension (d')")
    if figPath is not None:
        plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()

def dim_reduction_FA(data, distance_threshold=0.45):
    """
    Params:
        data: ndarry of shape (n_samples, n_features)
        distance_threshold: Optimal threshold value for similarity measure
    Returns: (reducedDimData, nReducedComponents)
    """
    agglo = FeatureAgglomeration(n_clusters=None,distance_threshold=distance_threshold)
    reducedDimData = agglo.fit_transform(data)
    return reducedDimData, agglo.n_clusters_

def elbowHeuristic_FA(data, markX=None, markY=None, annotX=None, annotY=None, figPath=None):
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
        agglo = FeatureAgglomeration(n_clusters=None,distance_threshold=i)
        clustering = agglo.fit_transform(data)
        nClusters.append(clustering.shape[1])
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
        '''
        plt.annotate('Elbow point\n'+str((np.around(markX,3), 7)), xy=(markX, markY+0.2),
                     xytext=(0.65, 15), arrowprops=dict(arrowstyle="->",
                     connectionstyle="angle3,angleA=0,angleB=-90"));
        '''
        if annotX is not None and annotY is not None:
            if annotY>markY:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY+0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"),
                             fontsize=21);
            else:
                plt.annotate('Elbow point\n'+str((np.around(markX,3), markY)),
                             xy=(markX, markY-0.2), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"),
                             fontsize=21);
    plt.xlim(0, 1)
    plt.ylim(0, 27)
    plt.xlabel("Distance Threshold")
    plt.ylabel("Reduced Dimension (d')")
    if figPath is not None:
        plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()