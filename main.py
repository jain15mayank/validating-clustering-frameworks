# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:25:36 2020

@author: Mayank Jain
"""

import numpy as np
import pandas as pd
import datetime
import read_data
import matplotlib.pyplot as plt
import matplotlib.dates as md
from pathlib import Path
from utils.utils import preProcessing_clustering, plotClusters
from utils.dimReduction import dim_reduction_PCA, elbowHeuristic_PCA
from utils.dimReduction import dim_reduction_FA, elbowHeuristic_FA
from utils.spectral import spectralClustering_KM_KNN_Euc, optimalKspectral
from utils.kmeans import kmeans, optimalK
from utils.agglomerative import agglomerativeClustering, elbowHeuristic_Agglo
from utils.cmeans import cmeans, optimal_fuzzifier, optimal_clusters
from utils.spectral import validate_spectral_clusters
from utils.objectiveValidation import validate_clusters
from utils.cviMetrics import compute_cvi

def plot_series(time, series, fmt="-", start=0, end=None):
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    plt.plot_date(time[start:end], series[start:end], fmt)
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=30)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def offset_extra_time(time, series):
    # Offset Operation: Delete extra readings which are outside the scope of a day
    pre_offset = 0
    while True:
        if time[pre_offset].hour != 0:
            pre_offset += 1
        else:
            break
    post_offset = 0
    while True:
        if time[-post_offset-1].hour != 23:
            post_offset += 1
        else:
            break
    if post_offset == 0:
        post_offset = None
    offset_time = time[pre_offset:-post_offset]
    offset_series = series[pre_offset:-post_offset]
    if len(time)==0:
        raise RuntimeError('Error in input series - offset operation failed')
    # Offset operation Successful
    return offset_time, offset_series

def daily_analysis(ori_time, ori_series, readings_per_day=24, plotting=True):
    if plotting:
        plt.figure(figsize=(10, 6))
    time, series = offset_extra_time(ori_time, ori_series)
    tot_days = int(len(time)/readings_per_day)
    net_series = np.zeros(readings_per_day)
    add_count = 0
    day_series = np.zeros((tot_days, readings_per_day))
    for day in range(tot_days):
        day_series[day,:] = series[day*readings_per_day : (day+1)*readings_per_day]
        time_series = time[day*readings_per_day : (day+1)*readings_per_day]
        time_series = [datetime.datetime.combine(datetime.date(2020,1,1), d.time()) for d in time_series]
        if not any(np.isnan(day_series[day,:])):
            net_series += day_series[day,:]
            add_count += 1
            if plotting:
                plt.scatter(time_series, day_series[day,:], c='k', marker=".")
    net_series /= add_count
    if plotting:
        plt.plot(time_series, net_series, 'b-')
        xfmt = md.DateFormatter('%H:%M')
        plt.gca().xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=30)
        plt.xlabel("Time Of The Day")
        plt.ylabel("Renewable Load Consumption (Watts)")
        ax = plt.axes()
        ax.set_xlim(left=datetime.datetime(2020,1,1,0,0), right=datetime.datetime(2020,1,1,23,0))
        plt.grid(True)
    return net_series, add_count, day_series, time_series

def daily_analysis_median(complete_day_series, plotting=True, clusterLabels=None, plotNorm=False, normScale=1000, clusterDir='Clusters'):
    median_series = np.zeros((len(complete_day_series), complete_day_series.shape[-1]))
    for houseIdx in range(len(complete_day_series)):
        temp = []
        for i in range(complete_day_series.shape[-1]):
            temp.append(complete_day_series[houseIdx,:,i][~np.isnan(complete_day_series[houseIdx,:,i])])
            median_series[houseIdx, i] = np.median(temp[i])
        if plotting:
            plt.figure(figsize=(15, 5))
            plt.boxplot(temp, flierprops = dict(marker='.', markerfacecolor='k', markersize=2, linestyle='none', markeredgecolor='k'))
            plt.plot(np.concatenate(([np.nan],median_series[houseIdx, :])), label="Median of Raw Data")
            norm = median_series[houseIdx, :] / (np.linalg.norm(median_series[houseIdx, :]))
            if plotNorm:
                plt.plot(np.concatenate(([np.nan],normScale*norm)), label="Normalized Median (Scaled Up by "+str(normScale)+")")
            locs, labels = plt.xticks()
            start = pd.Timestamp('2020-01-01')
            end = pd.Timestamp('2020-01-02')
            t = np.linspace(start.value, end.value, complete_day_series.shape[-1], endpoint=False)
            t = pd.to_datetime(t)
            newT = []
            for j in range(len(t)):
                newT.append(datetime.datetime.strptime(str(t[j]), '%Y-%m-%d %H:%M:%S').strftime('%H:%M'))
            plt.xticks(locs, newT)
            plt.xticks(rotation=30)
            plt.rc('font', size=21)         # controls default text sizes
            plt.rc('axes', titlesize=22)    # fontsize of the axes title
            plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
            plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
            plt.rc('legend', fontsize=21)   # legend fontsize
            plt.rc('figure', titlesize=22)  # fontsize of the figure title
            plt.xlabel("Time Of The Day")
            plt.ylabel("Load Consumption (Watts)")
            if plotNorm:
                plt.legend()
            if clusterLabels is None:
                Path("./Household Daily Analysis Plots").mkdir(parents=True, exist_ok=True)
                plt.savefig('./Household Daily Analysis Plots/House'+str(houseIdx)+'Median.png', bbox_inches = 'tight', pad_inches = 0.05)
                plt.savefig('./Household Daily Analysis Plots/House'+str(houseIdx)+'Median.pdf', bbox_inches = 'tight', pad_inches = 0.05)
            else:
                Path("./Household Daily Analysis Plots/"+clusterDir).mkdir(parents=True, exist_ok=True)
                plt.savefig('./Household Daily Analysis Plots/'+clusterDir+'/'+str(clusterLabels[houseIdx])+'House'+str(houseIdx)+'Median.png', bbox_inches = 'tight', pad_inches = 0.05)
                plt.savefig('./Household Daily Analysis Plots/'+clusterDir+'/'+str(clusterLabels[houseIdx])+'House'+str(houseIdx)+'Median.pdf', bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
    return median_series

'''#########################################################################'''
# Reading Data
'''#########################################################################'''
(timestamp, house) = read_data.read_data('data/Parent_consumption.csv')

'''#########################################################################'''
# Parsing Raw Data
'''#########################################################################'''

READINGS_PER_DAY = 24
TOTAL_DAYS = int(len(offset_extra_time(timestamp, house[0,:])[0])/READINGS_PER_DAY)
day_avg_consumption = np.zeros((len(house), READINGS_PER_DAY))
num_days_for_avg = np.zeros((len(house)))
complete_day_series = np.zeros((len(house), TOTAL_DAYS, READINGS_PER_DAY))
for i in range(len(house)):
    day_avg_consumption[i,:], num_days_for_avg[i], complete_day_series[i,:,:], time_series = \
        daily_analysis(timestamp, house[i,:], readings_per_day=READINGS_PER_DAY, plotting=False)
#day_median_consumption = daily_analysis_median(complete_day_series, plotting=False)

'''#########################################################################'''
# Pre-Processing and Reducing Dimensions of Parsed Data
'''#########################################################################'''
# Check if directory exists for all optimization plots to save henceforth
Path("./Optimization Plots").mkdir(parents=True, exist_ok=True)
# Pre-processing
processedData = preProcessing_clustering(complete_day_series)
# Dimensionality Reduction by PCA
elbowHeuristic_PCA(processedData, markX=0.9629912972027689, markY=7, annotX=0.65, annotY=15, figPath='./Optimization Plots/ElbowHeuristic_PCA.pdf')
day_median_consumption_reducedPCA, nComponents = dim_reduction_PCA(processedData, 0.96)
# Dimensionality Reduction by FA
elbowHeuristic_FA(processedData, markX=0.45, markY=7, annotX=0.65, annotY=15, figPath='./Optimization Plots/ElbowHeuristic_FA.pdf')
day_median_consumption_reducedFA, nComponentsFA = dim_reduction_FA(processedData, 0.45)

'''#########################################################################'''
# Clustering by PCA and Unnormalized Spectral Clustering
'''#########################################################################'''

kSCPCA = optimalKspectral(day_median_consumption_reducedPCA, nrefs=500, maxClusters=11, plotting=True, figPath='./Optimization Plots/Gap_PCA_Spectral.pdf', randomSeed=13)
print('Optimal Clusters (PCA + Spectral) is: ', kSCPCA)
clusPCA,_,nEigenVectorsPCA = spectralClustering_KM_KNN_Euc(day_median_consumption_reducedPCA, kSCPCA, randomSeed=13)
centersSCPCA = clusPCA.cluster_centers_
labelsSCPCA = clusPCA.labels_
#print(labelsSCPCA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsSCPCA, plotNorm=True, clusterDir='ClustersSCPCA')

'''#########################################################################'''
# Clustering by FA and Unnormalized Spectral Clustering
'''#########################################################################'''

kSCFA = optimalKspectral(day_median_consumption_reducedFA, nrefs=500, maxClusters=11, plotting=True, figPath='./Optimization Plots/Gap_FA_Spectral.pdf', randomSeed=13)
print('Optimal Clusters (FA + Spectral) is: ', kSCFA)
clusFA,_,nEigenVectorsFA = spectralClustering_KM_KNN_Euc(day_median_consumption_reducedFA, kSCFA, randomSeed=13)
centersSCFA = clusFA.cluster_centers_
labelsSCFA = clusFA.labels_
#print(labelsSCFA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsSCFA, plotNorm=True, clusterDir='ClustersSCFA')

'''#########################################################################'''
# Clustering by PCA and K-Means Clustering
'''#########################################################################'''

kPCA = optimalK(day_median_consumption_reducedPCA, nrefs=500, maxClusters=11, plotting=True, figPath='./Optimization Plots/Gap_PCA_K-Means.pdf', randomSeed=13)
print('Optimal Clusters (PCA + K-Means) is: ', kPCA)
centersKMPCA, labelsKMPCA = kmeans(day_median_consumption_reducedPCA, kPCA, randomSeed=13)
#print(labelsKMPCA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsKMPCA, plotNorm=True, clusterDir='ClustersKMPCA')

'''#########################################################################'''
# Clustering by FA and K-Means Clustering
'''#########################################################################'''

kFA = optimalK(day_median_consumption_reducedFA, nrefs=500, maxClusters=11, plotting=True, figPath='./Optimization Plots/Gap_FA_K-Means.pdf', randomSeed=13)
print('Optimal Clusters (FA + K-Means) is: ', kFA)
centersKMFA, labelsKMFA = kmeans(day_median_consumption_reducedFA, kFA, randomSeed=13)
#print(labelsKMFA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsKMFA, plotNorm=True, clusterDir='ClustersKMFA')

'''#########################################################################'''
# Clustering by PCA and Agglomerative Clustering
'''#########################################################################'''

elbowHeuristic_Agglo(day_median_consumption_reducedPCA, markX=0.5, markY=4, annotX=0.7, annotY=12, figPath='./Optimization Plots/ElbowHeuristic_PCA_Agglomerative.pdf')
print('Optimal Clusters (PCA + Agglomerative) is: ', 4)
centersACPCA, labelsACPCA = agglomerativeClustering(day_median_consumption_reducedPCA, distance_threshold=0.5)
#print(labelsACPCA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsACPCA, plotNorm=True, clusterDir='ClustersACPCA')

'''#########################################################################'''
# Clustering by FA and Agglomerative Clustering
'''#########################################################################'''

elbowHeuristic_Agglo(day_median_consumption_reducedFA, markX=0.3, markY=4, annotX=0.7, annotY=12, figPath='./Optimization Plots/ElbowHeuristic_FA_Agglomerative.pdf')
print('Optimal Clusters (FA + Agglomerative) is: ', 4)
centersACFA, labelsACFA = agglomerativeClustering(day_median_consumption_reducedFA, distance_threshold=0.3)
#print(labelsACFA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsACFA, plotNorm=True, clusterDir='ClustersACFA')

'''#########################################################################'''
# Clustering by PCA and Fuzzy C-Means Clustering
'''#########################################################################'''

fuzzifierPCA = optimal_fuzzifier(day_median_consumption_reducedPCA)
nclustersFCMPCA = optimal_clusters(day_median_consumption_reducedPCA, maxClusters=10,
                                   trials=100, fuzzifier = fuzzifierPCA, plotting=True,
                                   figPath='./Optimization Plots/FuzzyPartitionCoefficient_PCA_FCM.pdf')
print('Optimal Clusters (PCA + Fuzzy C-Means) is: ', nclustersFCMPCA)
fcmResultsPCA = cmeans(day_median_consumption_reducedPCA, nclustersFCMPCA, fuzzifierPCA, error=0.005, maxiter=1000)
centersFCMPCA = fcmResultsPCA[0]
fuzzyMemMatPCA = fcmResultsPCA[1]
labelsFCMPCA = np.argmax(fuzzyMemMatPCA, axis=0)
#print(labelsFCMPCA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsFCMPCA, plotNorm=True, clusterDir='ClustersFCMPCA')

'''#########################################################################'''
# Clustering by FA and Fuzzy C-Means Clustering
'''#########################################################################'''

fuzzifierFA = optimal_fuzzifier(day_median_consumption_reducedFA)
nclustersFCMFA = optimal_clusters(day_median_consumption_reducedFA, maxClusters=10,
                                  trials=100, fuzzifier = fuzzifierFA, plotting=True,
                                  figPath='./Optimization Plots/FuzzyPartitionCoefficient_FA_FCM.pdf')
print('Optimal Clusters (FA + Fuzzy C-Means) is: ', nclustersFCMFA)
fcmResultsFA = cmeans(day_median_consumption_reducedFA, nclustersFCMFA, fuzzifierFA, error=0.005, maxiter=1000)
centersFCMFA = fcmResultsFA[0]
fuzzyMemMatFA = fcmResultsFA[1]
labelsFCMFA = np.argmax(fuzzyMemMatFA, axis=0) # Convert Fuzzy Membership Matrix to Hard Labels
#print(labelsFCMFA)
#daily_analysis_median(complete_day_series, plotting=True, clusterLabels=labelsFCMFA, plotNorm=True, clusterDir='ClustersFCMFA')

'''#########################################################################'''
# Objective Validation of Clustering Results
'''#########################################################################'''

for partition in range(2,4):
    print('\nFor partition = ', partition)
    # Validate Spectral Clustering
    objResultsSCPCA = validate_spectral_clusters(centersSCPCA, labelsSCPCA, complete_day_series, nEigenVectorsPCA, partitions=partition, dimRedMethod='PCA')
    print('PCA+SC:', objResultsSCPCA[:-1])
    objResultsSCFA = validate_spectral_clusters(centersSCFA, labelsSCFA, complete_day_series, nEigenVectorsFA, partitions=partition, dimRedMethod='FA')
    print('FA+SC:', objResultsSCFA[:-1])
    # Validate K-Means Clustering
    objResultsKMPCA = validate_clusters(centersKMPCA, labelsKMPCA, complete_day_series, partitions=partition, dimRedMethod='PCA')
    print('PCA+KM:', objResultsKMPCA[:-1])
    objResultsKMFA = validate_clusters(centersKMFA, labelsKMFA, complete_day_series, partitions=partition, dimRedMethod='FA')
    print('FA+KM:', objResultsKMFA[:-1])
    # Validate Agglomerative Clustering
    objResultsACPCA = validate_clusters(centersACPCA, labelsACPCA, complete_day_series, partitions=partition, dimRedMethod='PCA')
    print('PCA+AC:', objResultsACPCA[:-1])
    objResultsACFA = validate_clusters(centersACFA, labelsACFA, complete_day_series, partitions=partition, dimRedMethod='FA')
    print('FA+AC:', objResultsACFA[:-1])
    # Validate Fuzzy C-Means Clustering
    objResultsFCMPCA = validate_clusters(centersFCMPCA, labelsFCMPCA, complete_day_series, partitions=partition, dimRedMethod='PCA')
    print('PCA+FCM:', objResultsFCMPCA[:-1])
    objResultsFCMFA = validate_clusters(centersFCMFA, labelsFCMFA, complete_day_series, partitions=partition, dimRedMethod='FA')
    print('FA+FCM:', objResultsFCMFA[:-1])

'''#########################################################################'''
# Validation of Clustering Results by Cluster Validation Indices (CVIs)
'''#########################################################################'''

print('\nCVIs for clustering followed by PCA: [SC, KM, AC, FCM]')
labelsMat = np.array([labelsSCPCA, labelsKMPCA, labelsACPCA, labelsFCMPCA]).T
SH_PCA = compute_cvi(day_median_consumption_reducedPCA, labelsMat, cvi_metric = 'silhouette')
print('Silhouette Score:\t', SH_PCA[1])
CH_PCA = compute_cvi(day_median_consumption_reducedPCA, labelsMat, cvi_metric = 'CHscore')
print('Calinski-Harabasz Index:', CH_PCA[1])
DB_PCA = compute_cvi(day_median_consumption_reducedPCA, labelsMat, cvi_metric = 'DBscore')
print('Davies-Bouldin Index:\t', DB_PCA[1])
DI_PCA = compute_cvi(day_median_consumption_reducedPCA, labelsMat, cvi_metric = 'DIscore')
print('Dunn Index:\t\t', DI_PCA[1])
XB_PCA = compute_cvi(day_median_consumption_reducedPCA, labelsMat[:,:-1], cvi_metric = 'XBscore',
                     fuzzyMemMat=fuzzyMemMatPCA, fuzzifier=fuzzifierPCA)
print('Xie-Beni Index:\t\t', XB_PCA[1])

print('\nCVIs for clustering followed by FA: [SC, KM, AC, FCM]')
labelsMat = np.array([labelsSCFA, labelsKMFA, labelsACFA, labelsFCMFA]).T
SH_FA = compute_cvi(day_median_consumption_reducedFA, labelsMat, cvi_metric = 'silhouette')
print('Silhouette Score:\t', SH_FA[1])
CH_FA = compute_cvi(day_median_consumption_reducedFA, labelsMat, cvi_metric = 'CHscore')
print('Calinski-Harabasz Index:', CH_FA[1])
DB_FA = compute_cvi(day_median_consumption_reducedFA, labelsMat, cvi_metric = 'DBscore')
print('Davies-Bouldin Index:\t', DB_FA[1])
DI_FA = compute_cvi(day_median_consumption_reducedFA, labelsMat, cvi_metric = 'DIscore')
print('Dunn Index:\t\t', DI_FA[1])
XB_FA = compute_cvi(day_median_consumption_reducedFA, labelsMat[:,:-1], cvi_metric = 'XBscore',
                     fuzzyMemMat=fuzzyMemMatFA, fuzzifier=fuzzifierFA)
print('Xie-Beni Index:\t\t', XB_FA[1])

'''#########################################################################'''
# Saving Clustering Results - for subjective validation
'''#########################################################################'''

# For Spectral Clustering
Path("./Clusters/PCA_SC").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsSCPCA, './Clusters/PCA_SC/Cluster')
Path("./Clusters/FA_SC").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsSCFA, './Clusters/FA_SC/Cluster')
# For K-Means Clustering
Path("./Clusters/PCA_KM").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsKMPCA, './Clusters/PCA_KM/Cluster')
Path("./Clusters/FA_KM").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsKMFA, './Clusters/FA_KM/Cluster')
# For Agglomerative Clustering
Path("./Clusters/PCA_AC").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsACPCA, './Clusters/PCA_AC/Cluster')
Path("./Clusters/FA_AC").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsACFA, './Clusters/FA_AC/Cluster')
# For Fuzzy C-Means Clustering
Path("./Clusters/PCA_FCM").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsFCMPCA, './Clusters/PCA_FCM/Cluster')
Path("./Clusters/FA_FCM").mkdir(parents=True, exist_ok=True)
plotClusters(processedData, labelsFCMFA, './Clusters/FA_FCM/Cluster')

'''#########################################################################'''
# Validation of Clustering Results by Cluster Validation Indices (CVIs) with
# original features of households instead of the dimensionally reduced ones
'''#########################################################################'''

print('\nCVIs for clustering followed by PCA: [SC, KM, AC, FCM]')
labelsMat = np.array([labelsSCPCA, labelsKMPCA, labelsACPCA, labelsFCMPCA]).T
SH_PCA = compute_cvi(processedData, labelsMat, cvi_metric = 'silhouette')
print('Silhouette Score:\t', SH_PCA[1])
CH_PCA = compute_cvi(processedData, labelsMat, cvi_metric = 'CHscore')
print('Calinski-Harabasz Index:', CH_PCA[1])
DB_PCA = compute_cvi(processedData, labelsMat, cvi_metric = 'DBscore')
print('Davies-Bouldin Index:\t', DB_PCA[1])
DI_PCA = compute_cvi(processedData, labelsMat, cvi_metric = 'DIscore')
print('Dunn Index:\t\t', DI_PCA[1])
XB_PCA = compute_cvi(processedData, labelsMat[:,:-1], cvi_metric = 'XBscore',
                     fuzzyMemMat=fuzzyMemMatPCA, fuzzifier=fuzzifierPCA)
print('Xie-Beni Index:\t\t', XB_PCA[1])

print('\nCVIs for clustering followed by FA: [SC, KM, AC, FCM]')
labelsMat = np.array([labelsSCFA, labelsKMFA, labelsACFA, labelsFCMFA]).T
SH_FA = compute_cvi(processedData, labelsMat, cvi_metric = 'silhouette')
print('Silhouette Score:\t', SH_FA[1])
CH_FA = compute_cvi(processedData, labelsMat, cvi_metric = 'CHscore')
print('Calinski-Harabasz Index:', CH_FA[1])
DB_FA = compute_cvi(processedData, labelsMat, cvi_metric = 'DBscore')
print('Davies-Bouldin Index:\t', DB_FA[1])
DI_FA = compute_cvi(processedData, labelsMat, cvi_metric = 'DIscore')
print('Dunn Index:\t\t', DI_FA[1])
XB_FA = compute_cvi(processedData, labelsMat[:,:-1], cvi_metric = 'XBscore',
                     fuzzyMemMat=fuzzyMemMatFA, fuzzifier=fuzzifierFA)
print('Xie-Beni Index:\t\t', XB_FA[1])