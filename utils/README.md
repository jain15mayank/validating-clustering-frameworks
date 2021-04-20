# Utilities repository for Clustering

+ `agglomerative.py`: Contains utility functions to perform agglomerative clusetring, and perform elbow heuristics for agglomerative clustering hyperparameters
+ `cmeans.py`: Contains utility functions to perform fuzzy c-means clusetring, determine optimal value of fuzzifier, and maximize fuzzy partition coefficient to obtain optimal number of clusters
+ `cviMetrics.py`: Contains utility functions to compute inter-cluster distances, cluster diameter, dunn's index, and xie-beni score. Further defines a compute_cvi function to compute the validation score and gives recommendation for the best clustering algorithm based on the defined Cluster Validation Indices (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, Dunn Index, and Xie-Beni Index.
+ `dimReduction.py`: Contains utility functions to perform dimensionality reduction via Principal Component Analysis (PCA) and Fuzzy Agglomeration (FA). Further defines additional methods which perform elbow heuristics to determine optimal number of reduced dimensions in both cases.
+ `kmeans.py`: Contains utility functions to perform k-Means clusetring, and perform Gap statistics for k-means clustering hyperparameters
+ `normalize_columns.py`: Utility functions to normalize columns - primarily tuned for Fuzzy C-Means algorithm functionality only.
+ `objectiveValidation.py`: Contains utility functions to perform the novel objective validation strategy
+ `spectral.py`: Contains utility functions to perform spectral clusetring, and perform Gap statistics for spectral clustering hyperparameters
+ `utils.py`: Contains miscellaneous utility functions like, performing pre-processing steps (extract median and normalize), calculate cluster centroids, and plot generated clusters
