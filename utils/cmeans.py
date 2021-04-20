"""
cmeans.py : Fuzzy C-means clustering algorithm.

Fetched From: https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/_cmeans.py
Fetched On: 19 June 2020

For first 2 functions:
    @author: Mayank Jain
"""
import numpy as np
from scipy.spatial.distance import cdist
from .normalize_columns import normalize_columns, normalize_power_columns
from sklearn.metrics.pairwise import distance_metrics
from statistics import stdev, mean
import matplotlib.pyplot as plt

def optimal_fuzzifier(data):
    """
    Determines the optimal value of fuzzifier for fuzzy c-means algorithm. This
    method employs the method described in Dembele & Kastner 2003.
    Reference: Dembele, D., & Kastner, P. (2003). Fuzzy C-means method for 
               clustering microarray data. bioinformatics, 19(8), 973-980.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data which needs to be clustered by Fuzzy C-Means
    Returns:
        mOpt: float
    """
    p = data.shape[1] # number of dimensions
    min_m = 1.0001 # Minimum possible value of m
    max_m = 50 # Realistic value corresponding to m->np.inf
    m = 2 # Initialize m for numerical approximation
    delta = 0.0001 # Acceptable error range for coeeficient of variation of Ym
    Ym = []
    for i in range (data.shape[0]):
        for j in range (i+1, data.shape[0]):
            temp = distance_metrics()['euclidean'](data[i].reshape(1,-1),data[j].reshape(1,-1))[0][0]
            Ym.append(pow(temp, 1/(m-1)))
    cv_Ym = stdev(Ym)/mean(Ym)
    while not (cv_Ym>(0.03*p)-delta and cv_Ym<(0.03*p)+delta):
        if cv_Ym<=(0.03*p)-delta:
            max_m = m
            m = (min_m+max_m)/2
        else:
            min_m = m
            m = (min_m+max_m)/2
        Ym = []
        for i in range (data.shape[0]):
            for j in range (i+1, data.shape[0]):
                temp = distance_metrics()['euclidean'](data[i].reshape(1,-1),data[j].reshape(1,-1))[0][0]
                Ym.append(pow(temp, 1/(m-1)))
        cv_Ym = stdev(Ym)/mean(Ym)
    if m>10:
        mOpt = 2
    else:
        mOpt = 1+(m/10)
    return mOpt

def optimal_clusters(data, maxClusters=10, trials=100, fuzzifier = None, plotting=False, figPath=None):
    """
    Determines the optimal number of clusters for fuzzy c-means algorithm. This
    is done by finding the value of fuzzy partition coefficient (FPC) for
    different number of clusters and then returning the number of clusters for
    which the value of FPC is maximum.
    Params:
        data: ndarray of shape (n_samples, n_features)
            - original data which needs to be clustered by Fuzzy C-Means
        maxClusters: int (>=2)
            - The maximum number of clusters for which the FPC needs to be
              evaluated
        trials: int (>=1)
            - The number of times for which the experiment needs to be repeated
              in order to diminish the effects of random number initialization
        fuzzifier: float (>1) or None
            - The value of fuzzifier to be used. If None, the optimal value is
              determined automatically using the method described in Dembele &
              Kastner 2003.
        plotting: boolean
            - if True, will plot the FPC curve w.r.t. different number of clusters
        figPath: string
            - path where figure needs to be stored
            - if None, figure will be plotted in console instead
    Returns:
        clusOpt: int
    """
    if fuzzifier == None:
        # Find the optimal value of fuzzifier
        fuzzifier = optimal_fuzzifier(data)
    # Perform analysis for different number of clusters
    fcmResults = []
    fpcVals = [] # Variable to store FPC values
    for nClusters in range(2, maxClusters+1):
        temp1 = list(cmeans(data, nClusters, fuzzifier, error=0.005, maxiter=1000))
        temp1 = [temp1[0], temp1[1], temp1[2], temp1[3], temp1[6]]
        # Perform the experiment for the specified number of trials to diminish
        # the effect of random initialization
        for i in range(trials-1):
            temp2 = list(cmeans(data, nClusters, fuzzifier, error=0.005, maxiter=1000))
            temp2 = [temp2[0], temp2[1], temp2[2], temp2[3], temp2[6]]
            temp1 = [(((i+1)*x[0])+x[1])/(i+2) for x in zip(temp1, temp2)]
        fcmResults.append(temp1)
        fpcVals.append(fcmResults[-1][-1])
    if plotting:
        plt.rc('font', size=21)         # controls default text sizes
        plt.rc('axes', titlesize=22)    # fontsize of the axes title
        plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
        plt.rc('legend', fontsize=21)   # legend fontsize
        plt.rc('figure', titlesize=22)  # fontsize of the figure title
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(2, 11).tolist(), fpcVals)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Fuzzy Partition Coefficient")
        low = min(fpcVals)
        high = max(fpcVals)
        plt.ylim([max(0,low-0.8*(high-low)), min(high+0.4*high, high+0.4*(high-low))])
        if figPath is not None:
            plt.savefig(figPath, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
    clusOpt = np.argmax(fpcVals)+2
    return clusOpt

def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters
    ----------
    data : 2d array, size (N, S)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    cntr : 2d array, size (c, S)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (c, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (c, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (c, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.


    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.

    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Transpose data so that the code can be used alongside the main code seamlessly
    data = data.T
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


def cmeans_predict(test_data, cntr_trained, m, error, maxiter, metric='euclidean', init=None,
                   seed=None):
    """
    Prediction of new data in given a trained fuzzy c-means framework [1].

    Parameters
    ----------
    test_data : 2d array, size (S, N)
        New, independent data set to be predicted based on trained c-means
        from ``cmeans``. N is the number of data sets; S is the number of
        features within each sample vector.
    cntr_trained : 2d array, size (S, c)
        Location of trained centers from prior training c-means.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.

    Notes
    -----
    Ross et al. [1]_ did not include a prediction algorithm to go along with
    fuzzy c-means. This prediction algorithm works by repeating the clustering
    with fixed centers, then efficiently finds the fuzzy membership at all
    points.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    """
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = _cmeans_predict0(test_data, cntr_trained, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc


def _cmeans_predict0(test_data, cntr, u_old, c, m, metric):
    """
    Single step in fuzzy c-means prediction algorithm. Clustering algorithm
    modified from Ross, Fuzzy Logic w/Engineering Applications (2010)
    p.352-353, equations 10.28 - 10.35, but this method to generate fuzzy
    predictions was independently derived by Josh Warner.

    Parameters inherited from cmeans()

    Very similar to initial clustering, except `cntr` is not updated, thus
    the new test data are forced into known (trained) clusters.
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d