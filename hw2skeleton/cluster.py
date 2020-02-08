from .utils import Atom, Residue, ActiveSite, Cluster, normalize_reps
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import umap
import math
import sys


def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    assert site_a.ld_rep is not None
    assert site_b.ld_rep is not None

    #euclidian distance between low-d vectors
    return math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, 
        zip(site_a.ld_rep, site_b.ld_rep))))


def kpp(active_sites, k):
    """
    use k++ centroid initialization to make clustering more consistent + 
    hopefully avoid really awful clustering
    """

    #1. Choose one center uniformly at random from among the data points.
    centroids = []
    centroids.append(active_sites[random.sample(range(len(active_sites)), 1)[0]])
    while len(centroids) < k:
        #2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        dists = []
        for active_site in active_sites:
            dists.append(min([compute_similarity(active_site, centroid) for centroid in centroids]))
        #3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        #scale weights to 1
        dists = (dists - np.min(dists))/np.ptp(dists)
        new_centroid = random.choices(active_sites, weights=dists, k=1)[0]
        centroids.append(new_centroid)
    #Repeat Steps 2 and 3 until k centers have been chosen.
    return centroids

def kmeans(k, active_sites):
    """
    Do k-means clustering for specific value of k
    """
    #k-means++ cluster initialization
    centroids = kpp(active_sites, k)
    
    #initialize empty clusters
    clusters = [[] for i in range(k)]

    #repeat until converge
    while True:
        #empty new clusters
        new_clusters = [[] for i in range(k)]

        #assign points to clusters based on distante metric
        for active_site in active_sites:
            new_clusters[np.argmin(np.array(
                [compute_similarity(active_site, centroid) for centroid in centroids]))].append(active_site)
        #if clusters converged, you're done
        if new_clusters == clusters:
            break
        #else, calculate new centroids and go again
        clusters = new_clusters

        #get new centroids, mean of current active sites in each cluster
        centroids = [compute_centroid(cluster, active_sites = active_sites) for cluster in clusters]
    return clusters, silhouette_score(clusters)

def compute_centroid(cluster, active_sites = None, name = 'centroid'):
    #take mean at each position
    if cluster:
        all_clusters = np.stack([active_site.ld_rep for active_site in cluster])
        new_centroid = ActiveSite(name)
        new_centroid.ld_rep = np.mean(all_clusters, axis=0)
        return new_centroid
    #catch empty clusters, reinitialize to random
    else:
        return active_sites[random.sample(range(len(active_sites)), 1)[0]]

def repeat_k(k, active_sites, num_repeats = 50):
    """
    Try a few times, this is to account for bad initial clusters
        -still need after different cluster initialization?
    """
    best_clusters = None
    best_sc = float("-inf")
    for _ in range(num_repeats):
        clusters, sc = kmeans(k, active_sites)
        if sc > best_sc:
            best_sc = sc
            best_clusters = clusters
        return best_clusters, sc


def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    best_clusters = None
    best_sc = float("-inf")
    all_clusters = []
    #perform for a range of km calculate silhouette score, take best
    for k in range(3, min(10, len(active_sites)+1)):
        clusters, sc = repeat_k(k, active_sites)
        all_clusters.append(clusters)
        #print(k, clusters, sc)
        if sc > best_sc:
            best_sc = sc
            best_clusters = clusters
    #compare_clusters(all_clusters)
    return best_clusters, best_sc

def create_matrix(clusters, dist = False):
    #generate distance matrix between all clusters
    df = pd.DataFrame(index = clusters, columns = clusters)
    for cluster_a in clusters:
        for cluster_b in clusters:
            if dist:
                sim = 0
            #distance to itself is nan, this is for taking the min later
            #in reality, distance is 0
            elif cluster_a == cluster_b:
                sim = np.nan
            else:
                #cluster has ld_rep, should work
                sim = compute_similarity(cluster_a, cluster_b)
            df[cluster_a][cluster_b] = sim
    return df

def update_matrix(df, new_cluster, old_a, old_b, dist):
    """
    Remove old clusters from cluster list, add new cluster, calculate new distances
    """

    #drop old clusters
    for i in range(2):
        for old_cluster in old_a, old_b:
            df = df.drop(old_cluster, axis=i)
    #make sure they're actually gone
    assert old_a not in df.index and old_a not in df.columns
    assert old_b not in df.index and old_a not in df.columns

    #add new column/row with distances from all other clusters to new cluster
    #using complete linkage clustering
    data = {}
    for cluster in df.index:
        site_bs = cluster.active_sites
        #get max distance between active sites in this cluster 
        #to active sites in each other cluster
        new_dist = max([compute_similarity(site_a, site_b) 
            for site_a in new_cluster.active_sites
            for site_b in site_bs])
        data[cluster] = new_dist
    data[new_cluster] = np.nan #self dist is NaN for min to work
    new_data = pd.Series(data=data, name=new_cluster)
    df = df.append(new_data) #add row
    df[new_cluster] = new_data #add col - unnecessary?

    assert new_cluster in df.index
    return df

def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    #each point starts as its own unique cluster
    clusters = [Cluster([active_site], active_site.ld_rep)
     for active_site in active_sites]

     #initial distance matrix, each cluster to each other cluster
    df = create_matrix(clusters)
    all_clusters = []
    best_clusters = None
    best_sc = float("-inf")

    #run until all sites are joined
    while df.shape[0] > 1:
        #find two closest in distances
        ri, ci = np.unravel_index(np.nanargmin(df.values), df.shape)

        #create new cluster with average of other active sites
        new_cluster_sites = df.index[ri].active_sites + df.columns[ci].active_sites
        new_cluster = Cluster(new_cluster_sites, compute_centroid(new_cluster_sites).ld_rep)

        #update distance
        df= update_matrix(df, new_cluster, df.index[ri],  
            df.columns[ci], df.iloc[ri][ci])

        #get current clusters
        current_clusters = [cluster.active_sites for cluster in df.index]

        #check silhouette score, return clusters with the best sc
        #Note: I currently limit this to between 9 and 3 clusters so that it aligns
        #with the ks used in k means clustering
        if len(current_clusters) < 10 and len(current_clusters) > 2:
            sc = silhouette_score(current_clusters)
            if sc > best_sc:
                best_clusters = current_clusters
                best_sc = sc
            all_clusters.append(current_clusters)

    return best_clusters, best_sc

def silhouette_score(clusters):
    """
    Determine clustering quality, ie how similar each element is to other elements
    in its cluster vs how similar it is to elements in the nearest other cluster
    """
    #cant calculate for a single cluster
    if len(clusters) == 1:
        return -1
    # the mean s(i) over all data of the entire dataset 
    # is a measure of how appropriately the data have been clustered
    scores = []
    for cluster_a in clusters:
        for i in cluster_a:
            #s(i) for clusters of length 1 = 0
            if len(cluster_a) == 1:
                scores.append(0)
                continue
            #get similarity to other elements in cluster
            a = _point_similarity(i, cluster_a) 

            #get the minimum distance to elements of another cluster
            b = min([_point_similarity(i, cluster_b) for cluster_b in clusters 
                if cluster_a != cluster_b])
           
            #calculate s(i)
            scores.append((b - a)/max(a, b))
    return sum(scores)/len(scores)

def _point_similarity(i, cluster):
    """
    Mean distance between i and all points in cluster
    """
    dists = [compute_similarity(i, site_b) for site_b in cluster if i != site_b]
    return sum(dists)/len(dists)

def compare_clusters(clusters_a, clusters_b, plot = False):
    """
    Get Jaccard index for the two clusterings
    """
    #jaccard index
    #j(C, C') = n_11/(n_11, n_10, n_01)

    #if both are empty, return 1
    if not clusters_a and not clusters_b:
        return 1

    n11, n00, n10, n01 = 0, 0, 0, 0
    #get active sites by unpacking cluster
    sites = [site for cluster in clusters_a for site in cluster]
    for site_a in sites:
        for site_b in sites:
            if site_a != site_b: #site should be in the same cluster as itself, if not something is wrong
                #check if they are in the same cluster in each clustering
                same_a = _same_cluster(site_a, site_b, clusters_a)
                same_b = _same_cluster(site_a, site_b, clusters_b)
                #update n_xs
                #same cluster in both
                if same_a == same_b == True:
                    n11 += 1
                #same in a, not b
                elif same_a and not same_b:
                    n10 += 1
                #same in b, not a
                elif same_b and not same_a:
                    n01 += 1
                #not in the same in either clustering
                else:
                    n00 += 1
    ji = n11/(n11 + n10 + n01)
    if plot:
         plot_clusters([clusters_a, clusters_b], ji)
    return ji


def _same_cluster(site_a, site_b, clusters):
    """
    Check if sites are in the same cluster
    """
    a_index = [site_a in cluster for cluster in clusters].index(True)
    b_index = [site_b in cluster for cluster in clusters].index(True)
    return a_index == b_index


def _plot_cluster(clusters, ax, embeddings, title):    
    for cluster in clusters:
        vals = [embeddings[x] for x in cluster]
        ax.scatter([val[0] for val in vals], [val[1] for val in vals])
    ax.set_title("{1}, nc = {2}, sc = {0:.2f}".format(silhouette_score(clusters), title, len(clusters)))

def plot_clusters(clusterings, ji):
    if len(clusterings) == 1:
        return 
    #list of active sites
    sites = [site for cluster in clusterings[0] for site in cluster]
    
    #get UMAP embedding for visualization
    reducer = umap.UMAP()
    embedding = reducer.fit_transform([site.ld_rep for site in sites])
    
    #assign embedding to site for consistend vis across clusterings
    embeddings = {}
    for site, em in zip(sites, embedding):
        embeddings[site] = em

    fig, axs = plt.subplots(ncols = len(clusterings), figsize=(6*len(clusterings), 5))
    titles = ['k-means', 'agglomerative']
    for i, (clustering, title) in enumerate(zip(clusterings, titles)):
        _plot_cluster(clustering, axs[i], embeddings, title)
    plt.suptitle("Jaccard index: {0:.3f}".format(ji))
    plt.tight_layout()
    plt.savefig('compare_clusters.png')

