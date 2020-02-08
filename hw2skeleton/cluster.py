from .utils import Atom, Residue, ActiveSite, Cluster, normalize_reps
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import umap
import math
import sys

"""
Objective #1: Implement a similarity metric to compare active sites. 
• There is no “right” answer, but you should think hard about choosing something biologically meaningful. 
Objective #2: Implement a partitioning algorithm to cluster the set of active sites. 
Objective #3: Implement a hierarchical algorithm to cluster the set of active sites. 
Objective #4: Implement a function to measure the quality of your clusterings. 
Objective #5: Implement a function to compare your two clusterings 

Please answer the following questions: 
1. Explain your similarity metric, and why it makes sense biologically. 
2. Explain your choice of partitioning algorithm. 
3. Explain your choice of hierarchical algorithm. 
4. Explain your choice of quality metric. How did your clusterings measure up?
5. Explain your function to compare clusterings. How similar were your two clusterings using this function? 
6. Did your clusterings have any biological meaning? 

"""

#TODO: check sihouette score impementation

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """

    #TODO: make sure both site_a and site_b have a low-d rep
    assert site_a.ld_rep is not None
    assert site_b.ld_rep is not None

    #sum of difference of the two representations
    #return sum(list(map(lambda x: abs(x[0] - x[1]), 
    #    zip(site_a.ld_rep, site_b.ld_rep))))

    #euclidian distance between low-d vectors
    return math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, 
        zip(site_a.ld_rep, site_b.ld_rep))))


def kpp(active_sites, k):
    #use k++ cluster initialization to make clustering more consistent

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

    # Fill in your code here!

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

        #check silhouette score, return clusters with the best
        if len(current_clusters) < 10 and len(current_clusters) > 2:
            sc = silhouette_score(current_clusters)
            if sc > best_sc:
                best_clusters = current_clusters
                best_sc = sc
            all_clusters.append(current_clusters)
    #compare_clusters(all_clusters)
    return best_clusters, best_sc

#check implementation

def silhouette_score(clusters):
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
    dists = [compute_similarity(i, site_b) for site_b in cluster if i != site_b]
    return sum(dists)/len(dists)

def compare_clusters(clusters_a, clusters_b, plot = False):
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
    #print(n11, n10, n01, n00)
    ji = n11/(n11 + n10 + n01)
    if plot:
         plot_clusters([clusters_a, clusters_b], ji)
    return ji


def _same_cluster(site_a, site_b, clusters):
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


#not using

"""
def _3d_distance(coords_a, coords_b):
    return math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(coords_a, coords_b))))


def align_centers(ma, mb):

    #align all to current centroid
    shorter = ma
    longer = mb
    if ma.shape[0] > mb.shape[0]:
        shorter = mb
        longer = ma

    #enumerate all overlaps and scores - change?
    #get starting point - loset 
    final_assignments = []
    best_score = float("inf")

    #first - assign random
    a_unassigned = list(range(shorter.shape[0]))
    b_unassigned = list(range(longer.shape[0]))

    final_assignments = None
    best_score = float("inf")

    perms = list(itertools.permutations(b_unassigned, r = shorter.shape[0]))
    all_assignments = []
    for perm in perms:
        assignments = list(zip(a_unassigned, perm))
        score = score_align(assignments, shorter, longer)
        if score < best_score:
            best_score = score
            final_assignments = assignments

    return assignments, best_score


def score_align(assignments, ma, mb):
    #assignments as tuple of current pairs
    a_assigned = [x[0] for x in assignments]
    b_assigned = [x[1] for x in assignments]

    a_edges = list(itertools.combinations(a_assigned, 2))
    b_edges = list(itertools.combinations(b_assigned, 2))

    dists1 = np.array([ma[x[0], x[1]] for x in a_edges])
    dists2 = np.array([mb[x[0], x[1]] for x in b_edges])

    return np.mean(np.absolute((dists1 - dists2)))

def _align_sequences(site_a, site_b):
    #needleman_wunsch to get full alignment
    gp = 1
    n = len(site_a.residues)
    m = len(site_b.residues)

    F = np.zeros(shape=(m+1, n+1), dtype=np.int)

    for i in range(1, m+1):
        F[i][0] = F[i-1][0] + gp

    for j in range(1, n+1):
        F[0][j] = F[0][j-1] + gp

    for i in range(1, m+1):
        for j in range(1, n+1):
            F[i][j] = max(F[i-1][j-1] + _sub_penalty(site_a.residues[j-1], site_b.residues[i-1]), # substitution
                        F[i-1][j] - gp,   #gap
                        F[i][j-1] - gp) #gap
    print(F)

    #traceback
    align1 = []
    align2 = []

    i = m
    j = n

    while i > 0 and j > 0:
        cur, diag, up, left = F[i][j], F[i-1][j-1], F[i][j-1], F[i-1][j]

        if cur == diag + _sub_penalty(site_a.residues[j-1], site_b.residues[i-1]):
            align1.append(site_a.residues[j-1])
            align2.append(site_b.residues[i-1])
            i -= 1
            j -= 1
        elif cur == up - gp:
            align1.append(site_a.residues[j-1])
            align2.append(None)
            j -= 1
        elif cur == left - gp:
            align1.append(None)
            align2.append(site_b.residues[i-1])
            i -= 1

    while j > 0:
        align1.append(site_a.residues[j-1])
        align2.append(None)
        j -= 1
    while i > 0:
        align1.append(None)
        align2.append(site_b.residues[i-1])
        i -= 1

    return align1[::-1], align2[::-1] #reverse

def _sub_penalty(a, b):
    a = tlc[a.type.split()[0]]
    b = tlc[b.type.split()[0]]
    return blosum62[a][b]

tlc = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

blosum62 = {'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0, 'B': -2, 'Z': -1, 'X': 0, '*': -4}, 'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3, 'B': -1, 'Z': 0, 'X': -1, '*': -4}, 'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3, 'B': 3, 'Z': 0, 'X': -1, '*': -4}, 'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3, 'B': 4, 'Z': 1, 'X': -1, '*': -4}, 'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1, 'B': -3, 'Z': -3, 'X': -2, '*': -4}, 'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2, 'B': 0, 'Z': 3, 'X': -1, '*': -4}, 'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 1, 'Z': 4, 'X': -1, '*': -4}, 'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3, 'B': -1, 'Z': -2, 'X': -1, '*': -4}, 'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3, 'B': 0, 'Z': 0, 'X': -1, '*': -4}, 'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3, 'B': -3, 'Z': -3, 'X': -1, '*': -4}, 'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1, 'B': -4, 'Z': -3, 'X': -1, '*': -4}, 'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 0, 'Z': 1, 'X': -1, '*': -4}, 'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1, 'B': -3, 'Z': -1, 'X': -1, '*': -4}, 'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1, 'B': -3, 'Z': -3, 'X': -1, '*': -4}, 'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2, 'B': -2, 'Z': -1, 'X': -2, '*': -4}, 'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2, 'B': 0, 'Z': 0, 'X': 0, '*': -4}, 'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0, 'B': -1, 'Z': -1, 'X': 0, '*': -4}, 'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3, 'B': -4, 'Z': -3, 'X': -2, '*': -4}, 'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1, 'B': -3, 'Z': -2, 'X': -1, '*': -4}, 'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4, 'B': -3, 'Z': -2, 'X': -1, '*': -4}, 'B': {'A': -2, 'R': -1, 'N': 3, 'D': 4, 'C': -3, 'Q': 0, 'E': 1, 'G': -1, 'H': 0, 'I': -3, 'L': -4, 'K': 0, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3, 'B': 4, 'Z': 1, 'X': -1, '*': -4}, 'Z': {'A': -1, 'R': 0, 'N': 0, 'D': 1, 'C': -3, 'Q': 3, 'E': 4, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 1, 'Z': 4, 'X': -1, '*': -4}, 'X': {'A': 0, 'R': -1, 'N': -1, 'D': -1, 'C': -2, 'Q': -1, 'E': -1, 'G': -1, 'H': -1, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -1, 'P': -2, 'S': 0, 'T': 0, 'W': -2, 'Y': -1, 'V': -1, 'B': -1, 'Z': -1, 'X': -1, '*': -4}, '*': {'A': -4, 'R': -4, 'N': -4, 'D': -4, 'C': -4, 'Q': -4, 'E': -4, 'G': -4, 'H': -4, 'I': -4, 'L': -4, 'K': -4, 'M': -4, 'F': -4, 'P': -4, 'S': -4, 'T': -4, 'W': -4, 'Y': -4, 'V': -4, 'B': -4, 'Z': -4, 'X': -4, '*': 1}}
"""


