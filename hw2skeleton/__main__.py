import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, compare_clusters
from .utils import normalize_reps

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H| -C] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])
normalize_reps(active_sites)

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering, _ = cluster_by_partitioning(active_sites)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clustering, _ = cluster_hierarchically(active_sites)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-C':
    print("Comparing both clustering methods")
    partition_clusters, partition_sc = cluster_by_partitioning(active_sites)
    hierarchical_clusters, hierarchical_sc = cluster_hierarchically(active_sites)
    jaccard = compare_clusters(partition_clusters, hierarchical_clusters, plot = True)

    #write to file
    write_mult_clusterings(sys.argv[3], [partition_clusters, hierarchical_clusters], 
        ["Partitioning", "Hierarchical"], [partition_sc, hierarchical_sc], jaccard)

    #print results
    print("Partitioning: {0}, SC: {1:3f}".format(len(partition_clusters), partition_sc)) 
    print("Hierarchical: {0}, SC: {1:3f}".format(len(hierarchical_clusters), hierarchical_sc))
    print("Cluster similarity: {0:3f}".format(jaccard))