import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, compare_clusters
from .utils import normalize_reps

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])
normalize_reps(active_sites)

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering = cluster_by_partitioning(active_sites)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(active_sites)
    write_mult_clusterings(sys.argv[3], clusterings)

if sys.argv[1][0:2] == '-C':
	print("Comparing both clustering methods")
	partition_clusters = cluster_by_partitioning(active_sites)
	hierarchical_clusters = cluster_hierarchically(active_sites)
	compare_clusters([partition_clusters, hierarchical_clusters], compare = True)
