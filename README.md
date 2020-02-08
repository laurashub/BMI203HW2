# HW2 Skeleton

[![Build
Status](https://travis-ci.org/laurashub/BMI203HW2.svg?branch=master)](https://travis-ci.org/laurashub/BMI203HW2)

Skeleton for clustering project.

## assignment

1. Implement a similarity metric
2. Implement a clustering method based on a partitioning algorithm
3. Implement a clustering method based on a hierarchical algorithm
4. Answer the questions given in the homework assignment


## structure

The main file that you will need to modify is `cluster.py` and the corresponding `test_cluster.py`. `utils.py` contains helpful classes that you can use to represent Active Sites. `io.py` contains some reading and writing files for interacting with PDB files and writing out cluster info.

```
.
├── README.md
├── data
│   ...
├── hw2skeleton
│   ├── __init__.py
│   ├── __main__.py
│   ├── cluster.py
│   ├── io.py
│   └── utils.py
└── test
    ├── test_cluster.py
    └── test_io.py
```

### added/modified functions
* __main__.py
  * added -C command that compares the two clusterings
* cluster.py
  * compute_similarity - return distance between two active sites
  * kpp - initialize centroids for k-means using k-means++ algorithm
  * k-means - perform k-means clustering on active sites for a given value of k
  * compute_centroid - get the new center of a cluster
  * repeat_k - call kmeans with same value of k num_repeats times
  * cluster_by_partitioning - perform partition clustering on active sites
  * create_matrix - calculate initial distance matrix for hierarchical clustering
  * update matrix - add new cluster and distances to distance matrix
  * cluster_hierarchically - perform hierarchical clustering on active sites
  * silhouette_score - determine quality of clustering
  * point_similarity - average distance between a given site and elements of a given cluster
  * compare_clusters - calculate Jaccard index between clusterings, maybe run UMAP
  * _same_cluster - determine if two sites are in the same cluster
  * _plot_cluster - plot a cluster based on UMAP embedding
  * plot_clusters - plot all clusters in two different clusterings  
* io.py
  * read_active_sites - changed to unpack results of read_active_site
  * read_active_site - changed to return list of active sites after splitting
  * split_chains - splits up active site by chains, combines identical chains
  * write_mult_clustering - added printing cluster names, Jaccard index
* utils.py
  * Cluster - define cluster to hold a list of active sites
  * normalize_reps - normalize low-d rep and scale certain indeces
  * get_low_d_rep - create array with aa count and avg inter-residue distance
  * res_center - calculate center of residues
  * distance_matrix - create matrix of distances between residues in an active site
  * _3d_distance - 3d euclidian distance
* test_cluster.py
  * all tests for similarity score, clustering, helper functions


## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `hw2skeleton/__main__.py`) can be run as
follows

```
python -m hw2skeleton -P data test.txt
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.


## contributors

Original design by Scott Pegg. Refactored and updated by Tamas Nagy.
