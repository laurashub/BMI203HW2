from hw2skeleton import cluster
from hw2skeleton import io
#from hw2skeleton import utils

import os
import numpy as np
import glob

##similarty metric tests
def test_similarity_identity():
    site_a = cluster.ActiveSite("A")
    site_a.ld_rep = np.zeros(21)

    assert cluster.compute_similarity(site_a, site_a) == 0.0

def test_similarity_nonnegative():
    #check that all are non-negative
    dir = os.getcwd()
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):
        active_sites += [active_site for active_site in io.read_active_site(filepath)]

    inter_dists = cluster.create_matrix(active_sites)
    assert all(x >= 0 for x in inter_dists.values if not np.isnan(x))

def test_similarity_symmetric():
    active_sites = _all_active_sites()
    inter_dists = cluster.create_matrix(active_sites)
    for site_a in active_sites:
        for site_b in active_sites:
            assert(cluster.compute_similarity(site_a, site_b) 
                == cluster.compare_similarity(site_b, site_a))

def test_similarity_triangle():
    #should hold, euclidian distance
    dir = os.getcwd()
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):
        active_sites += [active_site for active_site in io.read_active_site(filepath)]

    inter_dists = cluster.create_matrix(active_sites)
    for site_a in active_sites:
        for site_b in active_sites:
            dist1 = inter_dists.loc[site_a][site_b]
            for site_c in active_sites:
                dist2 = inter_dists.loc[site_a][site_c]
                dist3 = inter_dists.loc[site_b][site_c]

                if all(not np.isnan(x) for x in [dist1, dist2, dist3]):
                    #get hypotenuse
                    hyp, side1, side2 = _get_hyp(dist1, dist2, dist3)
                    assert hyp <= side1 + side2

def _get_hyp(a, b, c):
    if max(a, b, c) == a:
        return (a, b, c)
    elif max(a, b, c) == b:
        return (b, a, c)
    else:
        return(c, a, b)

###clustering tests
def test_partition_clustering():
    # tractable subset

    #17622 and 19267 chain As are super similar, should be together
    #26095, 34047, 20326 chain Cs are close, should be together
    pdb_ids = [19267, 17622, 26095, 34047, 20326]


    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites += [active_site for active_site in io.read_active_site(filepath)]
    cluster1 = active_sites[:2] #A chains of 19267, 17622
    cluster2 = active_sites[-3:] # C chains of  26095, 34047, 20326
    # will this fail sometimes based on cluster initialization?
    partition_clusters = cluster.cluster_by_partitioning(active_sites)
    assert cluster1 in partition_clusters
    assert cluster2 in partition_clusters

def test_hierarchical_clustering():
    #17622 and 19267 chain As are super similar, should be together
    #26095, 34047, 20326 chain Cs are close, should be together
    pdb_ids = [19267, 17622, 26095, 34047, 20326]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites += [active_site for active_site in io.read_active_site(filepath)]
    

    cluster1 = active_sites[:2] #A chains of 19267, 17622
    cluster2 = active_sites[-3:] # C chains of  26095, 34047, 20326
    # should be consistent, agglom clustering is determininstic
    hierarchichal_clusters = cluster.cluster_by_partitioning(active_sites)
    assert cluster1 in hierarchichal_clusters
    assert cluster2 in hierarchichal_clusters


###silhouette score
def test_silhouette_score():
    site_a = cluster.ActiveSite("A")
    site_a.ld_rep = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 4])

    site_b = cluster.ActiveSite("B")
    site_b.ld_rep = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 5])

    site_c = cluster.ActiveSite("C")
    site_c.ld_rep = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 4])

    site_d = cluster.ActiveSite("D")
    site_d.ld_rep = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 5])

    #more similar clusters together
    assert (cluster.silhouette_score([[site_a, site_b], [site_c, site_d]]) > 
        cluster.silhouette_score([[site_a, site_c], [site_b, site_d]]))

#### tests for helper/utility functions
def test_res_center():
    atom_coords = ((0, 0, 0), (0, 0, 0))
    atom1 = cluster.Atom("CA")
    atom1.coords = (0, 0, 0)

    atom2 = cluster.Atom("CA")
    atom2.coords = (1, 1, 1)

    residue = cluster.Residue("Asp", 0, "A")
    for atom in [atom1, atom2]:
        residue.atoms.append(atom)

    assert io.res_center(residue.atoms) == (0.5, 0.5, 0.5)

    atom3 = cluster.Atom("CA")
    atom3.coords = (2, 2, 2)
    residue.atoms.append(atom3)

    assert io.res_center(residue.atoms) == (1, 1, 1)

def test_split_chains():
    pdb_ids = [34088, 39299]


    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites += [active_site for active_site in io.read_active_site(filepath)]
    assert len(active_sites) == 3 #split 34088, combine 39299 

def _all_active_sites():
    dir = os.getcwd()
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):
        active_sites += [active_site for active_site in io.read_active_site(filepath)]
    return active_sites




