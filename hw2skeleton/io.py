import glob
import os
from .utils import Atom, Residue, ActiveSite, distance_matrix, res_center, get_low_d_rep
import sys
import numpy as np

def read_active_sites(dir):
    """
    Read in all of the active sites from the given directory.

    Input: directory
    Output: list of ActiveSite instances
    """
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):
        #change to accomodate splitting chains
        active_sites += [active_site for active_site in read_active_site(filepath)]

    print("Read in %d active sites"%len(active_sites))

    return active_sites


def read_active_site(filepath):
    """
    Read in a single active site given a PDB file

    Input: PDB file path
    Output: List of active sites in PDB
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] != ".pdb":
        raise IOError("%s is not a PDB file"%filepath)

    active_site = ActiveSite(name[0])

    r_num = 0
    chains = set()
    # open pdb file
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            if line[0:3] != 'TER':
                # read in an atom
                atom_type = line[13:17].strip()
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                atom = Atom(atom_type)
                atom.coords = (x_coord, y_coord, z_coord)

                residue_type = line[17:20]
                residue_number = int(line[23:26])

                #get chain, if theres no chain, label it X
                chain = line[21]
                if chain == " ":
                    chain = 'X'
                
                chains.add(chain)

                # make a new residue if needed
                if residue_number != r_num:
                    residue = Residue(residue_type, residue_number, chain)
                    r_num = residue_number

                # add the atom to the residue
                residue.atoms.append(atom)

            else:  # I've reached a TER card
                #calculate center of residue
                residue.center = res_center(residue.atoms)
                active_site.residues.append(residue)

    #split current active site based on the different chains present
    active_sites = split_chains(chains, active_site)

    #get the lowd rep here so we dont have to calculate it every time
    for active_site in active_sites:
        active_site.ld_rep = get_low_d_rep(active_site)
    return active_sites


def split_chains(chains, active_site):
    """
    Reads and splits chains within the active site, removes identical chains
    """
    new_sites = []
    for chain in sorted(chains):
        chain_reses = [res for res in active_site.residues if res.chain == chain]
        if not new_sites: #automatically add first chain
            new_site = ActiveSite(active_site.name + '_' + chain)
            new_site.residues = chain_reses
            new_sites.append(new_site)
        else:
            for site in new_sites:
                if (sorted([chain_res.type for chain_res in chain_reses])
                    == sorted([res.type for res in site.residues])):
                    pass  #already added this sequence, ignore
                else:
                    #new sequence, create new active site
                    new_site = ActiveSite(active_site.name + '_' + chain)
                    new_site.residues = chain_reses
                    new_sites.append(new_site)
    return new_sites

def write_clustering(filename, clusters):
    """
    Write the clustered ActiveSite instances out to a file.

    Input: a filename and a clustering of ActiveSite instances
    Output: none
    """

    out = open(filename, 'w')

    for i in range(len(clusters)):
        out.write("\nCluster %d\n--------------\n" % i)
        for j in range(len(clusters[i])):
            out.write("%s\n" % clusters[i][j])

    out.close()


def write_mult_clusterings(filename, clusterings, clustering_names, 
    clustering_scs, jaccard):
    """
    Write a series of clusterings of ActiveSite instances out to a file.

    Input: a filename and a list of clusterings of ActiveSite instances, names 
        of clustering algorithms, silhouette scores, jaccard index between the clusterings
    Output: none
    """
    out = open(filename, 'w')

    for i, (clusters, name, sc) in enumerate(zip(clusterings, clustering_names, clustering_scs)):
        out.write("Clustering algorithm: %s, SC: %f\n" % (name, sc))

        for j in range(len(clusters)):
            out.write("\nCluster %d\n------------\n" % j)
            for k in range(len(clusters[j])):
                out.write("%s\n" % clusters[j][k])
        out.write("\n================\n")
    out.write("Jaccard index: {0:.2f}".format(jaccard))
    out.close()
