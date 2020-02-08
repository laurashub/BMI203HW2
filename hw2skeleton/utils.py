# Some utility classes to represent a PDB structure
import numpy as np
import math

class Atom:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type):
        self.type = type
        self.coords = (0.0, 0.0, 0.0)

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.type

class Residue:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type, number, chain):
        self.type = type
        self.number = number
        self.chain = chain
        self.atoms = []
        self.center = (0.0, 0.0, 0.0)

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return "{0} {1}".format(self.type, self.number)

class ActiveSite:
    """
    A simple class for an active site
    """

    def __init__(self, name):
        self.name = name
        self.residues = []
        self.ld_rep = None
        self.inter_dists = None

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.name

class Cluster:
    """
    Holds information about a cluster, used in hierarchical clustering
    """
    def __init__(self, active_sites, ld_rep):
        self.active_sites = active_sites
        self.ld_rep  = ld_rep
        self.name = ", ".join([site.name for site in active_sites])

    def __repr__(self):
        return self.name

def normalize_reps(active_sites):
    """
    Normalize and scale the elements of the representations 
    Could be nicer-looking probably

    """
    ld_reps = [active_site.ld_rep for active_site in active_sites]
    stacked = np.stack(ld_reps)

    new_vals = []
    np.seterr(divide='ignore', invalid='ignore') #so we can divide by 0 -> NaN
    for i in range(stacked.shape[1]):
        a = stacked[:,i]
        normalized = (a - np.min(a))/np.ptp(a) #scale
        if any(np.isnan(x) for x in normalized): #replace 0
            normalized = np.zeros(stacked.shape[0])
        new_vals.append(normalized)

    #determine distance scale, avg of sum of amino acid rep
    dist_scale = sum([np.mean(new_vals[i]) for i in range(len(new_vals)-1)])
    
    scale = np.ones(21)
    #weight distance so its considered before aa comp, 
    #gives nicer looking clusters  
    scale[20] = dist_scale * 3

    for i, active_site in enumerate(active_sites):
        new_ld = np.array([scale[j]*new_vals[j][i] for j in range(stacked.shape[1])])
        active_site.ld_rep = new_ld

def get_low_d_rep(active_site):
    """
    Low-d representation is a 21-element vector
    First 20 elements are the count of each amino acid present in the active site
    Last element is the average distance between the center of each residue
    """
    aas = [sum(res.type == AA for res in active_site.residues) for AA in AAs]
    aas.append(np.nanmean(distance_matrix(active_site.residues).flatten()))
    return(np.array(aas))

def res_center(atoms):
    """
    get center of residue
    """
    return(tuple(map(lambda x: sum(x)/len(atoms), list(zip(*[atom.coords for atom in atoms])))))

def distance_matrix(residues):
    """
    Calculate distance between center of each residue 
    """
    centers = [res.center for res in residues]

    #populate array with distances
    dists = np.zeros(shape = (len(centers), len(centers)))
    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            dists[i][j] = _3d_distance(c1, c2)
    dists = np.tril(dists) #only count distances once
    dists[dists == 0] = np.nan #ignore 0s
    return dists

def _3d_distance(coords_a, coords_b):
    """
    3d euclidean distance
    """
    return math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(coords_a, coords_b))))

AAs = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL']

