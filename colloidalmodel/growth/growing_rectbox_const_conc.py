import lammps
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import ast
from collections import defaultdict, deque
import random
import signal
import networkx as nx

import write_lammps_more_monomers_3_types_vary_bond_stiffness
import write_lammps_more_monomers_3_types_vary_bond_stiffness_switch

"""
To be able to run this script and use fix/adapt with cosine squared potential add the following to src/EXTRA-PAIR/pair_cosine_squared.cpp:
    
void *PairCosineSquared::extract(const char *str, int &dim)
{
dim = 2;
if (strcmp(str, "eps") == 0) return (void *) epsilon;
if (strcmp(str, "sigma") == 0) return (void *) sigma;
return nullptr;
}

and the following to src/EXTRA-PAIR/pair_cosine_squared.h of your LAMMPS folder:

void *extract(const char *, int &) override;

Additionally, comment out the rotation in fix_deposit.cpp in the following way(excerpt from fix_deposit.cpp):

    // coords = coords of all atoms
    // for molecule, perform random rotation around center pt
    // apply PBC so final coords are inside box
    // also modify image flags due to PBC

    if (mode == ATOM) { 
      natom = 1;
      coords[0][0] = coord[0];
      coords[0][1] = coord[1];
      coords[0][2] = coord[2];
      imageflags[0] = ((imageint) IMGMAX << IMG2BITS) |
        ((imageint) IMGMAX << IMGBITS) | IMGMAX;
    } else {
      double rng = random->uniform();
      imol = 0;
      while (rng > molfrac[imol]) imol++;
      natom = onemols[imol]->natoms;
      /*
      if (dimension == 3) {
        if (orientflag) {
          r[0] = rx;
          r[1] = ry;
          r[2] = rz;
        } else {
          r[0] = random->uniform() - 0.5;
          r[1] = random->uniform() - 0.5;
          r[2] = random->uniform() - 0.5;
        }
      } else {
        r[0] = r[1] = 0.0;
        r[2] = 1.0;
      }
      double theta = random->uniform() * MY_2PI;
      MathExtra::norm3(r);
      MathExtra::axisangle_to_quat(r,theta,quat);
      MathExtra::quat_to_mat(quat,rotmat);
      for (i = 0; i < natom; i++) {
        MathExtra::matvec(rotmat,onemols[imol]->dx[i],coords[i]);
        coords[i][0] += coord[0];
        coords[i][1] += coord[1];
        coords[i][2] += coord[2];
      */
      
      // No rotation here, just apply translation directly
      for (i = 0; i < natom; i++) {
        coords[i][0] = onemols[imol]->dx[i][0] + coord[0];
        coords[i][1] = onemols[imol]->dx[i][1] + coord[1]; //dont change y axis coords 
        coords[i][2] = onemols[imol]->dx[i][2] + coord[2];

        imageflags[i] = ((imageint) IMGMAX << IMG2BITS) |
          ((imageint) IMGMAX << IMGBITS) | IMGMAX;
        domain->remap(coords[i],imageflags[i]);
      }
    }

Build LAMMPS with additional packages: EXTRA-PAIR and molecular
"""

def read_individual_txt_file(file_path): #done, fine
    
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Reads all lines into a list
        #print(lines)
        
    individual = lines[1]
    random_number_for_type = lines[3]
    #print(random_number_for_type)
    
    individual = ast.literal_eval(individual)
    random_number_for_type = float(ast.literal_eval(random_number_for_type))
    
    return individual, random_number_for_type

def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds, which_simulation, insertion=False):
    xlo, xhi, ylo, yhi = box_bounds
    side_interacting = 'unknown'
    for atom1 in molecule1:
        for atom2 in molecule2:
            atom_id1, x1, y1, atom_type1 = atom1[0], atom1[1], atom1[2], atom1[3]
            atom_id2, x2, y2, atom_type2 = atom2[0], atom2[1], atom2[2], atom2[3]
            
            dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
            dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < cutoff :
                if which_simulation == 'GA':
                    if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1):
                        return True, side_interacting
                elif which_simulation == 'littleguys':
                    if insertion==False:
                        if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6): #make sure to have full binding
                            if atom_type1 in [1,6] and atom_type2 in [2,7]:
                                side_interacting = 'front' ##FRONT AND BACK ARE FLIPPED!
                            elif atom_type1 in [2,7] and atom_type2 in [1,6]:
                                side_interacting = 'back'
                            return True,side_interacting
                    elif insertion==True:
                        if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1):
                            for atom1 in molecule1:
                                for atom2 in molecule2:
                                    atom_id1, x1, y1, atom_type1 = atom1[0], atom1[1], atom1[2], atom1[3]
                                    atom_id2, x2, y2, atom_type2 = atom2[0], atom2[1], atom2[2], atom2[3]
                                    
                                    dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
                                    dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
                                    distance = np.sqrt(dx**2 + dy**2)
                                    
                                    if distance < cutoff:
                                        if (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6):
                                            if atom_type1 in [1,6] and atom_type2 in [2,7]:
                                                side_interacting = 'front'
                                            elif atom_type1 in [2,7] and atom_type2 in [1,6]:
                                                side_interacting = 'back'
                                            return True,side_interacting
                                        
    return False,side_interacting

def analyze_bonds(bonds,molecules):
    """
    Build a molecular connectivity graph from bonds.

    Parameters
    ----------
    bonds : iterable of tuple
        Each tuple contains two molecule/node IDs representing a bond.
        Example: [(0, 1), (1, 2), (3, 4)]

    Returns
    -------
    G : nx.Graph
        Graph containing all molecules and bonds.

    clusters : list[set]
        Connected components of the graph.

    edge_molecules : list[list]
        Degree-1 molecules in each cluster.
    """

    G = nx.Graph()
    G.add_edges_from(bonds)

    # Find disconnected clusters
    clusters = list(nx.connected_components(G))

    # Find degree-1 ("edge") molecules in each cluster
    edge_molecules = [
        [node for node in cluster if G.degree[node] == 1]
        for cluster in clusters
    ]
    
    if len(bonds)==0:
        added_mols = []
        for molecule in molecules:
            if molecule not in added_mols:
                clusters.append([molecule])
                added_mols.append(molecule)

    return G, clusters, edge_molecules

def get_fullinterface_clusters(lmp, which_simulation='littleguys'):
    
    L= lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    pos = lmp.numpy.extract_atom("x")
    #mol = lmp.numpy.extract_atom("molecule")
    mol = (lmp.numpy.extract_atom("id")-1)//8
    atom_ids = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo  #Lower bounds
    xhi, yhi, zhi = boxhi  # Upper bounds
    num_of_molecules= max(mol)
    cutoff = 1.2
    
    clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict  = graph_new(pos, mol, types,atom_ids,cutoff, num_of_molecules,xlo,xhi,ylo,yhi,which_simulation, insertion = True)
    G, clusters_graph, edge_molecules =analyze_bonds(bonds,mol)
    
    return G, clusters_graph, edge_molecules, side_interacting_dict

def get_most_left_and_right_com(lmp,clusters):
    #gets positions of most left and right monomers in polymer for insertion of new monomers
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    
    # create mask to track only molecules in the polymer - not free monomers
    mask = np.array([any((x in subgroup) and (len(subgroup) > 1) for subgroup in clusters)for x in mol])
    pos = pos[mask]
    mol = mol[mask]

    max_x_index = np.argmax(pos[:, 0])  # Index of atom with max x
    min_x_index = np.argmin(pos[:, 0])  # Index of atom with min x
    max_mol_id = mol[max_x_index]       # molecule to which this atom belongs to for max x
    min_mol_id = mol[min_x_index]       # molecule to which this atom belongs to for min x

    # Boolean masks for atoms in the selected molecule
    max_mol_mask = (mol == max_mol_id)
    min_mol_mask = (mol == min_mol_id)

    # extract x-positions and masses of the atoms in that molecule
    max_mol_x = pos[max_mol_mask, 0]  
    min_mol_x = pos[min_mol_mask, 0]

    # center of mass of the molecule with max index?
    com_x_max_mol = np.mean(max_mol_x)
    # center of mass of the molecule with min index?
    com_x_min_mol = np.mean(min_mol_x)

    return com_x_max_mol,com_x_min_mol

def find_clusters(particles, bonds):
    
    # Create an adjacency list
    adjacency_list = defaultdict(list)
    for (a, b) in bonds:
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)

    # Function to perform BFS and find all nodes in the same cluster
    def bfs(start, visited):
        cluster = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            cluster.append(node)
            for neighbor in adjacency_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return cluster
    
    # Find all clusters
    visited = set()
    clusters = []
    
    for particle in particles:
        if particle not in visited:
            cluster = bfs(particle, visited)
            clusters.append(cluster)
    
    return clusters

def get_bonds(positions, molecules,types,atom_ids,cutoff_distance,xlo,xhi,ylo,yhi, which_simulation, insertion = False):
    box_bounds = (xlo,xhi,ylo,yhi)

    atoms_data = np.column_stack((atom_ids, molecules, types))
    atoms_data = np.column_stack((atoms_data, positions))
    #print(atoms_data)
    #print(atoms_data.shape)
    
    molecules = {}
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y, z = atom
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append((atom_id, x, y,atom_type))
        
    # Find touching molecules and add bonds
    bonds = []
    bonded_molecules = []
    side_interacting_dict = {}
        
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                are_touching, side = are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds,which_simulation,insertion) #if theyre touching move
                if are_touching:
                    bonds.append((mol_id1, mol_id2))
                    if mol_id1 not in bonded_molecules:
                        bonded_molecules.append(mol_id1)
                    if mol_id2 not in bonded_molecules:
                        bonded_molecules.append(mol_id2)
                    side_interacting_dict[(mol_id1, mol_id2)] = side
                    
    #print('BONDS BETWEEN:')
    #print(len(bonds))
    #print(bonds)
    return bonds,atoms_data,box_bounds, bonded_molecules,molecules, side_interacting_dict

def setup_simulation_littleguys(lmp, spath, seed, dump_steps, eps):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary f f p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/configuration.dat",
        f"molecule monomer_template {spath}/monomer.dat",
        "group moving id > 0",
        "group none empty",
        "group mol1and2 molecule 1 2",
        f"variable seed equal {seed}",
        f"variable dump_steps equal {dump_steps}",
        f"variable eps equal {eps}",
        "variable eps_C equal 3", # was opposite - 3/6
        "variable eps_T equal 6",
        "variable icoff equal 1.41421356237",
        "variable Kbond1 equal 1000.000000",
        #"variable Kbond2 equal 10000.000000",
        #"variable Kbond3 equal 100.000000",
        "variable ecoff equal 1.200000",
        "variable bcoff equal 1.200000",
        "variable rdist equal 1.000000",
        "variable tstep equal 0.10000",
        "special_bonds lj 0.0 1.0 1.0",
        "bond_style harmonic",
        "bond_coeff 1 ${Kbond1} 1.0",
        #"bond_coeff 2 ${Kbond2} 1.0",
        #"bond_coeff 3 ${Kbond3} 1.0",
        "pair_style hybrid/overlay zero 2.00 cosine/squared 2.00",
        "pair_coeff * * cosine/squared 0.00 1.00 1.10",
        "pair_coeff * * zero 2.00",
        "pair_coeff * * cosine/squared 1.00 1.00 1.00 wca",
        #LITTLE GUYS SETUP - specific binding sites
        "pair_coeff 2 7 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        "pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca",
        "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        "pair_coeff 6 7 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #LITTLE GUYS SETUP - extra bonds
        # "pair_coeff 3 4 cosine/squared ${eps} 1.00 ${bcoff} wca",
        # "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        # "pair_coeff 6 7 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #LITTLE GUYS SETUP - unspecific binding sites
        #"pair_coeff 2 2 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        #"pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca",
        #"pair_coeff 3 1 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #"pair_coeff 4 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #PREVIOUS GA SETUP
        #"pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #"pair_coeff 4 4 cosine/squared 2 1.00 ${bcoff} wca",
        #"pair_coeff 5 5 cosine/squared 5 1.00 ${bcoff} wca",
        #"compute cPE all pe/atom pair",
        #f"fix boundary all wall/reflect xlo {low} xhi {high} ylo {low} yhi {high}",
        #"fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        "fix xwall_low all wall/reflect xlo EDGE",
        "fix xwall_high all wall/reflect xhi EDGE",
        "fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
        #"minimize 1.0e-4 1.0e-6 100 1000",
        "fix fLang moving langevin 1.0 1.0 1.0 ${seed}",
        "fix fNVE moving nve",
        f"dump com_dump all custom {dump_steps} {spath}/output_com.xyz id mol type x y",
        '''dump_modify com_dump format line "%d %d %d %.2f %.2f " first yes''',
        #"thermo ${dump_steps}",
        "thermo 0",
        #"thermo_style custom step temp pe ke etotal epair ebond press vol",
        "compute cBonds all property/local batom1 batom2 btype",
        "compute cBondDxys all bond/local engpot force dist",
        f"dump 2 all local {dump_steps} {spath}/bonds.xyz c_cBonds[*]",
        '''dump_modify 2 format line "%.0f %.0f %.0f"''',
        "fix twodim all enforce2d",
        "timestep 0.01"
    ]
    
    # Execute each command
    for cmd in commands:
        lmp.command(cmd)
        
def setup_simulation_GA(lmp, spath, seed, dump_steps, eps):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary f f p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/configuration.dat",
        f"molecule monomer_template {spath}/monomer.dat",
        "group moving id > 0",
        "group none empty",
        "group mol1and2 molecule 1 2",
        "variable K_angle equal 20.0",
        f"variable seed equal {seed}",
        f"variable dump_steps equal {dump_steps}",
        f"variable eps equal {eps}",
        "variable icoff equal 1.41421356237",
        "variable Kbond1 equal 1000.000000", 
        #"variable Kbond2 equal 10000.000000",
        #"variable Kbond3 equal 100.000000",
        "variable ecoff equal 1.200000",
        "variable bcoff equal 1.200000", #WAS 1.1 for epsilon 10, 1.2 for 30, 1.5 for 50
        "variable rdist equal 1.000000",
        "variable tstep equal 0.10000",
        "special_bonds lj 0.0 1.0 1.0",
        "bond_style harmonic",
        "bond_coeff 1 ${Kbond1} 1.0",
        #"bond_coeff 2 ${Kbond2} 1.0",
        #"bond_coeff 3 ${Kbond3} 1.0",
        "angle_style harmonic",
        "angle_coeff 1 ${K_angle} 60.0",
        "angle_coeff 2 ${K_angle} 120.0",
        "angle_coeff 3 ${K_angle} 180.0",
        "pair_style hybrid/overlay zero 2.00 cosine/squared 2.00",
        "pair_coeff * * cosine/squared 0.00 1.00 1.10",
        "pair_coeff * * zero 2.00",
        "pair_coeff * * cosine/squared 1.00 1.00 1.00 wca",
        "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #"pair_coeff 4 4 cosine/squared 2 1.00 ${bcoff} wca",
        #"pair_coeff 5 5 cosine/squared 7 1.00 ${bcoff} wca",
        "compute cPE all pe/atom pair",
        #f"fix boundary all wall/reflect xlo {low} xhi {high} ylo {low} yhi {high}",
        #"fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        "fix xwall_low all wall/reflect xlo EDGE",
        "fix xwall_high all wall/reflect xhi EDGE",
        "fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
        #"minimize 1.0e-4 1.0e-6 100 1000",
        "fix fLang moving langevin 1.0 1.0 1.0 ${seed}",
        "fix fNVE moving nve",
        f"dump com_dump all custom {dump_steps} {spath}/output_com.xyz id mol type x y",
        '''dump_modify com_dump format line "%d %d %d %.2f %.2f " first yes''',
        #"thermo ${dump_steps}",
        "thermo 0",
        #"thermo_style custom step temp pe ke etotal epair ebond press vol",
        "compute cBonds all property/local batom1 batom2 btype",
        "compute cBondDxys all bond/local engpot force dist",
        f"dump 2 all local {dump_steps} {spath}/bonds.xyz c_cBonds[*]",
        '''dump_modify 2 format line "%.0f %.0f %.0f"''',
        "fix twodim all enforce2d",
        "timestep 0.01" #was 0.01 for 10 and 30, 0.005 for 100
    ]
    
    # Execute each command
    for cmd in commands:
        lmp.command(cmd)

def center_of_mass(atoms_data, target_mol_id):
    """
    Extracts all (x, y) positions of atoms belonging to a specific molecule ID
    and calculates the center of mass assuming uniform mass.
    
    Parameters:
    atoms_data (list): List of tuples (atom_id, mol_id, atom_type, x, y).
    target_mol_id (int): The molecule ID for which to calculate the center of mass.
    
    Returns:
    tuple: The center of mass (x_com, y_com) as a tuple of floats.
           Returns None if no atoms are found for the given molecule ID.
    """
    # Extract (x, y) positions for the specified molecule ID
    positions = [(x, y) for _, mol_id, _, x, y, z in atoms_data if mol_id == target_mol_id]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])

def graph_new(positions, molecules,types,atom_ids,cutoff_distance,num_of_molecules,xlo,xhi,ylo,yhi, which_simulation, insertion=False):
    bonds,atoms_data,_ , bonded_molecules, molecules, side_interacting_dict = get_bonds(positions, molecules,types,atom_ids,cutoff_distance,xlo,xhi,ylo,yhi,which_simulation,insertion)
    clusters = find_clusters(range(1,num_of_molecules+1),bonds)

    # Keep clusters if:
    # - they are longer than 1 (assumed always valid), or
    # - the single element is in mol
    # this is due to bug in the clustering algorithm
    mol_set = set(molecules)
    filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1 or cluster[0] in mol_set]

    return filtered_clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict

def analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion=False):
    '''
    Analyse the system at a given step of the simulation
    Create a graph of the system and print the sizes of the connected components, the largest molecule ID found in each graph, and the number of monomers left in the left and right sides of the box
    Output: Number of monomers left in the left and right sides of the box

    Inputs:
    step (int): Current step of the simulation
    pos (np.array): Array of positions of the atoms
    mol (np.array): Array of molecule IDs of the atoms
    cutoff (float): Distance cutoff criterion for the graph
    L (float): Length of the box in the x and y directions
    maxFrozenMol (int): Largest frozen molecule ID
    monomer_size (int): Number of atoms in a monomer
    print_mode (bool): Flag to enable or disable print statements
    '''
    
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    step = lmp.get_thermo('step')
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    atom_ids = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo  # Lower bounds
    xhi, yhi, zhi = boxhi  # Upper bounds
    num_of_molecules= max(mol)
            
    free_monomers = []
    left_monomers_num = 0
    right_monomers_num = 0
    clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict  = graph_new(pos, mol, types,atom_ids,cutoff, num_of_molecules,xlo,xhi,ylo,yhi,which_simulation,insertion)
    max_length = max([len(sublist) for sublist in clusters])

    #get most right and most left point of polymer
    max_cluster = max(clusters, key=len)#gets largest polymer
    #if len(max_cluster)==1: #if all polymers are monomers - then monomer 1 is the middle
    #    max_cluster = [1]
    # Create boolean mask: True where atom belongs to target molecules
    mask = np.isin(mol, max_cluster)
    #print('MOL')
    #print(mol)
    #print("MAX CLUSTER")
    #print(max_cluster)
    # Get positions of atoms in molecules in largest cluster (or seed)
    pos_selected = pos[mask]
    x_positions = pos_selected[:, 0]
    monomers_on_the_left = []
    monomers_on_the_right = []
    monomers_stuck_in_middle = []

    # Get min and max x-values
    x_min = x_positions.min()
    x_max = x_positions.max()

    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    for monomer in free_monomers:
        if monomer in mol:
            #print('Step:', step)
            #print('Free monomer:', monomer)
            
            com = center_of_mass(atoms_data, target_mol_id = monomer)
            comx = com[0]
            if comx<x_min:
                left_monomers_num+=1
                monomers_on_the_left.append(monomer)
                #print('Free monomers on the left:', left_monomers_num)
                #print('Monomer on left:', monomer)
                #print('COM min:',x_min)
            if comx>=x_min and comx<=x_max:
                monomers_stuck_in_middle.append(monomer)
                #print('Monomers stuck in the middle:' +  str(monomers_stuck_in_middle) + ' Step: '+str(step))
            elif comx>x_max:
                right_monomers_num+=1
                monomers_on_the_right.append(monomer)
                #print('Free monomers on the right:', right_monomers_num)
                #print('Monomer on right:', monomer)
                #print('COM ax:',x_max)

    if print_mode:
        print('Step:', step, '- Frame:', step/1000)
        print('Monomers left in left side:', monomers_free_num)
        print()

    detected_binding_molecules = [key for key in list_of_binding_events]

    for molecule in bonded_molecules:
        if molecule not in detected_binding_molecules:
            list_of_binding_events[molecule]=step #if a new binding is detected add to dictionary with the timestep at which it binds
            
    bonds = set(bonds)  # Convert list to set
    dimer = False
    if (1,2) in bonds or (2,1) in bonds:
        dimer = True

    return left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right,monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds

def get_last_timestep(dump_file):
    """Extract the last recorded timestep from a LAMMPS dump file by reading from the end."""
    last_timestep = None

    if os.path.exists(dump_file):
        with open(dump_file, 'r') as f:
            # Start from the end of the file
            f.seek(0, os.SEEK_END)
            position = f.tell()

            # Read backwards until finding 'ITEM: TIMESTEP'
            while position > 0:
                # Move the cursor to the previous line
                position -= 1
                f.seek(position)
                if f.read(1) == '\n':
                    # When we find a new line, check the next few characters
                    line = f.readline().strip()
                    if line == "ITEM: TIMESTEP":
                        # Read the next line for the timestep
                        last_timestep = int(f.readline().strip())
                        break

    return last_timestep

def run(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps, eps, which_simulation, delta_L=20, delta=10): #littleguys: delta_L = 20, delta =10, GA: delta_L=25, delta=15
    '''
    Run the LAMMPS simulation of the chevron-like molecules
    Run the simulation for a given number of steps and analyse the system at regular intervals

    Inputs:
    spath (str): Path to the directory where the input script is located (simulation folder)
    nMonsSol (int): Number of monomers in solution (per side of the box or reservoir)
    cutoff (float): Distance cutoff criterion for the graph
    loops (int): Number of iterations to run the simulation
    run_interval (int): Number of steps to run the simulation for each iteration
    maxFrozenMol (int): Largest frozen molecule ID
    debug_mode (bool): Flag to enable or disable LAMMPS outputs
    print_mode (bool): Flag to enable or disable print statements
    '''
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
    
    sucess_of_run = False# Instantiate LAMMPS eithout output

    if which_simulation == 'GA':
        setup_simulation_GA(lmp, sim_dir, seed, dump_steps, eps)
    elif which_simulation == 'littleguys':
        setup_simulation_littleguys(lmp, sim_dir, seed, dump_steps, eps)

    list_of_binding_events = dict()
    monomer_birth = dict()
    monomer_death = dict()
    unfixed_mols = []
    
    output_file = 'clusters_log.txt'
    
    with open(os.path.join(sim_dir, output_file), 'w') as f:
        f.write('##Lammps simulation started\n')
        f.write('###Step Clusters Bonds\n')
        
    with open(os.path.join(sim_dir, 'cluster_fullinterface_log.txt'), 'w') as f:
        f.write('##Lammps simulation started\n')
        f.write('###Step Clusters Edge_molecules Side_interacting_dict\n')
        
    with open(sim_dir + "/monomer_birth_log.txt", "w") as f:
        f.write("###LAMMPS simulation started\n")
        f.write("###Step, Monomer Birth\n")
        
    with open(sim_dir + "/monomer_death_log.txt", "w") as f:
        f.write("###LAMMPS simulation started\n")
        f.write("###Step, Monomer Death\n")

    # Get box size
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo") #init
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo  # Lower bounds
    xhi, yhi, zhi = boxhi  # Upper bounds
    


    # Get positions, molecules and ids
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    ids = lmp.numpy.extract_atom("id")

    #find number of monomers
    molecule_number = 1

    # Find the size of the monomer
    monomer_size = len(pos[mol == 1])

    types = lmp.numpy.extract_atom("type")

    # Initial graph
    step = lmp.get_thermo('step')
    left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation)
    G, clusters_graph, egde_molecules, side_interacting_dict = get_fullinterface_clusters(lmp, which_simulation)
    max_polymer_full_interface = max(clusters_graph, key=len)
    counter = 1
    counter_left = 0
    counter_right = 0
    
    max_polymer = max(clusters, key=len)

    # Iterate through loops
    for i in range(loops):
    #for i in tqdm(range(loops)):
        # Run for run_interval steps
        
        left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = False)
        new_max_polymer = max(clusters, key=len)
        step = lmp.get_thermo('step')
        if sorted(new_max_polymer) != sorted(max_polymer):
            with open(os.path.join(sim_dir, output_file), 'a') as f:
                f.write(f" {step}, {clusters} {bonds}\n")
        
        G, clusters_graph, egde_molecules, side_interacting_dict = get_fullinterface_clusters(lmp, which_simulation)
        new_max_polymer_full_interface = max(clusters_graph, key=len)
        if sorted(new_max_polymer_full_interface) != sorted(max_polymer_full_interface):
            with open(os.path.join(sim_dir, 'cluster_fullinterface_log.txt'), 'a') as f:
                f.write(f" {step}, {clusters_graph} {egde_molecules} {side_interacting_dict}\n")
                
        mols = (lmp.numpy.extract_atom("id")-1)//8
        for monomer in mols:
            if monomer not in monomer_birth:
                monomer_birth[monomer] = step
                with open(sim_dir + "/monomer_birth_log.txt", "a") as f:
                    f.write(f"{step}, {monomer}\n")
                monomer_death.pop(monomer, None)  # Remove from death dictionary if it is reborn
        
        for monomer in list(monomer_birth.keys()):
            if (monomer not in mols) and (monomer not in monomer_death):
                monomer_death[monomer] = step
                with open(sim_dir + "/monomer_death_log.txt", "a") as f:
                    f.write(f"{step}, {monomer}\n")
                monomer_birth.pop(monomer, None)  # Remove from birth dictionary if it dies

        
        max_cluster_length = max([len(sublist) for sublist in clusters])

        lmp.command('run %d'%(run_interval))
        if max_cluster_length>1:
            # Analyse the system
            #L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
            #step = lmp.get_thermo('step')
            #pos = lmp.numpy.extract_atom("x")
            #mol = lmp.numpy.extract_atom("molecule")
            #ids = lmp.numpy.extract_atom("id")
            #types = lmp.numpy.extract_atom("type")
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            #xlo, ylo, zlo = boxlo  # Lower bounds
            #xhi, yhi, zhi = boxhi  # Upper bounds
            #molecule_number = max(mol)
            x_walls_low = xlo
            x_walls_high = xhi

            if len(mol)==0:
                print('No more monomers left!')
                sucess_of_run = True
                break
            
            left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = False)
            #com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
            bounded_monomers = [key for key in list_of_binding_events]
            
            if dimer: # fix dimer in place
                lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')

            if left_monomers_num < nMonsSol and dimer:
                nmonomers = nMonsSol-left_monomers_num
                
                pos = lmp.numpy.extract_atom("x")
                x_max = pos.max()
                x_min = pos.min()
                
                left_monomers_num, right_monomers_num,monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = False)
                com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
                x_walls_lo, x_walls_hi = com_x_min_mol - delta_L, com_x_max_mol + delta_L
                
                boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
                xlo, ylo, zlo = boxlo  # Lower bounds
                xhi, yhi, zhi = boxhi  # Upper bounds
                increase_box_left = x_walls_lo - xlo
                """            
                #delete free monomers and re-add new ones at the same distance
                for cluster in clusters:
                    if len(cluster)==1:
                        free_monomer = cluster[0]
                        lmp.command('group mol_%d molecule %d'%(free_monomer,free_monomer))
                        lmp.command('delete_atoms group mol_%d'%(free_monomer))
                        lmp.command('group mol_%d delete '%(free_monomer))
                """ 
                #lmp.command('unfix xwalls')
                if x_max>x_walls_hi-5: #EXTRA CHECK - the polymer still moves around so if the monomer is very close to the box edge it can cause error when the box edge moves slightly - to prevent this use change_box delta instead and make sure the delta is 0 for the other side of the box
                    lmp.command('unfix xwall_low')
                    lmp.command('change_box          all x delta %f %f \n\n' %(increase_box_left, 0))
                    lmp.command("fix xwall_low all wall/reflect xlo EDGE")
                else:
                    lmp.command('unfix xwall_low')
                    lmp.command('unfix xwall_high')
                    lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
                    lmp.command("fix xwall_low all wall/reflect xlo EDGE")
                    lmp.command("fix xwall_high all wall/reflect xhi EDGE")
                #lmp.command("fix xwalls all wall/reflect xlo %f xhi %f" %(x_walls_low, x_walls_high))
                #lmp.command('run 10')
                
                # Extract box size
                boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
                xlo, ylo, zlo = boxlo  # Lower bounds
                xhi, yhi, zhi = boxhi  # Upper bounds


                lmp.command("region              insertion_region_left_%d block %f %f 0 0 0 0 units box" %(counter,com_x_min_mol - delta, com_x_min_mol - delta))
                lmp.command('fix                 add_mons_left_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left_%d '%(counter,nmonomers,seed,counter))
                
                #lmp.command("region              insertion_region_right_%d block %f %f 0 0 0 0 units box" %(counter,com_x_max_mol+delta , com_x_max_mol+delta))
                #lmp.command('fix                 add_mons_right_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d '%(counter,nmonomers,seed,counter))

                counter +=1
                counter_left +=1
                lmp.command('run 10')
                left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = False)

            if right_monomers_num < nMonsSol and dimer:
                nmonomers = nMonsSol-right_monomers_num

                left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = False)
                com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
                x_walls_lo, x_walls_hi = com_x_min_mol - delta_L, com_x_max_mol + delta_L
                
                boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
                xlo, ylo, zlo = boxlo  # Lower bounds
                xhi, yhi, zhi = boxhi  # Upper bounds
                increase_box_right = x_walls_hi - xhi
                """
                #delete free monomers and re-add new ones at the same distance
                for cluster in clusters:
                    if len(cluster)==1:
                        free_monomer = cluster[0]
                        lmp.command('group mol_%d molecule %d'%(free_monomer,free_monomer))
                        lmp.command('delete_atoms group mol_%d'%(free_monomer))
                        lmp.command('group mol_%d delete '%(free_monomer))
                """
                pos = lmp.numpy.extract_atom("x")
                x_max = pos.max()
                x_min = pos.min()
                
                #lmp.command('unfix xwalls')
                if x_min<x_walls_lo+5: #EXTRA CHECK - the polymer still moves around so if the monomer is very close to the box edge it can cause error when the box edge moves slightly - to prevent this use change_box delta instead and make sure the delta is 0 for the other side of the box
                    lmp.command('unfix xwall_high')
                    lmp.command('change_box          all x delta %f %f \n\n' %(0, increase_box_right))
                    lmp.command("fix xwall_high all wall/reflect xhi EDGE")
                else:
                    lmp.command('unfix xwall_low')
                    lmp.command('unfix xwall_high')
                    lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
                    lmp.command("fix xwall_low all wall/reflect xlo EDGE")
                    lmp.command("fix xwall_high all wall/reflect xhi EDGE")
                #lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
                #lmp.command("fix xwalls all wall/reflect xlo %f xhi %f" %(x_walls_low, x_walls_high))
                #lmp.command('run 10')
        
                # Unpacking values
                boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
                xlo, ylo, zlo = boxlo  # Lower bounds
                xhi, yhi, zhi = boxhi  # Upper bounds

                #lmp.command('unfix trap_bottom')
                #lmp.command('group bottom delete')
                
                #lmp.command("region              insertion_region_left_%d block %f %f 0 0 0 0 units box" %(counter,com_x_min_mol - delta, com_x_min_mol - delta))
                #lmp.command('fix                 add_mons_left_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left_%d '%(counter,nmonomers,seed,counter))
                
                lmp.command("region              insertion_region_right_%d block %f %f 0 0 0 0 units box" %(counter,com_x_max_mol+delta , com_x_max_mol+delta))
                lmp.command('fix                 add_mons_right_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d '%(counter,nmonomers,seed,counter))
                
                #lmp.command('create_atoms          0 single 5 5 0 mol monomer_template 1234')
                #lmp.command('group bottom type 1 2 3 ')
                #lmp.command('fix trap_bottom bottom addforce 0.0 v_force_trapping_bottom 0.0 every 1 ')
                #lmp.command('minimize 1.0e-4 1.0e-6 10 1000')
                counter +=1
                counter_right+=1
                lmp.command('run 10')
                
                #left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation,insertion = True)

            left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, 2, print_mode,list_of_binding_events, which_simulation, insertion = False)
            #HARD CODED ABOVE - shifted cutoff to be 0.5 larger to detect monomers as free only when a bit far away from polymer
            #delete unneccessary moelcules
            polymer_number = sum(1 for sublist in clusters if len(sublist) > 1)
            max_polymer_size = max(clusters, key=len) #if breaking occurs keep only the longest filament
            monomer_number = sum(1 for sublist in clusters if len(sublist) == 1 and sublist[0] in mol)
            flattened_mols = [item for sublist in clusters for item in sublist]
            molecules_to_delete = []
            if len(max_polymer_size) ==1:
                monomer_number_max = 3
            elif len(max_polymer_size)>1:
                monomer_number_max = 2

            left_deleted = False
            right_deleted = False
            if polymer_number>1: #if more than one polymer exists-this can happen through fragmentation of the polymer, delete other polymers+ monomers on that side only
                #print('Polymer number:' + str(polymer_number))
                #print('Monomer number:' + str(monomer_number))
                for molecule in flattened_mols:
                    if molecule not in max_polymer_size:
                        for cluster in clusters:
                            if len(cluster)>1 and molecule in cluster:
                                comx, comy = center_of_mass(atoms_data, target_mol_id = molecule)
                                com_longest_filament_x, com_longest_filament_y = center_of_mass(atoms_data, target_mol_id = max_polymer_size[0])
                                molecules_to_delete.append(molecule)
                                if comx<com_longest_filament_x:
                                    for left_mol in monomers_on_the_left:
                                        molecules_to_delete.append(left_mol)
                                        left_deleted = True
                                elif comx>com_longest_filament_x:
                                    for right_mol in monomers_on_the_right:
                                        molecules_to_delete.append(right_mol)
                                        right_deleted = True
                            
            if monomer_number>monomer_number_max: #if too many monomers exist - can happen through end depolymerisation, delete all monomers on this ONE side
                if len(monomers_on_the_left)>nMonsSol:
                    for mols in monomers_on_the_left:
                        molecules_to_delete.append(mols)
                        left_deleted = True
                elif len(monomers_on_the_right)>nMonsSol:
                    for mols in monomers_on_the_right:
                        molecules_to_delete.append(mols)
                        right_deleted = True
                        
            #print('Clusters: ' + str(clusters))
            #print('Max length ' + str(len(max_polymer_size)))
            if which_simulation == 'littleguys':
                if len(max_polymer_size)>2: #if a polymer exists, delete all monomers stuck in the middle
                    #print('Max length ' + str(len(max_polymer_size)) + ' detected!')
                    #print('Monomers stuck in the middle:' +  str(monomers_stuck_in_middle) + ' Step: '+str(step))
                    for mols in monomers_stuck_in_middle: #delete monomers stuck in the middle
                        molecules_to_delete.append(mols)
                        #print('Monomers stuck in the middle deleted:' +  str(monomers_stuck_in_middle) + ' Step: '+str(step))
            
            mol = lmp.numpy.extract_atom("molecule")
                    
            deleted = False
            for mols in molecules_to_delete:
                if mols in mol:
                    #print('Molecule %d deleted at step %d' %(mols, step))
                    deleted = True
                    lmp.command('group mol_%d molecule %d'%(mols,mols))
                    lmp.command('run 1')
                    lmp.command('delete_atoms group mol_%d'%(mols))
                    lmp.command('run 1')
                    lmp.command('group mol_%d delete '%(mols))
                    lmp.command('run 1')
                    #deleted_atoms.append(mols)
                    if mols in list_of_binding_events: #since LAMMPS recycles molecule IDs, we remove the information we had on the previous monomer during its deletion
                        list_of_binding_events.pop(mols)
                    if mols in unfixed_mols:
                        unfixed_mols.remove(mols)

            lmp.command('run 10')

            if deleted: #if deletions happened, reanalyse the system and make new walls
                if len(mol)==0:
                    #print('No more monomers left!')
                    break
                
                pos = lmp.numpy.extract_atom("x")
                x_max = pos.max()
                x_min = pos.min()
                
                boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
                xlo, ylo, zlo = boxlo  # Lower bounds
                xhi, yhi, zhi = boxhi  # Upper bounds
                
                left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, dimer, monomers_stuck_in_middle, bonds = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, which_simulation)
                com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
                x_walls_lo, x_walls_hi = com_x_min_mol - delta_L, com_x_max_mol + delta_L
                increase_box_left = x_walls_lo - xlo
                increase_box_right = x_walls_hi - xhi
                
                if x_min<x_walls_lo+5 and right_deleted == True: #EXTRA CHECK - the polymer still moves around so if the monomer is very close to the box edge it can cause error when the box edge moves slightly - to prevent this use change_box delta instead and make sure the delta is 0 for the other side of the box
                    lmp.command('unfix xwall_high')
                    lmp.command('change_box          all x delta %f %f \n\n' %(0, increase_box_right))
                    lmp.command("fix xwall_high all wall/reflect xhi EDGE")
                elif x_max>x_walls_hi-5 and left_deleted == True: #EXTRA CHECK - the polymer still moves around so if the monomer is very close to the box edge it can cause error when the box edge moves slightly - to prevent this use change_box delta instead and make sure the delta is 0 for the other side of the box
                    lmp.command('unfix xwall_low')
                    lmp.command('change_box          all x delta %f %f \n\n' %(increase_box_left, 0))
                    lmp.command("fix xwall_low all wall/reflect xlo EDGE")
                else:
                    #lmp.command('unfix xwalls') #make new walls to fit the remaining polymer
                    lmp.command('unfix xwall_low')
                    lmp.command('unfix xwall_high')
                    lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
                    lmp.command("fix xwall_low all wall/reflect xlo EDGE")
                    lmp.command("fix xwall_high all wall/reflect xhi EDGE")
                    #lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE") #%(x_walls_low, x_walls_high))
                    lmp.command('run 10')
                    
            max_polymer = new_max_polymer
            max_polymer_full_interface = new_max_polymer_full_interface
        
    # Clean up
    lmp.close()
    sucess_of_run = True
    return sucess_of_run


def write_info_file(path, nMonsSol, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps):
    '''
    Write an information file with the parameters of the simulation

    Inputs:
    spath (str): Path to the directory where the information file will be written
    Lbox (float): Length of the box in the x and y directions
    nMonsSol (int): Number of monomers in solution (per side of the box or reservoir)
    seed (int): Seed for the random number generator
    cutoff (float): Distance cutoff criterion for the graph
    run_interval (int): Number of steps to run the simulation for each iteration
    run_steps (int): Total number of steps to run the simulation
    dump_steps (int): Dumping interval in simulation steps
    debug_mode (bool): Flag to enable or disable LAMMPS outputs
    print_mode (bool): Flag to enable or disable print statements
    eps (float): Binding constant for the monomer-monomer interaction
    epsO (float): Binding constant for the open conformation pair interaction
    epsC (float): Binding constant for the closed conformation pair interaction
    '''
    info_file = open(f'{path}/info.txt', 'w')
    info_file.write(f'Simulation folder: {path}\n')
    info_file.write(f'nMonsSol: {nMonsSol}\n')
    info_file.write(f'run_interval: {run_interval}\n')
    info_file.write(f'run_steps: {run_steps}\n')
    info_file.write(f'epsilon: {eps}\n')
    info_file.write(f'dump_steps: {dump_steps}\n')
    info_file.write(f'debug_mode: {debug_mode}\n')
    info_file.write(f'print_mode: {print_mode}\n')
    info_file.close()

def single_seed(spath, nMonsSol, seed, loops, run_interval, debug_mode, dump_steps, print_mode, eps,cutoff,which_simulation, path_to_ind_text_file, init_region_L_width, init_region_L_height):
    '''
    Run the simulation for a single seed

    Inputs:
    spath (str): Path to the directory where the input script is located (simulation folder)
    Lbox (float): Length of the box in the x and y directions
    nMonsSol (int): Number of monomers in solution (per side of the box or reservoir)
    seed (int): Seed for the random number generator
    cutoff (float): Distance cutoff criterion for the graph
    loops (int): Number of iterations to run the simulation
    run_interval (int): Number of steps to run the simulation for each iteration
    debug_mode (bool): Flag to enable or disable LAMMPS outputs
    dump_steps (int): Dumping interval in simulation steps
    eps (float): Binding constant for the monomer-monomer interaction
    epsO (float): Binding constant for the open conformation pair interaction
    epsC (float): Binding constant for the closed conformation pair interaction
    print_mode (bool): Flag to enable or disable print statements
    '''

    # Check if the simulation folder exists
    if not os.path.exists(spath):
        r = os.makedirs(spath, exist_ok=True)
    #print()
    #print('Simulation folder:', spath)
    
    if which_simulation == 'GA':
        ind,random_n_type = read_individual_txt_file(os.path.join(path_to_ind_text_file, 'individual.txt'))
        #write_monomer_template(spath)
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps_one_monomer(ind,spath, init_region_L_width, init_region_L_height, random_n_type)

        # Setup the simulation and write the configuration.dat file
        #print()
        #print('Starting simulation setup...')
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps(ind,gpath = spath, box_width=init_region_L_width, box_height=init_region_L_height, number_of_monomers = 2,random_n=random_n_type)
        
    elif which_simulation == 'littleguys':
        ind,random_n_type = read_individual_txt_file(os.path.join(path_to_ind_text_file, 'individual.txt'))
        #write_monomer_template(spath)
        write_lammps_more_monomers_3_types_vary_bond_stiffness_switch.write_lammps_one_monomer(ind,spath, init_region_L_width, init_region_L_height, random_n_type)

        # Setup the simulation and write the configuration.dat file
        #print()
        #print('Starting simulation setup...')
        write_lammps_more_monomers_3_types_vary_bond_stiffness_switch.write_lammps(ind,gpath = spath, box_width=init_region_L_width, box_height=init_region_L_height, number_of_monomers = 2,random_n=random_n_type)

    
    sucess_of_run = run(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps,eps, which_simulation)
    
    return sucess_of_run

def run_simulation(path, path_to_ind_text_file, nMonsSol=1, run_interval=100, run_steps=1000000000, dump_steps=10000, eps = 8, cutoff =1.1, debug_mode=True, print_mode=False, Nseeds=1,which_simulation = 'littleguys', init_region_L_width=15, init_region_L_height=10): #littleguys: initial_box_width = height = 10, eps=15, GA eps=30, L=15
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")
    
    trial = 0
    success_of_run = False
    while not success_of_run and trial<50:
        seeds = np.random.randint(10000, 99999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(10000, 99999, Nseeds)

        seed = seeds[0]

        # Create the simulation folders
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Write the information file
        write_info_file(path, nMonsSol, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps)

        # Define number of loops
        loops = run_steps//run_interval

        # Run the simulations
        # Run the simulations
        # Define custom exception
        class TimeoutException(Exception):
            pass

        # Define signal handler
        def timeout_handler(signum, frame):
            raise TimeoutException()

        # Set the signal handler once
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            success_of_run = single_seed(path, nMonsSol, seed, loops, run_interval, debug_mode, dump_steps, print_mode, eps,cutoff,which_simulation,  path_to_ind_text_file, init_region_L_width, init_region_L_height)
            trial += 1
            print('Trial number' + str(trial) + 'for simulation at' + str(path))
            if success_of_run == True:
                print('Simulation successfully finished at ' + str(path))
        except Exception as e:
            print('ERROR' + str(e) +' IN SIMULATION AT ' + str(path))

#run_simulation(path='/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/make_GA_movie_run_longer/gen0take3', path_to_ind_text_file = '/nfs/scistore26/saricgrp/mkrstic/ga_code/run_vary_stiffness_rectbox_1_elite_justpolarity_10_seeds_per_ind_check_directionality_monomers_eval_all_eps_30/deme_8/gen_0/ind_34/seed_523744')