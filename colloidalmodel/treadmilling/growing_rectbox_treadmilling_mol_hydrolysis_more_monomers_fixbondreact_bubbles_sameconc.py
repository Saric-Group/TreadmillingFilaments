import lammps
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import ast
from collections import defaultdict, deque, Counter
import random
import signal
import networkx as nx


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

def find_endpoints(bonds):
    counts = Counter()
    for a, b in bonds:
        counts[a] += 1
        counts[b] += 1
    # numbers that appear only once
    return [num for num, count in counts.items() if count == 1]

def rotate_polymer(lmp,cutoff, print_mode,list_of_binding_events, which_simulation):
    
    #rotate the polymer such that it always aligns with the box
    #step 1: get clusters and see which molecules are in the polymer
    #step 2: get molecules positions - filter out only the ones got from step 1
    #step 3: calculate a direction vector of the polymer - find the end monomers and find the positions of the same particle id in them -> vector
    #step 4: based on the previous vector and the vector of the box direcitonality define a rotation to recitfy the polymer direction
    
    #step 1:
    left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right = analyse_simulation_monomerlife_fullinterface(lmp, 2, print_mode,list_of_binding_events, which_simulation, hydrolysis)
    polymer = max(clusters, key=len) #list of moelcules that are in the polymer
    polymer = [int(x) for x in polymer]
    
    if len(polymer)>1:
        
        #step 3: calculate direction of polymer
        
        #get the COM of most left and right molecule in the polymer
        com_x_max_mol, com_x_min_mol, com_y_max_mol, com_y_min_mol = get_most_left_and_right_com_x_and_y(lmp,clusters)
        max_mol_pos = np.array([com_x_max_mol,com_y_max_mol])
        min_mol_pos = np.array([com_x_min_mol,com_y_min_mol])
        polymer_direction = max_mol_pos - min_mol_pos
        polymer_direction = polymer_direction/ np.linalg.norm(polymer_direction)
        
        #step 4: get rotation matrix
        direction_of_box = np.array([1,0])
        
        # rotation angle from polymer direction to direction of box
        theta = np.arctan2(direction_of_box[1], direction_of_box[0]) - np.arctan2(polymer_direction[1], polymer_direction[0])
        
        #building the rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])
        
        #performing rotation of positions and velocities in LAMMPS
        rotated_data = rotate_molecules_2d(lmp, polymer, R)
    

def rotate_molecules_2d(lmp, mol_ids, R, rotate_com_velocity=False):
    """
    Rotate multiple molecules in 2D (xy-plane) using a precomputed rotation matrix R.
    Rotates both positions and velocities to preserve dynamics.

    Parameters:
        lmp : LAMMPS instance
            Your running LAMMPS object.
        mol_ids : list or array-like of int
            Molecule IDs to rotate.
        R : np.ndarray of shape (2,2)
            2D rotation matrix to apply to all molecules.
        rotate_com_velocity : bool
            If True, rotates all velocities; if False, keeps COM velocity unchanged.

    Returns:
        dict
            Dictionary mapping mol_id to tuple (coords_rot, vels_rot) of rotated positions and velocities.
    """
    # Extract positions (shape: N_atoms x 3), velocities (N_atoms x 3), and molecule IDs
    pos = lmp.numpy.extract_atom("x")  # shape (N,3) -> columns: x, y, z
    vel = lmp.numpy.extract_atom("v")  # shape (N,3) -> columns: vx, vy, vz
    mol = lmp.numpy.extract_atom("molecule")

    if pos is None or vel is None or mol is None:
        raise RuntimeError("LAMMPS atom arrays not available. Did you run 'run 0'?")

    rotated_data = {}

    for mol_id in mol_ids:
        mask = (mol == mol_id)
        if not np.any(mask):
            print(f"Warning: No atoms found for molecule {mol_id}, skipping.")
            continue

        coords = pos[mask, :2]  # take only x and y
        vels = vel[mask, :2]

        # Center-of-mass
        com = coords.mean(axis=0)
        coords_centered = coords - com

        # Rotate positions
        coords_rot = coords_centered @ R.T + com

        # Rotate velocities
        v_com = vels.mean(axis=0)
        if rotate_com_velocity:
            vels_rot = vels @ R.T
        else:
            vels_rot = (vels - v_com) @ R.T + v_com

        # Write back to LAMMPS arrays
        pos[mask, 0:2] = coords_rot
        vel[mask, 0:2] = vels_rot

        # Store rotated data for reference
        rotated_data[mol_id] = (coords_rot, vels_rot)

    # Force LAMMPS to update internal arrays
    lmp.command("run 0 pre no post no")

    return rotated_data



def hydrolysis_happens(hydrolysis_probability): #returns true or false
    # Generate a random number between 0 and 1
    U = random.random()
    # Check if the event occurs
    return U < hydrolysis_probability

def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds, which_simulation, binding_dump_mode):
    xlo, xhi, ylo, yhi = box_bounds
    total_bond_number = 0
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
                    #if (atom_type1 == 3 and atom_type2 == 4) or (atom_type1 == 4 and atom_type2 == 3):
                    if binding_dump_mode:
                        if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6) or (atom_type1 == 8 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 8) or (atom_type1 == 10 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 10) or (atom_type1 == 3 and atom_type2 == 4) or (atom_type1 == 4 and atom_type2 == 3): #make sure to have full binding
                            total_bond_number += 1
                            if total_bond_number==3:
                                if atom_type1 in [1,6,8,10,4] and atom_type2 in [2,7,9,11,3]:
                                    side_interacting = 'back'
                                elif atom_type1 in [2,7,9,11,3] and atom_type2 in [1,6,8,10,4]:
                                    side_interacting = 'front'
                                return True, side_interacting
                    else: 
                        if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6) or (atom_type1 == 8 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 8) or (atom_type1 == 10 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 10) or (atom_type1 == 3 and atom_type2 == 4) or (atom_type1 == 4 and atom_type2 == 3): #make sure to have full binding
                            if atom_type1 in [1,6,8,10,4] and atom_type2 in [2,7,9,11,3]:
                                side_interacting = 'back'
                            elif atom_type1 in [2,7,9,11,3] and atom_type2 in [1,6,8,10,4]:
                                side_interacting = 'front'
                            return True, side_interacting
    return False,side_interacting

def get_most_left_and_right_com_x_and_y(lmp, clusters):
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
    max_mol_y = pos[max_mol_mask, 1] 
    min_mol_x = pos[min_mol_mask, 0]
    min_mol_y = pos[min_mol_mask, 1]

    # center of mass of the molecule with max index?
    com_x_max_mol = np.mean(max_mol_x)
    com_y_max_mol = np.mean(max_mol_y )
    # center of mass of the molecule with min index?
    com_x_min_mol = np.mean(min_mol_x)
    com_y_min_mol = np.mean(min_mol_y)

    return com_x_max_mol, com_x_min_mol, com_y_max_mol, com_y_min_mol


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

def analyze_bonds(bonds):
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

    return G, clusters, edge_molecules

def get_bonds(positions, molecules,types,atom_ids,cutoff_distance,xlo,xhi,ylo,yhi, which_simulation, binding_dump_mode):
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
                are_they_touching, side_interacting = are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds,which_simulation, binding_dump_mode) #if theyre touching move
                if are_they_touching:
                    bonds.append((mol_id1, mol_id2))
                    if mol_id1 not in bonded_molecules:
                        bonded_molecules.append(mol_id1)
                    if mol_id2 not in bonded_molecules:
                        bonded_molecules.append(mol_id2)
                    side_interacting_dict[(mol_id1, mol_id2)] = side_interacting
                    
    #print('BONDS BETWEEN:')
    #print(len(bonds))
    #print(bonds)
    return bonds,atoms_data,box_bounds, bonded_molecules,molecules, side_interacting_dict

def setup_simulation_littleguys(lmp, spath, seed, dump_steps, eps,hydrolysis_probability,eps_top_final):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary p p p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/restart_config.dat extra/special/per/atom 1000 extra/bond/per/atom 5",
        f"molecule monomer_template {spath}/monomer.dat",
        "molecule mPreHydrolysis1 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis1_pre.txt",
        "molecule mPostHydrolysis1 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis1_post.txt",
        "molecule mPreHydrolysis2 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis2_pre.txt",
        "molecule mPostHydrolysis2 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis2_post.txt",
        "molecule mPreHydrolysis3 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis3_pre.txt",
        "molecule mPostHydrolysis3 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis3_post.txt",
        "molecule mPreHydrolysis4 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis4_pre.txt",
        "molecule mPostHydrolysis4 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis4_post.txt",
        "molecule mPreDeHydrolysis /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/dehydrolysis_pre.txt",
        "molecule mPostDeHydrolysis /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/dehydrolysis_post.txt",
        "group moving id > 0",
        "group none empty",
        "group mol1and2 molecule 1 2",
        f"variable seed equal {seed}",
        f"variable dump_steps equal {dump_steps}",
        f"variable eps equal {eps}",
        f"variable eps_final equal {eps_top_final}",
        "variable eps_C equal 3", # was opposite - 3/6
        "variable eps_T equal 6",
        "variable icoff equal 1.41421356237",
        "variable Kbond1 equal 1000.000000",
        #"variable Kbond2 equal 10000.000000",
        #"variable Kbond3 equal 100.000000",
        "variable ecoff equal 1.200000",
        "variable bcoff equal 1.200000",
        "variable reactcoff equal 3.0",
        "variable rdist equal 1.000000",
        "variable tstep equal 0.10000",
        "special_bonds lj 0 1.0 1.0",
        "bond_style harmonic",
        "bond_coeff 1 ${Kbond1} 1.0",
        #"bond_coeff 2 ${Kbond2} 1.0",
        #"bond_coeff 3 ${Kbond3} 1.0",
        "pair_style hybrid/overlay zero 3.00 cosine/squared 2.00",
        "pair_coeff * * cosine/squared 0.00 1.00 1.10",
        "pair_coeff * * zero 3.00",
        "pair_coeff * * cosine/squared 1.00 1.00 1.00 wca",
        #LITTLE GUYS SETUP - specific binding sites
        "pair_coeff 2 7 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        "pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca",
        "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        "pair_coeff 6 7 cosine/squared ${eps} 1.00 ${bcoff} wca",
        "pair_coeff 9 11 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        "pair_coeff 8 9 cosine/squared ${eps_final} 1.00 ${bcoff} wca",
        "pair_coeff 1 9 cosine/squared ${eps_final} 1.00 ${bcoff} wca",
        "pair_coeff 8 2 cosine/squared ${eps_final} 1.00 ${bcoff} wca", #back of monomer in hydrolysis
        "pair_coeff 10 11 cosine/squared ${eps_final} 1.00 ${bcoff} wca",
        "pair_coeff 6 11 cosine/squared ${eps_final} 1.00 ${bcoff} wca", #front of the filament
        "pair_coeff 10 7 cosine/squared ${eps_final} 1.00 ${bcoff} wca", #back of monomer in hydrolysis
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
        #"fix xwall_low all wall/reflect xlo EDGE",
        #"fix xwall_high all wall/reflect xhi EDGE",
        #"fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
        #"minimize 1.0e-4 1.0e-6 100 1000",
        #"comm_modify mode single cutoff 4", #hydrolysis time was 20000, trying out 10x more frequent, since the off rates are low
        f"fix fReact all bond/react react rHydrolysis1 all 20000 0 1.2 mPreHydrolysis1 mPostHydrolysis1 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis1_map.txt prob {hydrolysis_probability} 1234 react rHydrolysis2 all 1 0 1.2 mPreHydrolysis2 mPostHydrolysis2 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis2_map.txt prob {hydrolysis_probability} 1234 react rHydrolysis3 all 1 0 1.2 mPreHydrolysis3 mPostHydrolysis3 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis3_map.txt prob {hydrolysis_probability} 1234 react rHydrolysis4 all 1 0 1.2 mPreHydrolysis4 mPostHydrolysis4 /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/hydrolysis4_map.txt prob {hydrolysis_probability} 1234 ", 
        #react rDeHydrolysis all 1 0 1.2 mPreDeHydrolysis mPostDeHydrolysis /nfs/scistore26/saricgrp/mkrstic/ga_code/template_hydrolysis_reaction/dehydrolysis_map.txt prob 0 1234",
        "fix fLang moving langevin 1.0 1.0 1.0 ${seed}",
        "fix fNVE moving nve",
        f"dump com_dump all custom {dump_steps} {spath}/output_com.xyz id mol type x y",
        '''dump_modify com_dump format line "%d %d %d %.2f %.2f " first yes''',
        #"thermo ${dump_steps}",
        "thermo 0",
        #"thermo_style custom step temp pe ke etotal epair ebond press vol",
        "compute cBonds all property/local batom1 batom2 btype",
        "compute cBondDxys all bond/local engpot force dist",
        "compute cDistType2 all coord/atom cutoff ${bcoff} 2",
        "compute cDistType3 all coord/atom cutoff ${bcoff} 3",
        "compute cDistType7 all coord/atom cutoff ${bcoff} 7",
        "compute cDistType1 all coord/atom cutoff ${bcoff} 1",
        "compute cDistType4 all coord/atom cutoff ${bcoff} 4",
        "compute cDistType6 all coord/atom cutoff ${bcoff} 6",
        "compute cDistType9 all coord/atom cutoff ${bcoff} 9",
        "compute cDistType11 all coord/atom cutoff ${bcoff} 11",
        "compute cDistType8 all coord/atom cutoff ${bcoff} 8",
        "compute cDistType10 all coord/atom cutoff ${bcoff} 10",
        #computes for dehydrolysis - compute neighbors with a larger cutoff to turn back to full binding
        "compute cDehydDistType2 all coord/atom cutoff ${reactcoff} 2",
        "compute cDehydDistType3 all coord/atom cutoff ${reactcoff} 3",
        "compute cDehydDistType7 all coord/atom cutoff ${reactcoff} 7",
        "compute cDehydDistType1 all coord/atom cutoff ${reactcoff} 1",
        "compute cDehydDistType4 all coord/atom cutoff ${reactcoff} 4",
        "compute cDehydDistType6 all coord/atom cutoff ${reactcoff} 6",
        "compute cDehydDistType9 all coord/atom cutoff ${reactcoff} 9",
        "compute cDehydDistType11 all coord/atom cutoff ${reactcoff} 11",
        "compute cDehydDistType8 all coord/atom cutoff ${reactcoff} 8",
        "compute cDehydDistType10 all coord/atom cutoff ${reactcoff} 10",
        #variable for fix bond/react hydrolysis
        'variable vBound2 atom "c_cDistType2 > 0"',
        'variable vBound3 atom "c_cDistType3 > 0"',
        'variable vBound7 atom "c_cDistType7 > 0"',
        'variable vBound1 atom "c_cDistType1 > 0"',
        'variable vBound4 atom "c_cDistType4 > 0"',
        'variable vBound6 atom "c_cDistType6 > 0"',
        'variable vBound8 atom "c_cDistType8 > 0"',
        'variable vBound10 atom "c_cDistType10 > 0"',
        'variable vBound9 atom "c_cDistType9 > 0"',
        'variable vBound11 atom "c_cDistType11 > 0"',
        ##varaible for fix bond/react dehydrolysis
        'variable vDehydBound2 atom "c_cDehydDistType2 < 1"',
        'variable vDehydBound3 atom "c_cDehydDistType3 < 2"',
        'variable vDehydBound7 atom "c_cDehydDistType7 < 1"',
        'variable vDehydBound1 atom "c_cDehydDistType1 < 1"',
        'variable vDehydBound4 atom "c_cDehydDistType4 < 2"',
        'variable vDehydBound6 atom "c_cDehydDistType6 < 1"',
        'variable vDehydBound8 atom "c_cDehydDistType8 < 2"',
        'variable vDehydBound10 atom "c_cDehydDistType10 < 2"',
        'variable vDehydBound9 atom "c_cDehydDistType9 < 2"',
        'variable vDehydBound11 atom "c_cDehydDistType11 < 2"',
        #"compute cPairs all pair/local id1 id2 dist",
        #"dump dPairs all local 100 dump.neigh c_cPairs[*]",
        f"dump 2 all local {dump_steps} {spath}/bonds.xyz c_cBonds[*]",
        #"fix fDumpChemistry all ave/time 1 1 20000 f_fReact[1] file reactions.txt",
        '''dump_modify 2 format line "%.0f %.0f %.0f"''',
        "fix twodim all enforce2d",
        "timestep 0.005" #was 0.01 for lower eps
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
        "boundary p p p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/restart_config.dat",
        f"molecule monomer_template {spath}/monomer.dat",
        "group moving id > 0",
        "group none empty",
        "group mol1and2 molecule 1 2",
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
        #"fix xwall_low all wall/reflect xlo EDGE",
        #"fix xwall_high all wall/reflect xhi EDGE",
        #"fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
        "minimize 1.0e-4 1.0e-6 100 1000",
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

def graph_new(positions, molecules,types,atom_ids,cutoff_distance,num_of_molecules,xlo,xhi,ylo,yhi, which_simulation,binding_dump_mode = False):
    bonds,atoms_data,_ , bonded_molecules, molecules, side_interacting_dict = get_bonds(positions, molecules,types,atom_ids,cutoff_distance,xlo,xhi,ylo,yhi,which_simulation, binding_dump_mode )
    clusters = find_clusters(range(1,num_of_molecules+1),bonds)
    #print(clusters)

    # Keep clusters if:
    # - they are longer than 1 (assumed always valid), or
    # - the single element is in mol
    # this is due to bug in the clustering algorithm
    #molecules = lmp.numpy.extract_atom("molecule")
    mol_set = set(molecules)
    filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1 or cluster[0] in mol_set]
    #print('Filtered: ' + str(filtered_clusters))

    return filtered_clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict

def get_fullinterface_clusters(lmp, which_simulation='littleguys'):
    
    L= lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    pos = lmp.numpy.extract_atom("x")
    #mol = lmp.numpy.extract_atom("molecule")
    atom_ids = lmp.numpy.extract_atom("id")
    mol = (atom_ids-1)//8
    types = lmp.numpy.extract_atom("type")
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo  #Lower bounds
    xhi, yhi, zhi = boxhi  # Upper bounds
    num_of_molecules= int(len(mol)/8) if len(mol) > 0 else 0
    cutoff = 1.2
    
    clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict  = graph_new(pos, mol, types,atom_ids,cutoff, num_of_molecules,xlo,xhi,ylo,yhi,which_simulation, binding_dump_mode = True)
    G, clusters_graph, edge_molecules =analyze_bonds(bonds)
    
    return G, clusters_graph, edge_molecules, side_interacting_dict, atoms_data, bonds, bonded_molecules, molecules

def analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis, sim_dir):
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
    num_of_molecules= int(len(mol)/8) if len(mol) > 0 else 0
    
    #G, clusters_graph, edge_molecules, side_interacting_dict_graph = get_fullinterface_clusters(lmp, which_simulation)
    
    hydrolysed_atom_idx = [i for i, atom_type in enumerate(types) if atom_type in [8]]
    hydrolysed_mol_ids = set(mol[hydrolysed_atom_idx])
    for monomer_id in hydrolysed_mol_ids:
        if monomer_id not in hydrolysis:
            hydrolysis[monomer_id] = step
            
    free_monomers = []
    left_monomers_num = 0
    right_monomers_num = 0
    clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict  = graph_new(pos, mol, types,atom_ids,cutoff, num_of_molecules,xlo,xhi,ylo,yhi,which_simulation)
    #print(bonds)
    max_length = max([len(sublist) for sublist in clusters], default=0) #ERRORS OUT HERE CAUSE I FILTER MONOMERS OUT

    #get most right and most left point of polymer
    max_cluster = max(clusters, key=len, default=[])#gets largest polymer
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
    end_monomers = find_endpoints(bonds) #find mol ids for molecules that only have one bond - at the end of polymer
    end_mons_max_cluster = [mon for mon in end_monomers if mon in max_cluster]
    if len(end_mons_max_cluster) < 2:
        print("Warning: Less than 2 endpoints found in the largest cluster. Check the bonding structure.")
        print("Endpoints found:", end_mons_max_cluster)
        print("Max cluster:", max_cluster)
        print("Bonds:", bonds)
        print('Step:', step)
        print('Simulation dir:', sim_dir)
        com_0 = np.array([0,0])
        com_1 = np.array([0,0])
        unit_vector_polymer_direction = np.array([0,1])
    else:
        com_0 = center_of_mass(atoms_data, target_mol_id = end_mons_max_cluster[0])
        com_1 = center_of_mass(atoms_data, target_mol_id = end_mons_max_cluster[1])
    
        vector_polymer_direction = com_1 - com_0
        unit_vector_polymer_direction = vector_polymer_direction / np.linalg.norm(vector_polymer_direction)
    
    #find positions of the center of circle which defines the monomer pool region
    center_circle_left = com_0 
    center_circle_right = com_1 
    end_circle_left = com_0 - 0.7*unit_vector_polymer_direction * radius_region_pool
    end_circle_right = com_1 + 0.7*unit_vector_polymer_direction * radius_region_pool
            
    monomers_on_the_left = []
    monomers_on_the_right = []
    monomers_outside = []

    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    for monomer in free_monomers:
        if monomer in mol:
            #print('Step:', step)
            #print('Free monomer:', monomer)
            com = center_of_mass(atoms_data, target_mol_id = monomer)

            if np.linalg.norm(com - center_circle_left) < radius_region_pool:
                left_monomers_num+=1
                monomers_on_the_left.append(monomer)
                #print('Free monomers on the left:', left_monomers_num)
                #print('Monomer on left:', monomer)
                #print('COM min:',x_min)
            elif np.linalg.norm(com - center_circle_right) < radius_region_pool:
                right_monomers_num+=1
                monomers_on_the_right.append(monomer)
                #print('Free monomers on the right:', right_monomers_num)
                #print('Monomer on right:', monomer)
                #print('COM ax:',x_max)
            else:
                monomers_outside.append(monomer)

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

    return left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right,monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right

def analyse_simulation_monomerlife_fullinterface(lmp, cutoff, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis):
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
    num_of_molecules= int(len(mol)/8) if len(mol) > 0 else 0
    
    G, clusters, edge_molecules, side_interacting_dict, atoms_data, bonds, bonded_molecules, molecules = get_fullinterface_clusters(lmp, which_simulation)
    
    hydrolysed_atom_idx = [i for i, atom_type in enumerate(types) if atom_type in [8]]
    hydrolysed_mol_ids = set(mol[hydrolysed_atom_idx])
    for monomer_id in hydrolysed_mol_ids:
        if monomer_id not in hydrolysis:
            hydrolysis[monomer_id] = step
            
    free_monomers = []
    left_monomers_num = 0
    right_monomers_num = 0
    #clusters,atoms_data,bonds, bonded_molecules, molecules, side_interacting_dict  = graph_new(pos, mol, types,atom_ids,cutoff, num_of_molecules,xlo,xhi,ylo,yhi,which_simulation)
    #print(bonds)
    max_length = max([len(sublist) for sublist in clusters], default=0) #ERRORS OUT HERE CAUSE I FILTER MONOMERS OUT

    #get most right and most left point of polymer
    max_cluster = max(clusters, key=len, default=[])#gets largest polymer
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
    end_monomers = find_endpoints(bonds) #find mol ids for molecules that only have one bond - at the end of polymer
    end_mons_max_cluster = [mon for mon in end_monomers if mon in max_cluster]
    if len(end_mons_max_cluster) < 2:
        print("Warning: Less than 2 endpoints found in the largest cluster. Check the bonding structure.")
        print("Endpoints found:", end_mons_max_cluster)
        print("Max cluster:", max_cluster)
        print("Bonds:", bonds)
        print('Step:', step)
        print('Simula')
        com_0 = [0,0]
        com_1 = [0,0]
        unit_vector_polymer_direction = [0,1]
    else:
        com_0 = center_of_mass(atoms_data, target_mol_id = end_mons_max_cluster[0])
        com_1 = center_of_mass(atoms_data, target_mol_id = end_mons_max_cluster[1])
    
        vector_polymer_direction = com_1 - com_0
        unit_vector_polymer_direction = vector_polymer_direction / np.linalg.norm(vector_polymer_direction)
    
    #find positions of the center of circle which defines the monomer pool region
    center_circle_left = com_0 
    center_circle_right = com_1 
    end_circle_left = com_0 - 0.7*unit_vector_polymer_direction * radius_region_pool
    end_circle_right = com_1 + 0.7*unit_vector_polymer_direction * radius_region_pool
            
    monomers_on_the_left = []
    monomers_on_the_right = []
    monomers_outside = []

    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    for monomer in free_monomers:
        if monomer in mol:
            #print('Step:', step)
            #print('Free monomer:', monomer)
            com = center_of_mass(atoms_data, target_mol_id = monomer)

            if np.linalg.norm(com - center_circle_left) < radius_region_pool:
                left_monomers_num+=1
                monomers_on_the_left.append(monomer)
                #print('Free monomers on the left:', left_monomers_num)
                #print('Monomer on left:', monomer)
                #print('COM min:',x_min)
            elif np.linalg.norm(com - center_circle_right) < radius_region_pool:
                right_monomers_num+=1
                monomers_on_the_right.append(monomer)
                #print('Free monomers on the right:', right_monomers_num)
                #print('Monomer on right:', monomer)
                #print('COM ax:',x_max)
            else:
                monomers_outside.append(monomer)

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

    return left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right,monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right

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

def run(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps, eps, hydrolysis_probability, epsilon_top_final, which_simulation, radius_region_pool = 16, delta_L=25, delta=10): #circle radius was 8, doluting it to 10
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
        setup_simulation_littleguys(lmp, sim_dir, seed, dump_steps, eps, hydrolysis_probability, epsilon_top_final)

    list_of_binding_events = dict()
    list_of_hydrolyzed_molecules = dict()
    unfixed_mols = []
    hydrolysis = dict()
    monomer_birth = dict()
    monomer_death = dict()
    dumped_hydrolysis_events = []
    
    #dict_clusters_and_bonds = dict()
    
    output_file = "clusters_log.txt"

    # Create/overwrite the file at the start
    with open(sim_dir + "/" + output_file, "w") as f:
        f.write("###LAMMPS simulation started\n")
        f.write("###Step, Clusters, Bonds, Side Interacting\n")
        
    with open(sim_dir + "/clusters_fullinterface_log.txt", "w") as f:
        f.write("###LAMMPS simulation started\n")
        f.write("###Step, Clusters, Edge molecules, Side Interacting\n")

    with open(sim_dir + "/hydrolysis_log.txt", "w") as f:
        f.write("###LAMMPS simulation started\n")
        f.write("###Step, Hydrolyzed Molecules\n")
        
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
    left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis, sim_dir)
    G, clusters_graph, edge_molecules, side_interacting_dict_graph, atoms_data, bonds, bonded_molecules, molecules = get_fullinterface_clusters(lmp, which_simulation)
    max_polymer_fullinterface = max(clusters_graph, key=len, default=[])
    
    max_polymer = max(clusters, key=len, default=[])
    counter = 1
    counter_left = 1
    counter_right = 1
    flag_addition_left = False
    flag_addition_right = False
    recenter_activated = False
    #lmp.command('fix recenter_all all recenter 0 2.5 0')
    """
    # convert to string for LAMMPS
    max_polymer = max(clusters, key=len)
    cluster_max_pol_list_int = [int(m) for m in max_polymer]
    mol_str = " ".join(map(str, cluster_max_pol_list_int))
    # define the group
    lmp.command(f"group polymer_group_stop_rotation molecule {mol_str}")
    lmp.command('fix stop_rotation_fix polymer_group_stop_rotation momentum 1 angular')
    """
    #current_mol_number = 10 #random big number, higher than 2 to make the bug of fix deopsit not depoisting at the exact step of polymerization disappear

    #lmp.command("region              box_region block EDGE EDGE EDGE EDGE EDGE EDGE units box")
    
    #activate the fix deposit to keep molecule ids non overlapping and then activate at the right time and right region using var and region
    
    
    ##set the equal style variable to 0
    #set initial regions so the fix can run
    # lmp.command('region              insertion_region_left block 0.0 0.0 0.0 0.0 0.0 0.0 units box')
    # lmp.command('region              insertion_region_right block 0.0 0.0 0.0 0.0 0.0 0.0 units box')
    
    # lmp.command('variable            insertmonomerleftflag equal 0')
    # lmp.command('fix                 add_mons_left moving deposit %d 0 1 %d mol monomer_template region insertion_region_left near 2 var v_insertmonomerleftflag id next'%(1,seed))
    # lmp.command('variable            insertmonomerrightflag equal 0')
    # lmp.command('fix                 add_mons_right moving deposit %d 0 1 %d mol monomer_template region insertion_region_right near 2 var v_insertmonomerrightflag id next'%(1,seed))
    
    
    # Iterate through loops
    for i in range(loops):
    #for i in tqdm(range(loops)):
        # Run for run_interval steps
        
        #signal.alarm(300) #give 300 seconds for the loop to run
        
        G, clusters_graph, edge_molecules, side_interacting_dict_graph, atoms_data, bonds, bonded_molecules, molecules = get_fullinterface_clusters(lmp, which_simulation)
        left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events,radius_region_pool, which_simulation, hydrolysis, sim_dir)
        step = lmp.get_thermo('step')
        max_polymer_new = max(clusters, key=len, default=[])
        if sorted(max_polymer_new) != sorted(max_polymer):
            step = lmp.get_thermo('step')
            with open(sim_dir + "/clusters_log.txt", "a") as f: #save the clusters and bonds at each step in a log file
                f.write(f"{step}, {clusters}, {bonds}, {side_interacting_dict}\n")
                
        max_polymer_fullinterface_new = max(clusters_graph, key=len, default=[])
        if sorted(max_polymer_fullinterface_new) != sorted(max_polymer_fullinterface):
            step = lmp.get_thermo('step')
            with open(sim_dir + "/clusters_fullinterface_log.txt", "a") as f: #save the clusters and bonds at each step in a log file
                f.write(f"{step}, {clusters_graph}, {edge_molecules}, {side_interacting_dict_graph}\n")
                
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
                hydrolysis.pop(monomer, None)  # Remove from hydrolysis dictionary if it dies
                    
        for key in hydrolysis:
            monomer, hydrolysis_step = key, hydrolysis[key] 
            if (monomer,hydrolysis_step) not in dumped_hydrolysis_events:
                with open(sim_dir + "/hydrolysis_log.txt", "a") as f:
                    f.write(f"{hydrolysis_step}, {monomer}\n")
                dumped_hydrolysis_events.append((monomer,hydrolysis_step))
            
                
        
        #dict_clusters_and_bonds[step] = (clusters, bonds)
        #step = lmp.get_thermo('step')
        #print('Clusters at step %d: %s' % (step, str(clusters)))
        #print('Monomers on the left:', monomers_on_the_left)
        #print('Monomers on the right:', monomers_on_the_right)
        max_cluster_length = max([len(sublist) for sublist in clusters], default=0)
        if max_cluster_length<2:
            lmp.close()
            print('POLYMER GOT DOWN TO LESS THAN A DIMER IN ' + str(step) + 'step, at path' + str(sim_dir) + ' - LEAVING SIMULATION')
            return sucess_of_run
        
        """
        # convert to string for LAMMPS
        lmp.command('unfix stop_rotation_fix')
        lmp.command('group polymer_group_stop_rotation delete')
        max_polymer = max(clusters, key=len, default=[])
        cluster_max_pol_list_int = [int(m) for m in max_polymer]
        mol_str = " ".join(map(str, cluster_max_pol_list_int))
        # define the group
        lmp.command(f"group polymer_group_stop_rotation molecule {mol_str}")
        lmp.command('fix stop_rotation_fix polymer_group_stop_rotation momentum 1 angular')
        """
        ##step = lmp.get_thermo('step')
        #print('ROTATING POLYMER AT STEP' + str(step))
        #rotate_polymer(lmp,cutoff, print_mode,list_of_binding_events, which_simulation)

        # Analyse the system
        #L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        #step = lmp.get_thermo('step')
        #pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
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
        
        #left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict  = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis)
        #print(clusters)
        if max([len(c) for c in clusters], default=0) >= 2:
            com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(lmp, clusters)
        else:
            lmp.close()
            print('POLYMER GOT DOWN TO LESS THAN A DIMER - LEAVING SIMULATION')
            success_of_run = True
            break
        bounded_monomers = [key for key in list_of_binding_events]
        hydrolyzed_monomers = [key for key in list_of_hydrolyzed_molecules]

        if len(monomers_on_the_left) < nMonsSol:
            nmonomers = nMonsSol-len(monomers_on_the_left)
            #lmp.command("region              sphere_left_%d sphere %f %f %f %f units box" %(counter,center_circle_left[0], center_circle_left[1], 0.0, radius_region_pool))
            #lmp.command("region              insertion_region_left_%d intersect 2 box_region sphere_left_%d" %(counter,counter))
            ###OLD IMPLEMENTATION
            lmp.command("region              insertion_region_left_%d block %f %f %f %f %f %f units box" %(counter_left,end_circle_left[0],end_circle_left[0],end_circle_left[1] , end_circle_left[1], 0.0,0.0))
            #lmp.command('variable                 insertmonomerleftflag equal 1')
            lmp.command('fix                 add_mons_left_%d moving deposit %d 0 1 %d mol monomer_template region insertion_region_left_%d near 2'%(counter_left,nmonomers,seed,counter_left))
            #########
            #lmp.command("region              insertion_region_right_%d block %f %f 0 0 0 0 units box" %(counter,com_x_max_mol+delta , com_x_max_mol+delta))
            #lmp.command('fix                 add_mons_right_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d '%(counter,nmonomers,seed,counter))
            flag_addition_left = True

            #delete previous region and fix
            # if counter_left>1:
            #     lmp.command('unfix add_mons_left_%d' % (counter_left-1))
            #     lmp.command('region insertion_region_left_%d delete' % (counter_left-1))

            counter +=1
            counter_left +=1
            #lmp.command('run 1')
            #lmp.command('variable                 insertmonomerleftflag equal 0')
            #left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict  = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis)
        if len(monomers_on_the_right) < nMonsSol:
            nmonomers = nMonsSol-len(monomers_on_the_right)
            #lmp.command("region              sphere_right_%d sphere %f %f %f %f units box" %(counter,center_circle_right[0], center_circle_right[1], 0.0, radius_region_pool))
            #lmp.command("region              insertion_region_right_%d intersect 2 box_region sphere_right_%d" %(counter,counter))
            ######OLD IMPLEMENTATION
            lmp.command("region              insertion_region_right_%d block %f %f %f %f %f %f units box" %(counter_right,end_circle_right[0],end_circle_right[0],end_circle_right[1] , end_circle_right[1], 0.0,0.0))
            #lmp.command('variable                 insertmonomerrightflag equal 1')
            lmp.command('fix                 add_mons_right_%d moving deposit %d 0 1 %d mol monomer_template region insertion_region_right_%d near 2'%(counter_right,nmonomers,seed,counter_right))
            ##############
            #lmp.command("region              insertion_region_right_%d block %f %f 0 0 0 0 units box" %(counter,com_x_max_mol+delta , com_x_max_mol+delta))
            #lmp.command('fix                 add_mons_right_%d moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d '%(counter,nmonomers,seed,counter))
            flag_addition_right = True
            #delete previous region and fix
            # if counter_right>1:
            #     lmp.command('unfix add_mons_right_%d' % (counter_right-1))
            #     lmp.command('region insertion_region_right_%d delete' % (counter_right-1))
            
            counter +=1
            counter_right +=1
            #lmp.command('run 1')#to insert the molecule
            #lmp.command('variable                 insertmonomerleftflag equal 0')
            #lmp.command('run 5')
            #left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict  = analyse_simulation(lmp, cutoff, print_mode,list_of_binding_events,radius_region_pool, which_simulation, hydrolysis)


        left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict, end_circle_left, end_circle_right   = analyse_simulation(lmp, 2, print_mode,list_of_binding_events,radius_region_pool, which_simulation, hydrolysis, sim_dir)
        #HARD CODED ABOVE - shifted cutoff to be 2 to detect monomers as free only when a bit far away from polymer
        #delete unneccessary moelcules
        polymer_number = sum(1 for sublist in clusters if len(sublist) > 1)
        max_polymer_size = max(clusters, key=len, default=[]) #if breaking occurs keep only the longest filament
        monomer_number = sum(1 for sublist in clusters if len(sublist) == 1 and sublist[0] in mol)
        flattened_mols = [item for sublist in clusters for item in sublist]
        molecules_to_delete = []

        left_deleted = False
        right_deleted = False
        if polymer_number>1: #if more than one polymer exists-this can happen through fragmentation of the polymer, delete other polymers+ monomers on that side only
            #print('Polymer number:' + str(polymer_number))
            #print('Monomer number:' + str(monomer_number))
            for molecule in flattened_mols:
                if molecule not in max_polymer_size:
                    for cluster in clusters:
                        if len(cluster)>1 and molecule in cluster:
                            molecules_to_delete.append(molecule)
                         
        #if a hydrolyzed moonomer detached, delete it   
        for molecule in free_monomers:
            if molecule in list_of_hydrolyzed_molecules:
                molecules_to_delete.append(molecule)
                
        for molecule in monomers_outside:
            molecules_to_delete.append(molecule)
                
        #lmp.command('run 5')
        #left_monomers_num, right_monomers_num, monomers_on_the_left, monomers_on_the_right, monomers_free_num, list_of_binding_events, molecules, clusters, atoms_data, free_monomers, monomers_outside, center_circle_left, center_circle_right, bonds, side_interacting_dict  = analyse_simulation(lmp, 2, print_mode,list_of_binding_events, radius_region_pool, which_simulation, hydrolysis)
               
        """         
        i = 0
        while monomer_number>nMonsSol: #if too many monomers exist - can happen through end depolymerisation, delete all monomers on this ONE side
            molecules_to_delete.append(free_monomers[i])
            i+=1
            monomer_number-=1
        """
                
        mol = lmp.numpy.extract_atom("molecule")
        molecules_to_delete = list(set(molecules_to_delete)) #remove duplicates
        
        deleted = False
        for mols in molecules_to_delete:
            if mols in mol:
                deleted = True
                lmp.command('group mol_%d molecule %d'%(mols,mols))
                lmp.command('delete_atoms group mol_%d'%(mols))
                lmp.command('group mol_%d delete '%(mols))
                #deleted_atoms.append(mols)
                if mols in list_of_binding_events: #since LAMMPS recycles molecule IDs, we remove the information we had on the previous monomer during its deletion
                    list_of_binding_events.pop(mols)
                if mols in list_of_hydrolyzed_molecules:
                    list_of_hydrolyzed_molecules.pop(mols)
                    
        max_polymer = max_polymer_new
        max_polymer_fullinterface = max_polymer_fullinterface_new
        lmp.command('run %d'%(run_interval))
        
        #delete the fix and region for monomer addition after the run interval
        if flag_addition_left and counter_left>1:
            lmp.command('unfix add_mons_left_%d' % (counter_left-1))
            lmp.command('region insertion_region_left_%d delete' % (counter_left-1))
            flag_addition_left = False
        if flag_addition_right and counter_right>1:
            lmp.command('unfix add_mons_right_%d' % (counter_right-1))
            lmp.command('region insertion_region_right_%d delete' % (counter_right-1))
            flag_addition_right = False
        #signal.alarm(0) #cancel the alarm
        
    # Clean up
    lmp.close()
    sucess_of_run = True
    return sucess_of_run


def write_info_file(path, nMonsSol, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps, epsilon_top_final, hydrolysis_probability):
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
    info_file.write(f'epsilon post hydrolysis: {epsilon_top_final}\n')
    info_file.write(f'hydrolysis probability: {hydrolysis_probability}\n')
    info_file.write(f'dump_steps: {dump_steps}\n')
    info_file.write(f'debug_mode: {debug_mode}\n')
    info_file.write(f'print_mode: {print_mode}\n')
    info_file.close()

def single_seed(spath, nMonsSol, seed, loops, run_interval, debug_mode, dump_steps, print_mode, eps,cutoff, hydrolysis_probability,epsilon_top_final,which_simulation):
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

    # Write lammps input file
    #print('Writing input files:')
    #dt, epsilon_bottom, epsilon_top,epsilon_top_final, general_cutoff, xlo, xhi, ylo, yhi = module_LAMMPSinputs_treadmilling.write_input_files(spath)
    #cutoff_bond = general_cutoff-0.2
    sucess_of_run = run(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps,eps, hydrolysis_probability, epsilon_top_final, which_simulation)

    #end_time = time.time()
    #print()
    #print('Simulation done!')
    
    return sucess_of_run

def run_simulation(path, seed, nMonsSol=1, run_interval=1, run_steps=1000000000, dump_steps=10000, eps = 100, cutoff =1.1, debug_mode=True, print_mode=False, Nseeds=1, hydrolysis_probability=0.00001, epsilon_top_final = 8,which_simulation = 'littleguys'):
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")
    
    trial = 0
    success_of_run = False
    while not success_of_run and trial<1:
        #seeds = np.random.randint(10000, 99999, Nseeds)
        #while len(np.unique(seeds)) < Nseeds:
        #    seeds = np.random.randint(10000, 99999, Nseeds)

        #seed = seeds[0]

        # Create the simulation folders
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Write the information file
        write_info_file(path, nMonsSol, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps, epsilon_top_final, hydrolysis_probability)

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
            trial += 1
            success_of_run = single_seed(path, nMonsSol, seed, loops, run_interval, debug_mode, dump_steps, print_mode, eps,cutoff, hydrolysis_probability,epsilon_top_final,which_simulation)
            if success_of_run == True:
                print('Simulation successfully finished at ' + str(path))
        except Exception as e:
            print('ERROR' + str(e) +' IN SIMULATION AT ' + str(path))

#run_simulation(path='/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling_LG/new_runs_23_10_circle_centered_on_filament_nofullinterfaceanalyse/radius12/hydrolysis_prob_1e-06_eps_final_8_eps_nonhydrolyzed_100/seed_15002', seed = 15002)