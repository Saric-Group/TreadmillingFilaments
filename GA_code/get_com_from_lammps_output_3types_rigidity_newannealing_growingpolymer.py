import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pickle
from deap import base, creator, tools
from functools import lru_cache

# Cache for parsed dump files to avoid re-reading the same file multiple times
_dump_file_cache = {}

# Define the problem
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMin, history_index=None, type_rnd_number=-1)  # Use the new FitnessMin class

def save_deme_checkpoint(populations, halloffame, logbook, generation,
                         best_individuals_per_gen, worst_individuals_per_gen,
                         filename):
    """
    Save the populations, HallOfFame, logbook, and best/worst individuals for the checkpoint.
    """
    # Use a temporary copy of populations, since HallOfFame references individuals
    populations_copy = [[(list(ind), {"gen_id": getattr(ind, "gen_id", None), "work_dir": getattr(ind, "work_dir", None), "history_index": getattr(ind, "history_index", None), "term1": getattr(ind, "term1", 0), "term2": getattr(ind, "term2", 0), "term3": getattr(ind, "term3", 0), "term4": getattr(ind, "term4", 0), "term5": getattr(ind, "term5", 0), "term6": getattr(ind, "term6", 0), "term7": getattr(ind, "term7", 0), "type_rnd_number": getattr(ind, "type_rnd_number", -1)}) for ind in deme] for deme in populations]
    #populations_copy = [[(list(ind), {"gen_id": getattr(ind, "gen_id", None), "work_dir": getattr(ind, "work_dir", None), "history_index": getattr(ind, "history_index", None), "term1": getattr(ind, "term1", 0), "term2": getattr(ind, "term2", 0), "term3": getattr(ind, "term3", 0), "term4": getattr(ind, "term4", 0), "term5": getattr(ind, "term5", 0), "term6": getattr(ind, "term6", 0), "term7": getattr(ind, "term7", 0), "type_rnd_number": getattr(ind, "type_rnd_number", -1)}) for ind in deme] for deme in populations]


    with open(filename, "wb") as cp_file:
        pickle.dump({
            'populations': populations_copy,  # Save copies, not direct references
            'halloffame': halloffame,  # Pickle HallOfFame separately
            'logbook': logbook,  # Logbook is simple and should pickle fine
            'generation': generation,
            'best_individuals_per_gen': best_individuals_per_gen,
            'worst_individuals_per_gen': worst_individuals_per_gen
        }, cp_file)
    print(f"Checkpoint saved at generation {generation}")


def load_deme_checkpoint(filename):
    if os.path.exists(filename):
        print(f"Loading checkpoint from {filename}...")
        try:
            with open(filename, "rb") as cp_file:
                cp_data = pickle.load(cp_file)
            
            # Convert raw lists back into DEAP individuals
            def _restore_individual(ind_data):
                raw_ind, attrs = ind_data
                ind = creator.Individual(raw_ind)

                # Restore all attributes
                for key, value in attrs.items():
                    if key == 'gen_id' or key =='work_dir' or key == 'history_index':
                        setattr(ind, key, value)
                    else:
                        setattr(ind, key, 0)

                # Restore only the first 3 attributes
                #for key in list(attrs.keys())[:3]:
                #    setattr(ind, key, attrs[key])

                return ind


            populations = [
                [_restore_individual(ind_data) for ind_data in deme]
                for deme in cp_data['populations']
            ]



            print(f"Checkpoint successfully loaded from generation {cp_data['generation']}")
            return (populations,
                    cp_data['halloffame'],
                    cp_data['logbook'],
                    cp_data['generation'],
                    cp_data.get('best_individuals_per_gen', []),
                    cp_data.get('worst_individuals_per_gen', []))
        except Exception as e:
            print(f"Error while loading checkpoint: {e}")
            return None, None, None, 0, [], []
    else:
        print(f"Checkpoint file not found at {filename}")
        return None, None, None, 0, [], []

def find_last_timestep(file_path):
    """
    Reads the OVITO output file and finds the last timestep.

    Parameters:
        file_path (str): Path to the OVITO output file.

    Returns:
        int: The last timestep in the file.
    """
    last_timestep = None

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("ITEM: TIMESTEP"):
                    # The next line after "ITEM: TIMESTEP" contains the timestep value
                    last_timestep = int(next(file).strip())
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except ValueError:
        print(f"Error: Invalid data format in file: {file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return last_timestep

def _parse_all_timesteps(filename):
    """
    Parses all timesteps from a LAMMPS dump file once and caches the result.
    This avoids re-reading the file for each timestep.
    
    Parameters:
        filename (str): Path to the dump file.
    
    Returns:
        dict: Maps timestep_index to (atoms_data, box_bounds) tuples.
    """
    if filename in _dump_file_cache:
        return _dump_file_cache[filename]
    
    timestep_start = "ITEM: TIMESTEP"
    timesteps_dict = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Identify all timesteps in the file
    timestep_indices = []
    for i, line in enumerate(lines):
        if line.strip() == timestep_start:
            timestep_indices.append(i)
    
    if not timestep_indices:
        raise ValueError("No timesteps found in the file.")
    
    # Parse each timestep
    for ts_idx, ts_start_idx in enumerate(timestep_indices):
        # Parse box bounds
        box_bounds_line = ts_start_idx + 5
        box_bounds = []
        for i in range(2):  # Read 2 lines for x, y bounds
            bounds = tuple(map(float, lines[box_bounds_line + i].strip().split()))
            box_bounds.extend(bounds)
        box_bounds = tuple(box_bounds)
        
        # Parse the number of atoms
        num_atoms_line = ts_start_idx + 3
        num_atoms = int(lines[num_atoms_line].strip())
        
        # Parse atoms data
        atoms_data_start = ts_start_idx + 9
        atoms_data = []
        for line in lines[atoms_data_start:atoms_data_start + num_atoms]:
            parts = line.strip().split()
            if len(parts) < 5:
                raise ValueError("Malformed atoms data line: " + line)
            atom_id = int(parts[0])
            mol_id = int(parts[1])
            atom_type = int(parts[2])
            x, y = map(float, parts[3:5])
            atoms_data.append((atom_id, mol_id, atom_type, x, y))
        
        timesteps_dict[ts_idx] = (atoms_data, box_bounds)
    
    # Cache the result
    _dump_file_cache[filename] = timesteps_dict
    return timesteps_dict


def specific_atom_ID_for_mol(atoms_data, target_mol_id, atom_ID):
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
    positions = [(x, y) for atom_id, mol_id, atom_type, x, y in atoms_data if mol_id == target_mol_id and atom_id == atom_ID]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])


def read_lammps_dump(filename, timestep_index):
    """
    Reads a LAMMPS dump file and extracts atoms' data for a specific timestep.
    Uses caching to avoid re-reading the entire file for multiple timestep queries.
    
    Parameters:
        filename (str): Path to the dump file.
        timestep_index (int): Index of the timestep to read (-1 for the last timestep).
    
    Returns:
        atoms_data (list of tuples): Each tuple contains (atom_id, mol_id, atom_type, x, y, z).
        box_bounds (tuple): Box boundaries as (xlo, xhi, ylo, yhi, zlo, zhi).
    """
    # Load and cache all timesteps from this file
    timesteps_dict = _parse_all_timesteps(filename)
    
    # Determine which timestep to use
    num_timesteps = len(timesteps_dict)
    if timestep_index == -1:
        timestep_index = num_timesteps - 1
    if timestep_index >= num_timesteps:
        raise IndexError(f"Requested timestep_index {timestep_index} is out of range, at {filename}")
    
    # Retrieve the cached data
    atoms_data, box_bounds = timesteps_dict[timestep_index]
    
    return atoms_data, box_bounds





def apply_pbc(x, y, xlo, xhi, ylo, yhi):
    """Apply periodic boundary conditions to coordinates."""
    Lx = xhi - xlo
    Ly = yhi - ylo
    
    if x < xlo:
        x += Lx
    elif x >= xhi:
        x -= Lx

    if y < ylo:
        y += Ly
    elif y >= yhi:
        y -= Ly
    
    return x, y

def calculate_center_of_mass(atoms_data, box_bounds):
    xlo, xhi, ylo, yhi = box_bounds
    
    # Organize atoms by molecule
    molecules = {}
    for atom_id, mol_id, atom_type, x, y in atoms_data:
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append((atom_id, x, y))
    
    centers_of_mass = {}
    for mol_id, atoms in molecules.items():
        num_atoms = len(atoms)
        if num_atoms == 0:
            continue
        
        # Extract positions of the molecule
        atom_ids, xs, ys = zip(*atoms)
        
        L = xhi -xlo
        borderx = False #check if breaks on x border
        for j,x1 in enumerate(xs):
            for x2 in xs[j+1:]:
                dist = np.abs(x1-x2)
                if dist>L/2:
                    borderx = True
                    
        bordery = False #check if breaks on y border
        for i,y1 in enumerate(ys):
            for y2 in ys[i+1:]:
                dist = np.abs(y1-y2)
                if dist>L/2:
                    bordery = True       
        
        xs_list = list(xs)
        if borderx: #if breaks on x border shift the cut off ones by L either to left or right
            if sum(xs_list) >= 0:
                for k1,x in enumerate(xs_list):
                    if x<0:
                        xs_list[k1] += L
            elif sum(xs_list) < 0:
                for k2,x in enumerate(xs_list):
                    if x>0:
                        xs_list[k2] -= L
        xs = tuple(xs_list)
                
        ys_list = list(ys)
        if bordery: #if breaks on x border shift the cut off ones by L either to left or right
            if sum(ys_list) >= 0:
                for k3,y in enumerate(ys_list):
                    if y<0:
                        ys_list[k3] += L
            elif sum(ys_list) < 0:
                for k4,y in enumerate(ys_list):
                    if y>0:
                        ys_list[k4] -= L
        ys = tuple(ys_list)  
        
        # Calculate center of mass assuming virtual extended box
        com_x = sum(xs) / num_atoms
        com_y = sum(ys) / num_atoms
        
        # Apply periodic boundary conditions to COM
        com_x, com_y = apply_pbc(com_x, com_y, xlo, xhi, ylo, yhi)
        
        centers_of_mass[mol_id] = (com_x, com_y)
    
    return centers_of_mass

def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds):
    xlo, xhi, ylo, yhi = box_bounds
    for atom1 in molecule1:
        for atom2 in molecule2:
            x1, y1, atom_type1 = atom1[1], atom1[2], atom1[3]
            x2, y2, atom_type2 = atom2[1], atom2[2], atom2[3]
            
            dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
            dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < cutoff :
                if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1):
                    return True
    return False

def plot_centers_of_mass_and_bonds(centers_of_mass, bonds, box_bounds,simulation_folder):
    xlo, xhi, ylo, yhi = box_bounds
    x_coords = [com[0] for com in centers_of_mass.values()]
    y_coords = [com[1] for com in centers_of_mass.values()]
    print('X COORDS:')
    print(x_coords)
    print('Y COORDS:')
    print(y_coords)

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')
    
    for bond in bonds:
        com1 = centers_of_mass[bond[0]]
        com2 = centers_of_mass[bond[1]]
        plt.plot([com1[0], com2[0]], [com1[1], com2[1]], 'r-')
    
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Num of Bonds' + str(len(bonds)))
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(simulation_folder,'coms_and_bonds_last_step'))
    plt.close()
    #plt.show()

# Read the dump file
def get_bonds(output_file_path,timestep_index):
    atoms_data, box_bounds = read_lammps_dump(output_file_path,timestep_index)
    
    molecules = {}
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y = atom
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append((atom_id, x, y,atom_type))
        
    # Find touching molecules and add bonds
    cutoff_distance = 1.2
    bonds = []
        
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds): #if theyre touching move
                    bonds.append((mol_id1, mol_id2))
                    
    #print('BONDS BETWEEN:')
    #print(len(bonds))
    #print(bonds)
    return bonds,atoms_data,molecules

def get_com_from_coms(com1,com2):
    com1_x, com1_y = com1
    com2_x, com2_y = com2
    
    comx_total = (com1_x+com2_x)/2
    comy_total = (com1_y+com2_y)/2
    
    return comx_total,comy_total

def get_relative_coms(output_file_positions,com_dump,sim_steps,dump_freq):
    num_of_measurements = int(sim_steps/dump_freq)
    total_measurements = range(num_of_measurements)
    delta_x = []
    delta_y = []
    count = 0
    
    for i in total_measurements:
        bonds,_,_ = get_bonds(output_file_positions,timestep_index=i)
        if len(bonds) !=0:
            for i1,i2 in bonds:
                count+=2 #adding two displacements
                deltax_i1, deltay_i1, deltax_i2, deltay_i2 = extract_com_values(com_dump, target_timestep = i*dump_freq,i1=i1,i2=i2)
                
                delta_x.append(deltax_i1)
                delta_x.append(deltax_i2)
                
                delta_y.append(deltay_i1)
                delta_y.append(deltay_i2)
            
    return delta_x,delta_y,count

def get_relative_com_1_particle(output_file_positions,dump_freq):
    #num_of_measurements = int(sim_steps/dump_freq)
    #total_measurements = range(num_of_measurements) #100 for now

    last_timestep = find_last_timestep(output_file_positions)
    total_measurements = range(int(last_timestep/dump_freq))

    total_delta_r = []
    clusters_over_time = []
    branched_nodes = []
    directionalities=[]
    
    for i in total_measurements:
        bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=i)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        clusters = find_clusters(range(1,num_of_molecules+1),bonds)
        filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules]  # Filter out single-molecule clusters
        clusters_over_time.append(filtered_clusters)
        max_polymer_size = max(filtered_clusters, key=len)
        delta_r = []
        for j in range(1,21):
            com_j = np.zeros(2)
            if j not in molecules:
                delta_r.append(0)
            else:
                com = np.zeros(2)
                for cluster in filtered_clusters:
                    if j in cluster:
                        for molecule in cluster:
                            com += center_of_mass(atoms_data, molecule) 
                        com_j = center_of_mass(atoms_data, j) 
                        if i==total_measurements[-1] and molecule in max_polymer_size:
                            com_mol1 = center_of_mass(atoms_data,max_polymer_size[0])
                            com_mol2 = center_of_mass(atoms_data,max_polymer_size[-1])
                            #vector_polymer_com = com/len(cluster) - com_mol1
                            vector_polymer_com = com_mol2 - com_mol1
                            com_mol1_type1 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=1)
                            com_mol1_type2 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=2)
                            if com_mol1_type1 is None:
                                com_mol1_type1 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=4)
                            if com_mol1_type2 is None:
                                com_mol1_type2 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=5)
                            #try:
                            vector_monomer = com_mol1_type2 - com_mol1_type1
                            #if 1 in cluster:
                            if 1 in cluster:
                                directionalities.append(np.dot(vector_monomer, vector_polymer_com))
                            #except:
                            #    vector_monomer = np.array([0,0])
                            #    print('NO SECOND BINDING SITE AT '+str(output_file_positions))
                        delta_rj = np.linalg.norm(com/len(cluster)-com_j)  #relative com of j - CHECK IF LAMMPS COM RESPECTS PBC
                        delta_r.append(delta_rj)
        total_delta_r.append(delta_r)
    #total_delta_r = np.array(total_delta_r)
    #check if all monomers point in the same direction

    signs = [x > 0 for x in directionalities] #signs of dot product of every monmer with center of mass
    if all(signs):
        prefactor = 1
    else:
        prefactor = -1
    
    shifts_all_particles = []
    for k in range(20):
        #shifts = total_delta_r[:-1,k]-total_delta_r[1:,k]
        shift = total_delta_r[-1][k] - total_delta_r[0][k]
        #shifts = [total_delta_r[i][k] - total_delta_r[i+1][k] for i in range(len(total_delta_r) - 1)]
        shifts_all_particles.append(np.abs(shift))

    #find_fastest_growing_pair(np.array(clusters_over_time,dtype = object),num_of_molecules) #used to find the pair that grew fastest in annealing simulations
        
    max_shift = shifts_all_particles[0]
    max_index = 0

    max_shift = max_shift*prefactor #coupling monomer directionality to polymer directionality
            
    final_clusters = clusters_over_time[-1] #number of clusters in the end of self assembly
    max_shift_per_cluster = []
    max_index_per_cluster = []
    cluster_dynamics = []
    cluster_dynamics.append(clusters_over_time[0])
    filtered_history = clusters_over_time
    initial_size = len(filtered_history[0])
    for t in range(1, len(filtered_history)):
        current_size = len(filtered_history[t])
        if current_size!=initial_size:
            #print(filtered_history[t])
            cluster_dynamics.append(clusters_over_time[t])
            initial_size = current_size
        elif current_size == initial_size:
            clusters1 =  filtered_history[t-1]
            clusters2 =  filtered_history[t]
            for i in np.arange(len(filtered_history[t])):
                if sorted(clusters1[i]) != sorted(clusters2[i]):
                    #print(filtered_history[t])
                    cluster_dynamics.append(clusters_over_time[t])
                    initial_size = current_size       
                    
    return max_shift, cluster_dynamics, clusters_over_time

def get_relative_com_1_particle_binding_GA_monomer_directionality(output_file_positions,dump_freq):
    #num_of_measurements = int(sim_steps/dump_freq)
    #total_measurements = range(num_of_measurements) #100 for now

    last_timestep = find_last_timestep(output_file_positions)
    total_measurements = range(int(last_timestep/dump_freq))

    total_delta_r = []
    clusters_over_time = []
    branched_nodes = []
    directionalities=[]
    
    for i in total_measurements:
        bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=i)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        size_of_monomer = sum(1 for tup in atoms_data if tup[1] == 1) #this is to get max mol ID
        clusters = find_clusters(range(1,num_of_molecules+1),bonds)
        filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules]  # Filter out single-molecule clusters
        clusters_over_time.append(filtered_clusters)
        delta_r = []
        for j in range(1,21):
            com_j = np.zeros(2)
            if j not in molecules:
                delta_r.append(0)
            else:
                com = np.zeros(2)
                for cluster in filtered_clusters:
                    if j in cluster:
                        for molecule in cluster:
                            com += center_of_mass(atoms_data, molecule) 
                        com_j = center_of_mass(atoms_data, j) 
                        if i==total_measurements[-1] and 1 in cluster:
                            com_mol1 = center_of_mass(atoms_data,1)
                            vector_polymer_com = com/len(cluster) - com_mol1
                            atom_id1 = 1 + (j-1)*size_of_monomer #get atom id of first binding site of monomer j
                            atom_id2 = j*size_of_monomer #get atom id of last binding site of monomer j
                            com_molj_first_edge_monomer = specific_atom_ID_for_mol(atoms_data, target_mol_id=j, atom_ID=atom_id1)
                            com_molj_last_edge_monomer = specific_atom_ID_for_mol(atoms_data, target_mol_id=j, atom_ID=atom_id2)
                            #com_mol1_type1 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=1)
                            #com_mol1_type2 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=2)
                            #try:
                            vector_monomer =  com_molj_last_edge_monomer - com_molj_first_edge_monomer 
                            #if 1 in cluster:
                            if 1 in cluster:
                                directionalities.append(np.dot(vector_monomer, vector_polymer_com))
                            #except:
                            #    vector_monomer = np.array([0,0])
                            #    print('NO SECOND BINDING SITE AT '+str(output_file_positions))
                        delta_rj = np.linalg.norm(com/len(cluster)-com_j)  #relative com of j - CHECK IF LAMMPS COM RESPECTS PBC
                        delta_r.append(delta_rj)
        total_delta_r.append(delta_r)
    #total_delta_r = np.array(total_delta_r)
    #check if all monomers point in the same direction

    signs = [x > 0 for x in directionalities] #signs of dot product of every monmer with center of mass
    if all(signs):
        prefactor = 1
    else:
        prefactor = -1
    
    shifts_all_particles = []
    for k in range(20):
        #shifts = total_delta_r[:-1,k]-total_delta_r[1:,k]
        shift = total_delta_r[-1][k] - total_delta_r[0][k]
        #shifts = [total_delta_r[i][k] - total_delta_r[i+1][k] for i in range(len(total_delta_r) - 1)]
        shifts_all_particles.append(np.abs(shift))

    #find_fastest_growing_pair(np.array(clusters_over_time,dtype = object),num_of_molecules) #used to find the pair that grew fastest in annealing simulations
        
    max_shift = shifts_all_particles[0]
    max_index = 0

    max_shift = max_shift*prefactor #coupling monomer directionality to polymer directionality
            
    final_clusters = clusters_over_time[-1] #number of clusters in the end of self assembly
    max_shift_per_cluster = []
    max_index_per_cluster = []
    cluster_dynamics = []
    cluster_dynamics.append(clusters_over_time[0])
    filtered_history = clusters_over_time
    initial_size = len(filtered_history[0])
    for t in range(1, len(filtered_history)):
        current_size = len(filtered_history[t])
        if current_size!=initial_size:
            #print(filtered_history[t])
            cluster_dynamics.append(clusters_over_time[t])
            initial_size = current_size
        elif current_size == initial_size:
            clusters1 =  filtered_history[t-1]
            clusters2 =  filtered_history[t]
            for i in np.arange(len(filtered_history[t])):
                if sorted(clusters1[i]) != sorted(clusters2[i]):
                    #print(filtered_history[t])
                    cluster_dynamics.append(clusters_over_time[t])
                    initial_size = current_size       
                    
    return max_shift, cluster_dynamics, clusters_over_time

def get_relative_com_1_particle_binding_monomer_directionality(output_file_positions,dump_freq):
    #num_of_measurements = int(sim_steps/dump_freq)
    #total_measurements = range(num_of_measurements) #100 for now

    last_timestep = find_last_timestep(output_file_positions)
    total_measurements = range(int(last_timestep/dump_freq))

    total_delta_r = []
    clusters_over_time = []
    branched_nodes = []
    directionalities=[]
    
    for i in total_measurements:
        bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=i)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        size_of_monomer = sum(1 for tup in atoms_data if tup[1] == 1) #this is to get max mol ID
        clusters = find_clusters(range(1,num_of_molecules+1),bonds)
        filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules]  # Filter out single-molecule clusters
        clusters_over_time.append(filtered_clusters)
        delta_r = []
        for j in range(1,21):
            com_j = np.zeros(2)
            if j not in molecules:
                delta_r.append(0)
            else:
                com = np.zeros(2)
                for cluster in filtered_clusters:
                    if j in cluster:
                        for molecule in cluster:
                            com += center_of_mass(atoms_data, molecule) 
                        com_j = center_of_mass(atoms_data, j) 
                        if i==total_measurements[-1] and 1 in cluster:
                            com_mol1 = center_of_mass(atoms_data,1)
                            vector_polymer_com = com/len(cluster) - com_mol1
                            atom_id1 = 1 + (j-1)*size_of_monomer #get atom id of first binding site of monomer j
                            atom_id2 = j*size_of_monomer #get atom id of last binding site of monomer j
                            com_molj_first_edge_monomer = specific_atom_ID_for_mol(atoms_data, target_mol_id=j, atom_ID=atom_id1)
                            com_molj_last_edge_monomer = specific_atom_ID_for_mol(atoms_data, target_mol_id=j, atom_ID=atom_id2)
                            #com_mol1_type1 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=1)
                            #com_mol1_type2 = center_of_mass_specific_type(atoms_data, target_mol_id=j, target_atom_type=2)
                            #try:
                            vector_monomer =  com_molj_last_edge_monomer - com_molj_first_edge_monomer 
                            #if 1 in cluster:
                            if 1 in cluster:
                                directionalities.append(np.dot(vector_monomer, vector_polymer_com))
                            #except:
                            #    vector_monomer = np.array([0,0])
                            #    print('NO SECOND BINDING SITE AT '+str(output_file_positions))
                        delta_rj = np.linalg.norm(com/len(cluster)-com_j)  #relative com of j - CHECK IF LAMMPS COM RESPECTS PBC
                        delta_r.append(delta_rj)
        total_delta_r.append(delta_r)
    #total_delta_r = np.array(total_delta_r)
    #check if all monomers point in the same direction

    signs = [x > 0 for x in directionalities] #signs of dot product of every monmer with center of mass
    if all(signs):
        prefactor = 1
    else:
        prefactor = -1
    
    shifts_all_particles = []
    for k in range(20):
        #shifts = total_delta_r[:-1,k]-total_delta_r[1:,k]
        shift = total_delta_r[-1][k] - total_delta_r[0][k]
        #shifts = [total_delta_r[i][k] - total_delta_r[i+1][k] for i in range(len(total_delta_r) - 1)]
        shifts_all_particles.append(np.abs(shift))

    #find_fastest_growing_pair(np.array(clusters_over_time,dtype = object),num_of_molecules) #used to find the pair that grew fastest in annealing simulations
        
    max_shift = shifts_all_particles[0]
    max_index = 0

    max_shift = max_shift*prefactor #coupling monomer directionality to polymer directionality
            
    final_clusters = clusters_over_time[-1] #number of clusters in the end of self assembly
    max_shift_per_cluster = []
    max_index_per_cluster = []
    cluster_dynamics = []
    cluster_dynamics.append(clusters_over_time[0])
    filtered_history = clusters_over_time
    initial_size = len(filtered_history[0])
    for t in range(1, len(filtered_history)):
        current_size = len(filtered_history[t])
        if current_size!=initial_size:
            #print(filtered_history[t])
            cluster_dynamics.append(clusters_over_time[t])
            initial_size = current_size
        elif current_size == initial_size:
            clusters1 =  filtered_history[t-1]
            clusters2 =  filtered_history[t]
            for i in np.arange(len(filtered_history[t])):
                if sorted(clusters1[i]) != sorted(clusters2[i]):
                    #print(filtered_history[t])
                    cluster_dynamics.append(clusters_over_time[t])
                    initial_size = current_size       
                    
    return max_shift, cluster_dynamics, clusters_over_time

def get_relative_com_1_particle_postanalysis_new(output_file_positions,dump_freq):
    #num_of_measurements = int(sim_steps/dump_freq)
    #total_measurements = range(num_of_measurements) #100 for now

    last_timestep = find_last_timestep(output_file_positions)
    total_measurements = range(int(last_timestep/dump_freq))

    total_delta_r = []
    clusters_over_time = []
    branched_nodes = []
    prefactors = []
    
    for i in total_measurements:
        bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=i)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        clusters = find_clusters(range(1,num_of_molecules+1),bonds)
        filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules]  # Filter out single-molecule clusters
        clusters_over_time.append(filtered_clusters)
        delta_r = []
        for j in range(1,31):
            com_j = np.zeros(2)
            if j not in molecules:
                delta_r.append(0)
            else:
                com = np.zeros(2)
                for cluster in filtered_clusters:
                    if j in cluster:
                        for molecule in cluster:
                            com += center_of_mass(atoms_data, molecule) 
                        com_j = center_of_mass(atoms_data, j) 
                        if j==1:
                            vector_polymer_com = com/len(cluster) - com_j
                            com_mol1_type1 = center_of_mass_specific_type(atoms_data, target_mol_id=1, target_atom_type=1)
                            com_mol1_type2 = center_of_mass_specific_type(atoms_data, target_mol_id=1, target_atom_type=2)
                            try:
                                vector_monomer = com_mol1_type2 - com_mol1_type1
                            except:
                                vector_monomer = np.array([0,0])
                                print('NO SECOND BINDING SITE AT '+str(output_file_positions))
                        directionality = np.dot(vector_monomer, vector_polymer_com)
                        prefactor = 1
                        if directionality<0:
                            prefactor=-1
                        prefactors.append(prefactor)
                        delta_rj = np.linalg.norm(com/len(cluster)-com_j)  #relative com of j - CHECK IF LAMMPS COM RESPECTS PBC
                        delta_r.append(delta_rj)
        total_delta_r.append(delta_r)
    #total_delta_r = np.array(total_delta_r)
    
    shifts_all_particles = []
    for k in range(29):
        #shifts = total_delta_r[:-1,k]-total_delta_r[1:,k]
        shift = total_delta_r[-1][k] - total_delta_r[0][k]
        #shifts = [total_delta_r[i][k] - total_delta_r[i+1][k] for i in range(len(total_delta_r) - 1)]
        shifts_all_particles.append(np.abs(shift))

    most_growing = [1,2]#find_fastest_growing_pair(np.array(clusters_over_time,dtype = object),num_of_molecules)
        
    max_shift = 0
    for l in most_growing:
        if shifts_all_particles[l-1]>max_shift:
            max_shift = shifts_all_particles[l-1]
            max_index = l

    max_shift = max_shift*prefactor #coupling monomer directionality to polymer directionality
            
    final_clusters = clusters_over_time[-1] #number of clusters in the end of self assembly
    max_shift_per_cluster = []
    max_index_per_cluster = []
    cluster_dynamics = []
    time_cluster_change = []
    cluster_dynamics.append(clusters_over_time[0])
    filtered_history = clusters_over_time
    initial_size = len(filtered_history[0])
    for t in range(1, len(filtered_history)):
        current_size = len(filtered_history[t])
        if current_size!=initial_size:
            #print(filtered_history[t])
            cluster_dynamics.append(clusters_over_time[t])
            time_cluster_change.append(t)
            initial_size = current_size
        elif current_size == initial_size:
            clusters1 =  filtered_history[t-1]
            clusters2 =  filtered_history[t]
            for i in np.arange(len(filtered_history[t])):
                if sorted(clusters1[i]) != sorted(clusters2[i]):
                    #print(filtered_history[t])
                    cluster_dynamics.append(clusters_over_time[t])
                    time_cluster_change.append(t)
                    initial_size = current_size          
                    
    return max_shift, cluster_dynamics, clusters_over_time, time_cluster_change, total_delta_r, prefactors

def get_relative_com_1_particle_postanalysis(output_file_positions,dump_freq):
    #num_of_measurements = int(sim_steps/dump_freq)
    #total_measurements = range(num_of_measurements) #100 for now

    last_timestep = find_last_timestep(output_file_positions)
    total_measurements = range(int(last_timestep/dump_freq))

    total_delta_r = []
    clusters_over_time = []
    branched_nodes = []
    
    for i in total_measurements:
        bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=i)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        clusters = find_clusters(range(1,num_of_molecules+1),bonds)
        filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules]  # Filter out single-molecule clusters
        clusters_over_time.append(filtered_clusters)
        delta_r = []
        for j in range(1,31):
            if j>num_of_molecules or j not in molecules:
                delta_r.append(0)
            else:
                com = np.zeros(2)
                for cluster in filtered_clusters:
                    if j in cluster:
                        for molecule in cluster:
                            com += center_of_mass(atoms_data, molecule) 
                        com_j = center_of_mass(atoms_data, j) 
                        delta_rj = np.linalg.norm(com/len(cluster)-com_j)  #relative com of j - CHECK IF LAMMPS COM RESPECTS PBC
                        delta_r.append(delta_rj)
        total_delta_r.append(delta_r)
    total_delta_r = np.array(total_delta_r)
    
    shifts_all_particles = []
    for k in range(30):
        shifts = total_delta_r[:-1,k]-total_delta_r[1:,k]
        shifts_all_particles.append(np.abs(sum(shifts)))

    most_growing = [1,2]#find_fastest_growing_pair(np.array(clusters_over_time,dtype = object),num_of_molecules)
        
    max_shift = 0
    for l in most_growing:
        if shifts_all_particles[l-1]>max_shift:
            max_shift = shifts_all_particles[l-1]
            max_index = l
            
    final_clusters = clusters_over_time[-1] #number of clusters in the end of self assembly
    max_shift_per_cluster = []
    max_index_per_cluster = []
    cluster_dynamics = []
    time_cluster_change = []
    cluster_dynamics.append(clusters_over_time[0])
    filtered_history = clusters_over_time
    initial_size = len(filtered_history[0])
    for t in range(1, len(filtered_history)):
        current_size = len(filtered_history[t])
        if current_size!=initial_size:
            #print(filtered_history[t])
            cluster_dynamics.append(clusters_over_time[t])
            time_cluster_change.append(t)
            initial_size = current_size
        elif current_size == initial_size:
            clusters1 =  filtered_history[t-1]
            clusters2 =  filtered_history[t]
            for i in np.arange(len(filtered_history[t])):
                if sorted(clusters1[i]) != sorted(clusters2[i]):
                    #print(filtered_history[t])
                    cluster_dynamics.append(clusters_over_time[t])
                    time_cluster_change.append(t)
                    initial_size = current_size       
                    
    return max_shift, cluster_dynamics, clusters_over_time,time_cluster_change, total_delta_r, prefactor

def get_number_of_annealing_events(cluster_dyn):

    annealing = 0
    breaking = 0
    for i in np.arange(len(cluster_dyn)-1):
        new_clusters = []
        curr_clusters = cluster_dyn[i]
        next_clusters = cluster_dyn[i+1]
        for cluster in next_clusters:
            if cluster not in curr_clusters:
                new_clusters.append(cluster) #finds clusters that werent in the previous timestep
                
        lengths_of_previous_clusters = []
        detected_clusters = []
        for new_cluster in new_clusters:
            length_of_previous_cluster = []
            for particle in new_cluster:#for each particle in new cluster find the previous cluster
                for cluster in curr_clusters:
                    if particle in cluster:
                        #print(cluster)
                        if cluster not in detected_clusters:
                            length_of_previous_cluster.append(len(cluster))  # Return the cluster containing the particle
                        detected_clusters.append(cluster)
            if len(length_of_previous_cluster)>0:
                lengths_of_previous_clusters.append(length_of_previous_cluster)
        #print('LENGTHS AT ' + str(i))
        #print(lengths_of_previous_clusters)
            
        for merging_cluster_sizes in lengths_of_previous_clusters:
            if len(merging_cluster_sizes)>1: #otherwise its detaching
                count = sum(1 for x in merging_cluster_sizes if x > 1) #merging clusters of size more than 1
                if count>1:
                    annealing+=1
            elif len(merging_cluster_sizes)==1 and len(curr_clusters)>1:
                max_len_next_clusters = max([len(cluster) for cluster in next_clusters])
                #print('MAX LEN CURR CLUSTERS AT ' + str(i))
                #print(max_len_next_clusters)
                if merging_cluster_sizes[0]-max_len_next_clusters>1:
                #print('NEW CLUSTERS')
                # print(new_clusters)
                # print('OLD CLUSTERS')
                # print(curr_clusters)
                    breaking+=1
                    
    return annealing, breaking

# Calculate the centers of mass
#centers_of_mass = calculate_center_of_mass(atoms_data, box_bounds)
#print(bonds)

#shift everything so that it is compact
def shift_end_coms(coms,box_bounds,bonds):
    xlo, xhi, ylo, yhi = box_bounds
    L = xhi - xlo
    x_list = [coms[index][0] for index in coms]
    y_list = [coms[index][1] for index in coms]
    x_sum = sum(x_list)
    y_sum = sum(y_list)
    
    for mol1 in coms:
        x1 = coms[mol1][0]
        y1 = coms[mol1][1]
        for mol2 in coms:
            x2 = coms[mol2][0]
            y2 = coms[mol2][1]
            
            if mol1 < mol2:
                borderx = False #check if breaks on x border
                if (mol1,mol2) in bonds:
                    dist = np.abs(x1-x2)
                    if dist>L/2:
                        borderx = True
                            
                bordery = False #check if breaks on y border
                if (mol1,mol2) in bonds:
                    dist = np.abs(y1-y2)
                    if dist>L/2:
                        bordery = True
                        
                if borderx==True:
                    if x_sum>=0:
                        if x1<0:
                            coms[mol1][0]+=L
                        if x2<0:
                            coms[mol2][0]+=L
                    elif x_sum<0:
                        if x1>0:
                            coms[mol1][0]-=L
                        if x2>0:
                            coms[mol2][0]-=L
                            
                if bordery==True:
                    if y_sum>=0:
                        if y1<0:
                            coms[mol1][1]+=L
                        if y2<0:
                            coms[mol2][1]+=L
                    elif y_sum<0:
                        if y1>0:
                            coms[mol1][1]-=L
                        if y2>0:
                            coms[mol2][1]-=L
                            
    return coms

# Plot the shifted centers of mass and bonds
#plot_centers_of_mass_and_bonds(centers_of_mass, bonds, box_bounds)

def find_clusters(particles, bonds):
    from collections import defaultdict, deque
    
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

# Example usage
particles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bonds = [(1, 2), (2, 3), (3, 4), (5, 6), (7, 8), (8, 9)]
clusters = find_clusters(particles, bonds)
#print(clusters)

def center_of_mass_specific_type(atoms_data, target_mol_id, target_atom_type):
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
    positions = [(x, y) for _, mol_id, atom_type, x, y in atoms_data if mol_id == target_mol_id and atom_type == target_atom_type]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])

def specific_atom_ID_for_mol(atoms_data, target_mol_id, atom_ID):
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
    positions = [(x, y) for atom_id, mol_id, atom_type, x, y in atoms_data if mol_id == target_mol_id and atom_id == atom_ID]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])

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
    positions = [(x, y) for _, mol_id, _, x, y in atoms_data if mol_id == target_mol_id]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])

def extract_com_values(filename, target_timestep,i1,i2):
    
    with open(filename, 'r') as file:
        # Skip the first two comment lines
        file.readline()
        file.readline()
        
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            timestep = int(columns[0])
            
            # Check if the current line corresponds to the target timestep
            if timestep == target_timestep:
                # Extract c_com1[1] and c_com1[2] values
                c_comi1_x = float(columns[2*i1-1])
                c_comi1_y = float(columns[2*i1])
                
                com1 = [c_comi1_x,c_comi1_y]
                
                c_comi2_x = float(columns[2*i2-1])
                c_comi2_y = float(columns[2*i2])
                
                com2 = [c_comi2_x,c_comi2_y]
                
                comx_total, comy_total  = get_com_from_coms(com1,com2)
                
                deltax_i1 = comx_total - c_comi1_x
                deltay_i1 = comy_total - c_comi1_y
                
                deltax_i2 = comx_total - c_comi2_x
                deltay_i2 = comx_total - c_comi2_y
                
                return deltax_i1, deltay_i1, deltax_i2, deltay_i2
                
    # If the target timestep is not found, return None
    return None

def just_extract_com_values(filename, target_timestep,i1):
    
    with open(filename, 'r') as file:
        # Skip the first two comment lines
        file.readline()
        file.readline()
        
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            timestep = int(columns[0])
            
            # Check if the current line corresponds to the target timestep
            if timestep == target_timestep:
                # Extract c_com1[1] and c_com1[2] values
                c_comi1_x = float(columns[2*i1-1])
                c_comi1_y = float(columns[2*i1])
                
                com1 = np.array([c_comi1_x,c_comi1_y])
            
                return com1
                
    # If the target timestep is not found, return None
    return None

def linear_fit(coms):
    x_vals = [com[0] for com in coms.values()]
    y_vals = [com[1] for com in coms.values()]

    reg = LinearRegression().fit(x_vals, y_vals)
    params = reg.get_params()
    y_pred = reg.predict(x_vals)

    mse = mean_squared_error(y_vals, y_pred)
    return mse,params,y_pred

def extract_gyration_radius(filename):
    
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
        columns = last_line.strip().split()
        #print('COLUMNS')
        #print(columns)
        gyr = float(columns[1])
        #print('GYR RADIUS')
        #print(gyr)
        
    return gyr

def find_fastest_growing_pair(cluster_array):
    # Dictionary to track the size of the cluster for each number over time
    growth = {}

    # Iterate over the clusters over time
    for t, clusters in enumerate(cluster_array):
        cluster_map = {}
        for cluster in clusters:
            cluster_size = len(cluster)
            for number in cluster:
                cluster_map[number] = cluster_size
        growth[t] = cluster_map
                
        
    growth_steps = np.zeros(10)
    for k in range(1,11):#ten monomers
        growing = 0
        for t in range(1,len(cluster_array)):
            #compare t and t+1 - for each growth +1 point
            old_cluster_size = growth[t-1][k]
            new_cluster_size = growth[t][k]
            if new_cluster_size > old_cluster_size:
                growing +=1
        growth_steps[k-1]=growing
    
    a = np.where(growth_steps == max(growth_steps))[0] 

    return a 

def find_end_monomers(clusters, bonds):
    # Create a dictionary to count the bonds for each monomer
    bond_count = defaultdict(int)
    
    # Populate bond counts
    for bond in bonds:
        bond_count[bond[0]] += 1
        bond_count[bond[1]] += 1
    
    end_monomers_output = []
    fully_connected_output = []
    just_end_monomers_output = []
    
    # Process each cluster
    for cluster in clusters:
        end_monomers = []
        
        # Check bond counts for each monomer in the cluster
        for monomer in cluster:
            if bond_count[monomer] == 1:  # Monomer with 1 bond is an "end"
                end_monomers.append(monomer)
        
        # If exactly 2 end monomers, add to the result
        if len(end_monomers) == 2:
            end_monomers_output.append((cluster, end_monomers))
            just_end_monomers_output.append(end_monomers)
        # If all monomers have exactly 2 bonds, it's fully connected (loop or chain)
        elif all(bond_count[monomer] == 2 for monomer in cluster):
            fully_connected_output.append(cluster)
    
    return end_monomers_output, fully_connected_output, just_end_monomers_output

def com_distances(com_mol1,com_mol2):
    x1,y1 = com_mol1
    x2,y2 = com_mol2
    
    com_distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    return com_distance
