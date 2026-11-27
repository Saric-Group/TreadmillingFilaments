import pandas as pd 
import numpy as np 
import pandas as pd
import os, re, glob, random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def extract_parameters_from_filename(keyword, filename, sep = "_"):
    
    split_filename = filename.split(sep)
    i = 0
    for i in range(len(split_filename)):
        if split_filename[i] == keyword:
            return split_filename[i+1]
    return None

def get_desired_attributes(filename, keywords, forbidden_keys = []):

    # reads out the parameters in the name of a file
    # (more advanced version of 'extract_parameters_from_filename')

    filename = filename.split('/')[-1]

    parameters = {}
    for keys in keywords:
        cast_as = keywords[keys]
        if keys not in forbidden_keys:
            param   = cast_as(extract_parameters_from_filename(keys, filename, sep = "_"))
            parameters[keys] = param 

    return parameters

def generate_name_file(combo_dict, data, wildcard_keys = None):

    # generate the name of the file
    dirname = ""
    for keyword in data["filename_structure"]:
        dirname+=f"_{keyword}_"
        if keyword in wildcard_keys:
            dirname+="*"
        elif keyword not in combo_dict.keys():
            dirname+=f"{data[keyword]}"
        else: 
            dirname+=f"{combo_dict[keyword]}"
            data[keyword] = combo_dict[keyword]

    return dirname, data
    
# --------------------------------------
# FUNCTIONS FROM MARIJA

def read_lammps_dump(filename, timestep_index = -1):

    """
    Reads a LAMMPS dump file and extracts atoms' data for a specific timestep.
    
    Parameters:
        filename (str): Path to the dump file.
        timestep_index (int): Index of the timestep to read (-1 for the last timestep).
    
    Returns:
        atoms_data (list of tuples): Each tuple contains (atom_id, mol_id, atom_type, x, y, z).
        box_bounds (tuple): Box boundaries as (xlo, xhi, ylo, yhi, zlo, zhi).
    """

    timestep_start = "ITEM: TIMESTEP"
    box_bounds_start = "ITEM: BOX BOUNDS"
    atoms_start = "ITEM: ATOMS"

    atoms_data = []
    box_bounds = None
    timestep_indices = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Identify all timesteps in the file
    for i, line in enumerate(lines):
        if line.strip() == timestep_start:
            timestep_indices.append(i)

    # Validate the requested timestep index
    if not timestep_indices:
        raise ValueError("No timesteps found in the file.")
    if timestep_index == -1:  # Use the last timestep
        timestep_index = len(timestep_indices) - 1
    if timestep_index >= len(timestep_indices):
        #print('FILENAME IS' + str(filename))
        raise IndexError(f"Requested timestep_index {timestep_index} is out of range, at {filename}")

    # Get the starting line index of the desired timestep
    timestep_start_index = timestep_indices[timestep_index]

    # Parse box bounds
    box_bounds_line = timestep_start_index + 5
    box_bounds = []
    for i in range(2):  # Read 2 lines for x, y bounds
        bounds = tuple(map(float, lines[box_bounds_line + i].strip().split()))
        box_bounds.extend(bounds)
    box_bounds = tuple(box_bounds)  # Flatten to (xlo, xhi, ylo, yhi, zlo, zhi)

    # Parse the number of atoms
    num_atoms_line = timestep_start_index + 3
    num_atoms = int(lines[num_atoms_line].strip())

    # Parse atoms data
    atoms_data_start = timestep_start_index + 9  # Adjust offset as needed
    atoms_data = []
    for line in lines[atoms_data_start:atoms_data_start + num_atoms]:
        parts = line.strip().split()
        if len(parts) < 5:  # Ensure all components (atom_id, mol_id, type, x, y, z) are present
            raise ValueError("Malformed atoms data line: " + line)
        atom_id = int(parts[0])
        mol_id = int(parts[1])
        atom_type = int(parts[2])
        x, y = map(float, parts[3:5])
        atoms_data.append((atom_id, mol_id, atom_type, x, y))

    return atoms_data, box_bounds

"""
def get_bonds(output_file_path, timestep_index, cutoff_distance):

    atoms_data, box_bounds = read_lammps_dump(output_file_path, timestep_index)
    
    molecules = {}
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y = atom
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append((atom_id, x, y,atom_type))
        
    # Find touching molecules and add bonds
    bonds = []
        
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds): #if theyre touching move
                    bonds.append((mol_id1, mol_id2))
                    
    return bonds, atoms_data, molecules
"""

def write_current_coordinates_ovito(molecules, params):

    molecule_ids = list(molecules.keys())
    num_molecules = len(molecule_ids)
    num_atoms = len(molecules[molecule_ids[0]])

    params['fovito'].writelines(f"ITEM: TIMESTEP\n{params['current_time']}\n")
    params['fovito'].writelines(f"ITEM: NUMBER OF ATOMS\n{num_molecules*num_atoms}\n")
    params['fovito'].writelines(f'ITEM: BOX BOUNDS ff ff pp\n')
    params['fovito'].writelines(f'-100 100\n')
    params['fovito'].writelines(f'-7.5 7.5\n')
    params['fovito'].writelines(f'-2.5 2.5\n')
    params['fovito'].writelines(f'ITEM: ATOMS id mol type x y\n')
    for mols in molecules:
        temp_data = molecules[mols]
        for i in range(len(temp_data)):
            params['fovito'].writelines(f'{temp_data[i][0]} {mols} {temp_data[i][1]} {temp_data[i][2]} {temp_data[i][3]}\n')

def generatedataframe_buildingblock(args):

    """
    Generates a dataframe to facilitate the monitoring of a desired
    quantity. That quantity depends on the file passed, 
    as well as the entry line given. See examples below.
    """

    if not os.path.exists(args['datafile']):
        print(f"File {args['datafile']} does not exist!")
        return -1

    print("Generating dataframe for monitoring...")

    save_dataframe = open(args['save_dataframe'], 'w')

    # in case we want to make it a frequency thing
    taken_snapshots = 0

    with open(args['datafile'], 'r') as f:
        
        for line in f:

            # for a given timestep in the trajectory
            if "ITEM: TIMESTEP" in line:
                timestep = int(next(f).strip())
                taken_snapshots+=1

            # for a given number of elements
            elif "ITEM: NUMBER OF ENTRIES" in line:
                num_entries = int(next(f).strip())
                #print("Current number of entries: ", num_entries)

            # extract the coordinates
            # example of entry line can be 
            # 'ITEM: ENTRIES c_cBonds[1] c_cBonds[2] c_cBonds[3] c_cDistBonds[1] c_cDistBonds[2]'
            # 'ITEM: ENTRIES c_cAngles[1] c_cAngles[2] c_cAngles[3] c_cAngles[4] c_ctheta'
            # 'ITEM: ATOMS id mol type x y'
            elif (args['entry_line'] in line):

                if (taken_snapshots%args['freq']==0):
                    for _ in range(num_entries):
                        data_line = next(f).strip()

                        # introduce further analysis here if
                        # you only want to focus on a specific building block
                        save_dataframe.write(f"{timestep} {data_line}\n")

    save_dataframe.close()
    print("Done.")

def extract_parameters_from_dirname(dirname):
    parameters = {}
    raw_dirname = (dirname.split("/"))[-1]
    split_dirname = raw_dirname.split("_")
    for i in range(1, len(split_dirname)-1, 2):
        parameters[split_dirname[i]] = split_dirname[i+1]
    return parameters

def produce_directory_name(data, combo_dict):

    """
    This function takes a dictionary of parameter_name --> parameter_value pairs
    and constructs the name of the directory
    """
    
    dirname = "_"
    for ky in combo_dict:
        data[ky] = combo_dict[ky]
        dirname+=f"{ky.replace('_','')}_{combo_dict[ky]}_"

    return dirname

def visualize_desired_dataset(A, B, C, df, angle_at_open, angle_at_close, ax, color, ls, flag=False):
    sustract = 0
    if flag:
        sustract = 90

    # actual angle value 
    dfS = df[((df[1]==A) & (df[2] == B) & (df[3] == C)) | ((df[1]==B) & (df[2] == A) & (df[3] == C)) | ((df[1]==A) & (df[2] == C) & (df[3] == B)) | ((df[1]==C) & (df[2] == B) & (df[3] == A)) | ((df[1]==C) & (df[2] == A) & (df[3] == B)) | ((df[1]==B) & (df[2] == C) & (df[3] == A))]
    ax.plot(dfS[0].values, dfS[5].values-sustract, label = 'Trajectory', color = color, ls = ls)
    
    # mean value
    #mean_value = dfS[5].mean()-sustract
    #ax.plot(dfS[0].values, mean_value*np.ones(len(dfS[0])), ls = '--', color = 'red')
    
    # expected angle values
    ax.plot(dfS[0].values, angle_at_open*np.ones(len(dfS[0])), ls = ':', color = 'r', label = 'Open state')
    ax.plot(dfS[0].values, angle_at_close*np.ones(len(dfS[0])), ls = ':', color = 'g', label = 'Closed state')

def plot_angles_buildingblock(open_angle, dataframe = 'default.csv', figsize = (10, 3), savefig = 'Default.pdf'):

    """
    Structure of the data is:
    timestep A B C type angle

    Figures have 4 columns because there are 4 different
    types of angles in the system
    """

    #load_pretty_figure_setup()
    fig, ax = plt.subplots(tight_layout=True, figsize = figsize, ncols = 4)
    
    # load data
    df = pd.read_csv(dataframe, header = None, index_col = False, sep = " ")
    
    # 180 degree angles
    visualize_desired_dataset(1, 2, 3, df, 180, ax[0], '1&2&3', 'C0', '-')
    visualize_desired_dataset(4, 5, 6, df, 180, ax[0], '4&5&6', 'C1', '--')

    # 90 degree angles
    visualize_desired_dataset(1, 2, 5, df, 90, ax[1], '1&2&5', 'C0', '-')
    visualize_desired_dataset(3, 2, 5, df, 90, ax[1], '3&2&5', 'C1', '--')

    # variable angle 1
    visualize_desired_dataset(2, 5, 6, df, 0, ax[2], '2&5&6', 'C0', '-', flag=True)

    # variable angle 2
    visualize_desired_dataset(2, 5, 4, df, 90-open_angle, ax[3], '2&5&4', 'C0', '-')

    for i in range(3):
        ax[i].set_xlabel("Simulation steps")
        ax[i].set_ylabel("Bond length")
        ax[i].legend()

    ax[0].set_title("Angle type 1 (180)")
    ax[1].set_title("Angle type 2 (90)")
    ax[2].set_title(f"Angle type 3 (closed: 90;\nopen: {90+open_angle})")
    ax[3].set_title(f"Angle type 4 (closed: 90;\nopen: {90-open_angle})")

    fig.savefig(savefig)

def plot_reactioncoordinate_buildingblock(open_angle, dataframe = 'default.csv', figsize = (10, 3), savefig = 'Default.pdf', title = None):

    """
    Plot the time evolution of the reaction coordinate in the system
    """

    #load_pretty_figure_setup()
    fig, ax = plt.subplots(tight_layout=True, figsize = figsize)
    
    # load data
    df = pd.read_csv(dataframe, header = None, index_col = False, sep = " ")

    # variable angle 2
    visualize_desired_dataset(2, 5, 4, df, 90-open_angle, 90, ax, 'C0', '-')

    ax.legend()
    ax.set_xlabel("Simulation steps")
    ax.set_ylabel("Reaction coordinate")
    ax.set_title(f"{title}", fontsize = 6)

    fig.savefig(savefig)
    plt.close(fig)

def dfs(node,adjacency_list,visited):
        stack = [node]
        while stack:
            current = stack.pop()
            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

def count_subgraphs_com(num_vertices, bonds):
    
    # Create adjacency list from bonds
    adjacency_list = defaultdict(list)
    for u, v in bonds:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    visited = set()
    num_subgraphs = 0
    
    # Traverse all vertices
    for vertex in range(1, num_vertices + 1):
        if vertex not in visited:
            num_subgraphs += 1
            visited.add(vertex)
            dfs(vertex,adjacency_list,visited)
    
    return num_subgraphs

def count_filaments(clusters):

    filament_num = 0
    filament = []
    for entry in clusters:
        # we determine whether there are filaments
        # by checking whether the entries contain
        # more than one molecule (working definition
        # for a filament
        if len(entry)>1: 
            filament_num +=1
            filament.append(entry)

    return filament_num, filament

def find_clusters(particles, bonds):
    
    def bfs(start, visited):
        # perform BFS and find all nodes in the same cluster
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
    
    # create an adjacency list for connectivity graph
    adjacency_list = defaultdict(list)
    for (a, b) in bonds:
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)
    
    # Find all clusters
    visited = set()
    clusters = []

    # technically, this is 'for molecules in molecule...'
    for particle in particles:
        if particle not in visited:
            cluster = bfs(particle, visited)
            clusters.append(cluster)

    return clusters

def are_molecules_touching(molecule1, molecule2, cutoff, model0_condition = False):

    for atom1 in molecule1:
        for atom2 in molecule2:
            atom_type1, x1, y1 = atom1[1], atom1[2], atom1[3]
            atom_type2, x2, y2 = atom2[1], atom2[2], atom2[3]
            
            dx = x1-x2
            dy = y1-y2

            # since these simulations don't have PBC, no need for the translation
            #dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
            #dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= cutoff:
                # in model 0 we consider that molecules are bound when 
                # the topmost particles interact with each other
                # notice that due to hydrolysis, particles will change type
                # we therefore need to account for all these particle types
                if model0_condition:
                    if atom_type1 in [4, 6, 9, 10] and atom_type2 in [4, 6, 9, 10]:
                        return True
                    #if (atom_type1 == 4 and atom_type2 == 6) or (atom_type1 == 6 and atom_type2 == 4):
                    #    return True
                    #elif (atom_typ1 == 4 and atom_type2 == 10) or (atom_type1 == 10 and atom_type2 == 4):
                    #    return True
                else:
                    return True

    return False

def get_com_molecule(ids_in_filament, molecule_data):

    xcm, ycm, atoms_in_molecule = 0, 0, 0
    for molecule_ids in ids_in_filament:
        current_molecule_data = molecule_data[molecule_ids]
        atoms_in_molecule += len(current_molecule_data)
        for data in current_molecule_data:
            xcm += data[2]
            ycm += data[3]
    
    return np.array([xcm/atoms_in_molecule, ycm/atoms_in_molecule])
    
def get_max_filament(ids_in_filament, molecule_data):

    maxx, maxy = -999999999, 0
    for molecule_ids in ids_in_filament:
        current_molecule_data = molecule_data[molecule_ids]
        for data in current_molecule_data:
            if data[2]>maxx:
                maxx = data[2]
                maxy = data[3]
    
    return np.array([maxx, maxy])

def get_min_filament(ids_in_filament, molecule_data):

    minx, miny = 999999999, 0
    for molecule_ids in ids_in_filament:
        current_molecule_data = molecule_data[molecule_ids]
        for data in current_molecule_data:
            if data[2]<minx:
                minx = data[2]
                miny = data[3]
    
    return np.array([minx, miny])

def determine_filament_connectivity(molecules, params, cutoff_fullyformed = 0.9):

    connected_molecules = []
    bonded_molecules    = []

    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_fullyformed*params["cutoff"], model0_condition = params["model0_condition"]): 
                    connected_molecules.append((mol_id1, mol_id2))
                    if mol_id1 not in bonded_molecules:
                        bonded_molecules.append(mol_id1)
                    if mol_id2 not in bonded_molecules:
                        bonded_molecules.append(mol_id2)
    
    return connected_molecules, bonded_molecules

def run_analysis_polymerization(molecules, params, min_key):

    #print(f"CURRENT TIME IS: {params['current_time']}")
    
    # determine how many molecules there are
    num_molecules = len(molecules.keys())
    #print(f"Snapshot contains {num_molecules} molecule(s)")
    #wait = input()

    # determine what molecules are connected to each other
    connected_molecules, bonded_molecules = determine_filament_connectivity(molecules, params)
    #print(f"Connected molecules are : {connected_molecules}")
    
    # find how many connected components (i.e., interacting/bound molecules) there are
    # the lowest molecule ID is 1, hence the range values
    connected_components = find_clusters(range(1,num_molecules+1), connected_molecules)
    #print(f"Connected components are : {connected_components}")

    # check how many filaments there are   
    filament_num, filament = count_filaments(connected_components)
    #print(f"There are {filament_num} filaments and the filament are : {filament}")
    #wait = input()

    if params["monitor_flow_of_matter"] == False: # only monitor the dynamics of the filament
        # if there is no filament (nothing has polymerised yet, just seeds and individual molecules)
        if filament_num == 0:
            com_seed     = get_com_molecule([min_key], molecules)
            max_seed     = get_max_filament([min_key], molecules)
            min_seed     = get_min_filament([min_key], molecules)
            params["fdata"].writelines(f"{params['current_time']} {com_seed[0]} {com_seed[0]} {max_seed[0]} {min_seed[0]}\n")
            params["fdata"].flush()
        elif filament_num == 1:
            com_seed     = get_com_molecule([min_key], molecules)
            com_filament = get_com_molecule(filament[0], molecules)
            max_filament = get_max_filament(filament[0], molecules)
            min_filament = get_min_filament(filament[0], molecules)
            params["fdata"].writelines(f"{params['current_time']} {com_seed[0]} {com_filament[0]} {max_filament[0]} {min_filament[0]}\n")
            params["fdata"].flush()
        else:
            print("More than one filament found. Molecules might be not connected to the seed!")
            params["fdata"].writelines(f"{params['current_time']} {'NaN'} {'NaN'} {'NaN'} {'NaN'}\n")
            params["fdata"].flush()

    elif params["monitor_flow_of_matter"]==True:

        # if we just detected a filament
        if filament_num == 1:

            # we can only compare once we have introduced
            # at least two different filaments
            if params["filament0"] == None:
                params["filament0"] = filament[0]
            elif params["filament1"] == None:
                params["filament1"] = filament[0]

            if (params["filament0"] is not None) and (params["filament1"] is not None):
                f0 = params["filament0"]
                f1 = params["filament1"]

                if f0 == f1:
                    #print("These filaments are identical")
                    # if they are identical, nothing was removed, nothing was added
                    params["fdata"].writelines(f"{params['current_time']} {0} {0}\n")
                    params["fdata"].flush()

                else:

                    #print("This is filament 0 --> ", f0)
                    #print("This is filament 1 --> ", f1)

                    # find the intersection between arrays
                    set_f1 = frozenset(f1)
                    intersection = [x for x in f0 if x in set_f1]
                    #print("Intersection ", intersection)

                    # where in the array can I find the roots
                    si_f0 = f0.index(intersection[0])+1
                    si_f1 = f1.index(intersection[0])+1
                    sf_f0 = f0.index(intersection[-1])+1
                    sf_f1 = f1.index(intersection[-1])+1

                    pos_root_left_f0 = si_f0
                    pos_root_right_f0 = len(f0)-sf_f0
                    pos_root_left_f1 = si_f1
                    pos_root_right_f1 = len(f1)-sf_f1

                    left_end = pos_root_left_f1-pos_root_left_f0
                    right_end = pos_root_right_f1-pos_root_right_f0

                    #print("The left end has changed: ", left_end)
                    #print("The right end has changed: ", right_end)

                    params["fdata"].writelines(f"{params['current_time']} {left_end} {right_end}\n")
                    params["fdata"].flush()
                    #wait = input()

                # update the filament status
                params["filament0"] = f1
                params["filament1"] = filament[0]



def run_analysis_treadmilling(molecules, params):

    #print(f"CURRENT TIMESTEP IS: {params['current_time']}")

    # determine how many molecules there are
    num_molecules = len(molecules.keys())
    params["fnummolecules"].writelines(f"{params['current_time']} {num_molecules}\n")
    #print(f"Snapshot contains {num_molecules} molecule(s)")

    # determine what molecules are connected to each other (adjacency graph filament)
    connected_molecules, bonded_molecules = determine_filament_connectivity(molecules, params)
    params["fadjacency"].writelines(f"{params['current_time']} ")
    for i in range(len(connected_molecules)):
        params["fadjacency"].writelines(f"{connected_molecules[i][0]} {connected_molecules[i][1]} ")
    params["fadjacency"].writelines(f"\n")
    #print(f"Connected molecules are : {connected_molecules}")

    # find how many connected components (i.e., interacting molecules) there are
    connected_components = find_clusters(bonded_molecules, connected_molecules)
    params["fconnected"].writelines(f"{params['current_time']} ")
    for i in range(len(connected_components)):
        component = connected_components[i]
        for j in range(len(component)):
            params["fconnected"].writelines(f"{component[j]} ")
        params["fconnected"].writelines(f"-111 ")
    params["fconnected"].writelines(f"\n")
    #print(f"Connected components are : {connected_components}")

    # check how many filaments there are   
    max_filament_length, id_max_filament_length = -1, -1
    filament_num, filament = count_filaments(connected_components)
    params["fprop"].writelines(f"{params['current_time']} ")
    params["fprop"].writelines(f"{filament_num} ")
    for i in range(len(filament)):
        filament_chunk = filament[i]
        params["fprop"].writelines(f" {len(filament_chunk)} ")
        if len(filament_chunk)>max_filament_length:
            max_filament_length=len(filament_chunk)
            id_max_filament_length = i
        for j in range(len(filament_chunk)):
            params["fprop"].writelines(f"{filament_chunk[j]} ")
        if len(filament)!=1:
            params["fprop"].writelines(f"-111 ")
    params["fprop"].writelines(f"\n")
    #print(f"There are {filament_num} filaments and the filament are : {filament}")
    
    #print(f"This is the maximum filament length: {max_filament_length}")
    # get the com of the longest filament
    if max_filament_length != -1:
        #print("Analyzing the filament!")
        com_filament = get_com_molecule(filament[id_max_filament_length], molecules)
        params["fdata"].writelines(f"{params['current_time']} {max_filament_length} {com_filament[0]} {com_filament[1]}\n")
        params["fdata"].flush()
    else:
        print("PROBLEM: There is a filament with negative length -- filament died?")
        #print("More than one filament found. Molecules might be not connected to the seed!")
        params["fdata"].writelines(f"{params['current_time']} {'NaN'} {'NaN'} {'NaN'}\n")
        params["fdata"].flush()

def extract_polymerization_kinetics(path, namefile, savefile, params, indices_vector = (3, 4), step_interval = 1, temporary_ovito = None):

    """
    Analyze each snapshot to determine the polymerization
    kinetics of the growing filament
    """

    # open a file to write down the kinetics/flow of matter
    fanalysis = open(savefile, "w")
    params["fdata"] = fanalysis

    if temporary_ovito is not None:
        fovito = open(temporary_ovito, "w")
        params["fovito"] = fovito

    # file that we are always going to analyze
    file_path = path+namefile

    # taken snaphots
    taken_snapshots = 0

    with open(file_path, 'r') as f:
        
        for line in f:

            # for a given timestep in the trajectory
            if "ITEM: TIMESTEP" in line:
                timestep = int(next(f).strip())
                params['current_time'] = timestep
                taken_snapshots+=1

            # for a given number of atoms
            elif "ITEM: NUMBER OF ATOMS" in line:
                num_atoms = int(next(f).strip())

            # extract the coordinates of the molecules
            elif "ITEM: ATOMS id mol type x y" in line:
                
                # extract the info about the molecules
                molecules = defaultdict(list)
                for _ in range(num_atoms):
                    atom_line = next(f).strip().split()
                    atom_id, mol_id, atom_type, x, y = int(atom_line[0]), int(atom_line[1]), int(atom_line[2]), float(atom_line[3]), float(atom_line[4])
                    molecules[mol_id].append([atom_id, atom_type, x, y])

                #print(f"CURRENT MOLECULE DICTIONARY: {molecules}")
                # conduct the desired analysis
                if taken_snapshots%step_interval == 0:
                    
                    # since the seed can rotate, we align it along
                    # the x axis, rotate the whole system,
                    # and then proceed as usual with the analysis
                    
                    # get the properties of the seed (not obvious seed will be there?)
                    keys = list(molecules.keys()) 
                    min_key = min(keys)
                    seed_properties = np.array(molecules[min_key])
                    particles_in_seed = len(seed_properties)

                    # for simulations where the seed is not fixed in space
                    if params['rotation_needed']:

                        # define an internal vector in the seed
                        index_3 = np.where(seed_properties[:, 0] == indices_vector[0])[0][0]
                        index_4 = np.where(seed_properties[:, 0] == indices_vector[1])[0][0]
                        coordinates_3 = seed_properties[index_3][2:]
                        coordinates_4 = seed_properties[index_4][2:]
                        
                        #print(f"Coordinates 3: {coordinates_3}")
                        #print(f"Coordinates 4: {coordinates_4}")

                        # current orientation of the seed
                        vx = coordinates_3[0]-coordinates_4[0]
                        vy = coordinates_3[1]-coordinates_4[1]
                        monomer_orientation = np.array([vx, vy])
                        monomer_orientation_3d = np.array([vx, vy, 0])
                        #print(f"Monomer orientation: {monomer_orientation}")
                        mod_monomer_orientation = np.sqrt(np.dot(monomer_orientation, monomer_orientation))
                        #print(f'Modulus: {mod_monomer_orientation}')

                        # define the desired alignment/orientation
                        desired_orientation = np.array([1, 0])
                        mod_desired_orientation = 1

                        # find the rotation that aligns the current
                        # orientation with the positive x axis
                        # compute the vector product
                        cross_product = np.cross(monomer_orientation_3d, np.array([1, 0, 0]))
                        dot_product = np.dot(monomer_orientation, desired_orientation)
                        theta_rotation = np.arccos(dot_product/(mod_desired_orientation*mod_monomer_orientation))
                        #print(f'Angle rotation: {theta_rotation}')
                        if cross_product[2]<0:
                            theta_rotation = 2*np.pi-theta_rotation
                        
                        # apply the rotation to all particles in the system
                        rotation_matrix = np.array([
                            [np.cos(theta_rotation), -np.sin(theta_rotation)], 
                            [np.sin(theta_rotation), np.cos(theta_rotation)]
                        ])
                        
                        #print("Test rotation: should bring alignment vector to the x-axis")
                        test = np.array([monomer_orientation[0], monomer_orientation[1]])
                        test_rotation = np.dot(rotation_matrix, test)

                        for mols in molecules:
                            #print(mols)
                            for atoms in range(len(molecules[mols])):
                                temp_array = molecules[mols][atoms][2:]
                                #print(molecules[mols][atoms])
                                #print(temp_array)
                                coordinates = np.array([temp_array[0], temp_array[1]])
                                apply_rotation = np.dot(rotation_matrix, coordinates)
                                molecules[mols][atoms][2] = apply_rotation[0]
                                molecules[mols][atoms][3] = apply_rotation[1]
                                #print(molecules[mols][atoms])
                                #wait = input()

                        if temporary_ovito is not None: 
                            write_current_coordinates_ovito(molecules, params)

                    # run the analysis on the data
                    run_analysis_polymerization(molecules, params, min_key)

    if temporary_ovito is not None:
        fovito.close()
    fanalysis.close() 


def extract_treadmilling_kinetics(path, namefile, params, step_interval = 1):

    """
    Analyze each snapshot to determine the treadmilling
    kinetics of a growing filament with hydrolysis
    """

    # file that we are always going to analyze
    file_path = path+namefile
    if not os.path.exists(file_path):
        print("File does not exist!")
        return -1

    # open a file to write down the available volume
    # (if the protrusion exists, for plotting purposes)
    fanalysis           = open(path+"/treadmilling_kinetics.dat", "w")
    fnummolecules       = open(path+"/num_molecules.dat", "w")
    fadjacency          = open(path+"/adjacency_lists.dat", "w")
    fconnected          = open(path+"/connected_components.dat", "w")
    ffilamentproperties = open(path+"/filament_properties.dat", "w")

    params["fdata"] = fanalysis
    params["fnummolecules"] = fnummolecules
    params["fadjacency"] = fadjacency
    params["fconnected"] = fconnected
    params["fprop"] = ffilamentproperties

    # taken snaphots
    taken_snapshots = 0

    with open(file_path, 'r') as f:
        
        for line in f:

            # for a given timestep in the trajectory
            if "ITEM: TIMESTEP" in line:
                timestep = int(next(f).strip())
                params['current_time'] = timestep
                taken_snapshots+=1

            # for a given number of atoms
            elif "ITEM: NUMBER OF ATOMS" in line:
                num_atoms = int(next(f).strip())

            # extract the coordinates of the molecules
            elif "ITEM: ATOMS id mol type x y" in line:
                
                # extract the info about the molecules
                molecules = {}
                for _ in range(num_atoms):
                    atom_line = next(f).strip().split()
                    atom_id, mol_id, atom_type, x, y = int(atom_line[0]), int(atom_line[1]), int(atom_line[2]), float(atom_line[3]), float(atom_line[4])
                    # exclude walls from analysis
                    if atom_type != 8:
                        if mol_id not in molecules:
                            molecules[mol_id] = []
                        molecules[mol_id].append((atom_id, atom_type, x, y))

                # conduct the desired analysis
                if taken_snapshots%step_interval == 0:

                    # get the properties of the seed
                    keys = list(molecules.keys()) 
                    min_key = min(keys)
                    seed_properties = np.array(molecules[min_key])
                    particles_in_seed = len(seed_properties)
                    
                    # run the analysis on the data
                    run_analysis_polymerization(molecules, params, min_key)

    fanalysis.close() 

def update_average_quantity_and_error(quantity, average_quantity, squared_quantity, count_per_time):

    """
    Updates the running average of any quantity and its squared sum to calculate the error,
    with a new trajectory.
    
    Parameters:
        average_quantity (np.ndarray): Running sum of the computed quantity, to be updated.
        squared_quantity (np.ndarray): Running sum of the computed quantity squared values.
        count_per_time (np.ndarray): Array tracking how many trajectories contribute to each time step.
        quantity (list or np.ndarray): List or array of values over time for the new trajectory.

    Returns:
        np.ndarray: Updated average quantity.
        np.ndarray: Updated sum of squared quantity values for error calculation.
        np.ndarray: Updated count of contributing trajectories at each time step.
    """
    
    # Number of time steps for this trajectory
    N = len(quantity)
    
    # If this is the first trajectory, initialize the arrays
    if average_quantity is None:
        average_quantity = np.zeros(N)
        squared_quantity = np.zeros(N)
        count_per_time = np.zeros(N)
    elif N > len(average_quantity):
        # Extend the arrays if the new trajectory is longer
        additional_length = N - len(average_quantity)
        average_quantity = np.concatenate([average_quantity, np.zeros(additional_length)])
        squared_quantity = np.concatenate([squared_quantity, np.zeros(additional_length)])
        count_per_time = np.concatenate([count_per_time, np.zeros(additional_length)])
    
    # Update the running sum of the quantity
    average_quantity[:N] += quantity
    squared_quantity[:N] += quantity**2
    
    # Update the count of contributing trajectories at each time step
    count_per_time[:N] += 1
    
    return average_quantity, squared_quantity, count_per_time

def finalize_average_quantity_and_error(average_quantity, squared_quantity, count_per_time, threshold=0):
    """
    Finalizes the average of the quantity and computes the error (standard error) after processing all trajectories.

    Parameters:
        average_quantity (np.ndarray): Running sum of the computed quantity.
        squared_quantity (np.ndarray): Running sum of squared computed quantity values.
        count_per_time (np.ndarray): Array of counts of how many trajectories contribute to each time step.

    Returns:
        np.ndarray: Final average of the quantity.
        np.ndarray: Standard error (SE) of the quantity at each time step.
    """
    
    # Avoid division by zero by checking where count_per_time is greater than (some threshold)
    valid_steps = count_per_time > threshold
    
    # Finalize the average quantity
    final_average_quantity = np.zeros_like(average_quantity)
    final_average_quantity[valid_steps] = average_quantity[valid_steps] / count_per_time[valid_steps]
    
    # Compute the standard error (SE)
    variance = np.zeros_like(squared_quantity)
    variance[valid_steps] = (squared_quantity[valid_steps] / count_per_time[valid_steps]) - (final_average_quantity[valid_steps] ** 2)
    
    # Calculate standard error (SE)
    standard_error = np.zeros_like(variance)
    standard_error[valid_steps] = np.sqrt(variance[valid_steps] / count_per_time[valid_steps])
    
    return final_average_quantity[valid_steps], standard_error[valid_steps]  