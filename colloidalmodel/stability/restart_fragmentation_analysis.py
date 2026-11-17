import numpy as np
import os

def create_restart_excluding_molecules(atom_dump_file, bond_dump_file, which_simulation,
                                       output_file, include_chevron_bonds = False,timestep_index=-2):
    """
    Creates a LAMMPS restart/data file excluding specified molecules.
    
    Parameters:
        atom_dump_file (str): Path to LAMMPS atom dump file.
        bond_dump_file (str or None): Path to LAMMPS bond dump file (optional).
        exclude_molecules (list of int): List of molecule IDs to exclude.
        output_file (str): Output restart file path.
        timestep_index (int): Timestep index to read (-1 for last timestep).
    """
    # Read atoms
    atoms_data, box_bounds = read_lammps_dump(atom_dump_file, timestep_index)
    xlo, xhi, ylo, yhi = box_bounds
    zlo, zhi = -0.25, 0.25  # 2D simulation
    xlo, xhi = -1000,1000
    #total_length = xhi-xlo
    ylo,yhi = -1000,1000
    #ylo, yhi = -total_length/2, total_length/2
    
    #get mol ids of which molecules to exclude - monomers and not fully bound to polymer
    exclude_molecules, exclude_atoms = molecules_to_remove(atom_dump_file,which_simulation = which_simulation,timestep_index=timestep_index)
    
    #remaining number of atoms
    filtered_atoms = [atom for atom in atoms_data if atom[1] not in exclude_molecules]
    filtered_atoms = sorted(filtered_atoms, key=lambda x: x[0]) #sort atom ids in order

    # If bonds are provided, read and filter them
    filtered_bonds = []
    if bond_dump_file is not None:
        bonds_data, _ = read_lammps_bonds_dump(bond_dump_file, timestep_index)
        new_bond_id = 1
        for bond_id, atom1, atom2, bond_type in bonds_data:
            if atom1 not in exclude_atoms and atom2 not in exclude_atoms:
                filtered_bonds.append((new_bond_id, bond_type, atom1, atom2))
                new_bond_id += 1
                
    if include_chevron_bonds:
        extra_bonds = add_extra_chevron_bond(atoms_data, exclude_molecules)
        for atom1, atom2 in extra_bonds:
            if atom1 not in exclude_atoms and atom2 not in exclude_atoms:
                filtered_bonds.append((new_bond_id, bond_type, atom1, atom2))
                new_bond_id += 1

    # Write restart/data file
    with open(output_file, "w") as f:
        f.write("LAMMPS data file via Python\n\n")
        f.write(f"{len(filtered_atoms)} atoms\n")
        f.write(f"{len(filtered_bonds)} bonds\n" if bond_dump_file else "")
        
        # Infer types automatically
        atom_types = set(a[2] for a in filtered_atoms)
        bond_types = set(b[1] for b in filtered_bonds) if bond_dump_file else set()
        f.write(f"11 atom types\n")
        if bond_dump_file:
            f.write(f"{len(bond_types)} bond types\n")
        f.write(f"\n{xlo} {xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n\n")
        
        f.write("Masses\n\n")
        for atom_type in range(1,12):
            f.write(f"{atom_type} 1.0\n")
        
        f.write("\nAtoms\n\n")
        mapping_atomid=[]
        for i,atom in enumerate(filtered_atoms):
            atom_id, mol_id, atom_type, x, y = atom
            mapping_atomid.append((atom_id,i+1))
            f.write(f"{i+1} {mol_id} {atom_type} {x} {y} 0.0\n")  # z = 0

        if filtered_bonds:
            f.write("\nBonds\n\n")
            for bond in filtered_bonds:
                atom1_old, atom2_old = bond[2], bond[3]
                atom1_new = next(new_id for old_id, new_id in mapping_atomid if old_id == atom1_old)
                atom2_new = next(new_id for old_id, new_id in mapping_atomid if old_id == atom2_old)
                f.write(f"{bond[0]} {bond[1]} {atom1_new} {atom2_new}\n")
                
    return xlo,xhi,ylo,yhi


    
def read_number_of_types_from_atoms(dumpfile):
    atoms = []
    with open(dumpfile) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "ITEM: ATOMS" in lines[i]:
            headers = lines[i].split()[2:]
            n_atoms = int(lines[i-3])  # assumes standard dump with "NUMBER OF ATOMS" 2 lines before
            atoms = [dict(zip(headers, lines[i+1+j].split())) for j in range(n_atoms)]
            i += n_atoms + 1
        else:
            i += 1
    # Get unique atom types
    atom_types = set(int(a['type']) for a in atoms)
    return len(atom_types)

    
    
#functions to see which molecules are free - not in polymer

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

def read_lammps_bonds_dump(filename, timestep_index=-1):
    """
    Reads a LAMMPS bonds dump file and extracts bond data for a specific timestep.

    Parameters:
        filename (str): Path to the bonds dump file.
        timestep_index (int): Index of the timestep to read (-1 for the last timestep).

    Returns:
        bonds_data (list of tuples): Each tuple contains (bond_id, atom1, atom2, bond_type)
        box_bounds (tuple): Box boundaries as (xlo, xhi, ylo, yhi, zlo, zhi)
    """
    timestep_start = "ITEM: TIMESTEP"
    box_bounds_start = "ITEM: BOX BOUNDS"
    bonds_start = "ITEM: ENTRIES"

    bonds_data = []
    box_bounds = None
    timestep_indices = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Identify all timesteps
    for i, line in enumerate(lines):
        if line.strip() == timestep_start:
            timestep_indices.append(i)

    if not timestep_indices:
        raise ValueError("No timesteps found in the bonds file.")
    if timestep_index == -1:
        timestep_index = len(timestep_indices) - 1
    if timestep_index >= len(timestep_indices):
        raise IndexError(f"Requested timestep_index {timestep_index} is out of range.")

    timestep_start_index = timestep_indices[timestep_index]

    # Parse box bounds
    box_bounds_line = timestep_start_index + 5
    box_bounds = []
    for i in range(3):  # Read 3 lines for x, y, z bounds
        bounds = tuple(map(float, lines[box_bounds_line + i].strip().split()))
        box_bounds.extend(bounds)
    box_bounds = tuple(box_bounds)

    # Parse number of entries (bonds)
    num_entries_line = timestep_start_index + 3
    num_entries = int(lines[num_entries_line].strip())

    # Parse bond data
    bonds_start_line = timestep_start_index + 9  # Adjust if necessary
    bonds_data = []
    bond_id = 1
    for line in lines[bonds_start_line : bonds_start_line + num_entries]:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError("Malformed bond line: " + line)
        atom1, atom2, bond_type = map(int, parts[:3])
        bonds_data.append((bond_id, atom1, atom2, bond_type))
        bond_id += 1

    return bonds_data, box_bounds

def add_extra_chevron_bond(atoms_data, exclude_molecules):

    #find and add extra bonds for every chevron molecule
    extra_bonds = []
    atoms_to_be_connected = []
    molecules_present = []
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y = atom
        if mol_id not in exclude_molecules:
            if mol_id not in molecules_present:
                molecules_present.append(mol_id)
            if atom_type == 3 or atom_type == 4: #if its chevron part
                atoms_to_be_connected.append((atom_id, mol_id, atom_type))
                
    for mol_id in molecules_present:
        chevron_atoms = [atom for atom in atoms_to_be_connected if atom[1] == mol_id]
        if len(chevron_atoms) == 2:
            extra_bonds.append((chevron_atoms[0][0], chevron_atoms[1][0])) #add bond between the two chevron atoms
            
    return extra_bonds
    

def read_lammps_dump(filename, timestep_index):
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
    molecules = []
    for line in lines[atoms_data_start:atoms_data_start + num_atoms]:
        parts = line.strip().split()
        if len(parts) < 5:  # Ensure all components (atom_id, mol_id, type, x, y, z) are present
            raise ValueError("Malformed atoms data line: " + line)
        atom_id = int(parts[0])
        mol_id = int(parts[1])
        atom_type = int(parts[2])
        x, y = map(float, parts[3:5])
        atoms_data.append((atom_id, mol_id, atom_type, x, y))
        if mol_id not in molecules:
            molecules.append(mol_id)
            
    max_mol_id = max(molecules)
    all_molecules_no_skipping = list(range(1, max_mol_id+1))
    skipped_molecules = set(all_molecules_no_skipping) - set(molecules)
    
    for i,atom in enumerate(atoms_data):
        atom_id, mol_id, atom_type, x, y = atom
        reduce_mol_id = 0
        for skipped_mol in skipped_molecules:
            if mol_id > skipped_mol:
                reduce_mol_id += 1
        atoms_data[i] = (atom_id, mol_id-reduce_mol_id, atom_type, x, y)        

    return atoms_data, box_bounds


def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds, which_simulation):
    xlo, xhi, ylo, yhi = box_bounds
    for atom1 in molecule1:
        for atom2 in molecule2:
            x1, y1, atom_type1 = atom1[1], atom1[2], atom1[3]
            x2, y2, atom_type2 = atom2[1], atom2[2], atom2[3]
            
            dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
            dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < cutoff :
                if which_simulation == 'GA':
                    if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1):
                        return True
                # elif which_simulation == 'littleguys': #for full binding make sure both 1-2 and 6-7 interface exists
                #     if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 1 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 1) or (atom_type1 == 2 and atom_type2 == 8) or (atom_type1 == 8 and atom_type2 == 2) or (atom_type1 == 8 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 8):
                #         for atom1 in molecule1:
                #             for atom2 in molecule2:
                #                 x1, y1, atom_type1 = atom1[1], atom1[2], atom1[3]
                #                 x2, y2, atom_type2 = atom2[1], atom2[2], atom2[3]
                                
                #                 dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
                #                 dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
                #                 distance = np.sqrt(dx**2 + dy**2)
                                
                #                 if distance < cutoff :
                #                     if (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6) or (atom_type1 == 6 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 6) or (atom_type1 == 7 and atom_type2 == 10) or (atom_type1 == 10 and atom_type2 == 7) or (atom_type1 == 10 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 10):
                #                         return True
                elif which_simulation == 'littleguys': #for full binding make sure both 1-2 and 6-7 interface exists
                    if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 1 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 1) or (atom_type1 == 2 and atom_type2 == 8) or (atom_type1 == 8 and atom_type2 == 2) or (atom_type1 == 8 and atom_type2 == 9) or (atom_type1 == 9 and atom_type2 == 8) or (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6) or (atom_type1 == 6 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 6) or (atom_type1 == 7 and atom_type2 == 10) or (atom_type1 == 10 and atom_type2 == 7) or (atom_type1 == 10 and atom_type2 == 11) or (atom_type1 == 11 and atom_type2 == 10):
                        return True
    return False

def get_bonds(output_file_path,timestep_index,which_simulation):
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
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds,which_simulation): #if theyre touching move
                    bonds.append((mol_id1, mol_id2))
                    
    #print('BONDS BETWEEN:')
    #print(len(bonds))
    #print(bonds)
    return bonds,atoms_data,molecules


# Example function that returns molecules to exclude
def molecules_to_remove(output_file_positions,which_simulation,timestep_index=-1):
    # e.g., remove all molecules with type 2
    bonds,atoms_data,molecules = get_bonds(output_file_positions,timestep_index=timestep_index,which_simulation=which_simulation)
    num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
    clusters = find_clusters(range(1,num_of_molecules+1),bonds)
    filtered_clusters = [cluster for cluster in clusters if cluster[0] in molecules and len(cluster) > 1][0] #filter out monomers
    mol_ids = []
    atom_ids = []
    for entry in atoms_data:
        atom_id, mol_id, atom_type, x, y = entry
        if mol_id not in filtered_clusters:  # or any custom condition
            mol_ids.append(mol_id)
            atom_ids.append(atom_id)
    return mol_ids, atom_ids

def write_input_file_littleguys(eps, seed, spath, dump_steps=10000, run_steps=1000000):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary f f p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/restart_config.dat",
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
        #LITTLE GUYS SETUP - unspecific binding sites
        #"pair_coeff 2 2 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        #"pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca",
        #"pair_coeff 3 1 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #"pair_coeff 4 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #"compute cPE all pe/atom pair",
        #f"fix boundary all wall/reflect xlo {low} xhi {high} ylo {low} yhi {high}",
        "fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        "fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        "fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0",
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
        "timestep 0.01",
        f"run {run_steps}"
    ]
    
    # Write to a text file
    with open(sim_path_new+"/lammps_init.in", "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
def write_input_file_GA_outputs(eps, seed, spath, dump_steps=10000, run_steps=1000000):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary f f p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/restart_config.dat",
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
        "fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        "fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        "fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0",
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
        "timestep 0.01", 
        f"run {run_steps}"
    ]
    
    # Write to a text file
    with open(sim_path_new+"/lammps_init.in", "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")


