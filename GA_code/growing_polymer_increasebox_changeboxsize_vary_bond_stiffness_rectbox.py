import lammps
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import write_lammps_more_monomers_3_types_vary_bond_stiffness
import ast
import get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer
import subprocess
import signal

def get_bonds(positions, molecules,types,atom_ids,Lbox,cutoff_distance):
    box_bounds = (-Lbox,Lbox,-Lbox,Lbox)

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
        
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds): #if theyre touching move
                    bonds.append((mol_id1, mol_id2))
                    
    #print('BONDS BETWEEN:')
    #print(len(bonds))
    #print(bonds)
    return bonds,atoms_data,box_bounds


def get_most_left_and_right_com(pos,mol):
    #gets positions of most left and right monomers in polymer for insertion of new monomers

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

    # center of mass of the molecule with max x
    com_x_max_mol = np.mean(max_mol_x)
    # center of mass of the molecule with min x
    com_x_min_mol = np.mean(min_mol_x)

    return com_x_max_mol,com_x_min_mol

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

def graph_new(positions, molecules,types,atom_ids,Lbox,cutoff_distance,num_of_molecules):
    bonds,atoms_data,_ = get_bonds(positions, molecules,types,atom_ids,Lbox,cutoff_distance)
    clusters = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.find_clusters(range(1,num_of_molecules+1),bonds)
    
    mol_set = set(molecules)
    filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1 or cluster[0] in mol_set]
    return filtered_clusters,atoms_data,bonds

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

def setup_simulation(lmp, spath, seed, dump_steps, eps):
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


def analyse_simulation(step, pos, mol, cutoff, L, types,atom_ids,num_of_molecules, print_mode=True):
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
    free_monomers = []
    left_monomers_num = 0
    right_monomers_num = 0
    left_monomers_mol_list = []
    right_monomers_mol_list = []
    clusters,atoms_data,bonds = graph_new(pos, mol, types,atom_ids,L,cutoff, num_of_molecules)
    max_length = max([len(sublist) for sublist in clusters])
    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    for monomer in free_monomers:
        try: #deleting excess monomers 
            com = center_of_mass(atoms_data, target_mol_id = monomer)
            comx = com[0]
            if comx<0:
                left_monomers_num+=1
                left_monomers_mol_list.append(monomer)
            elif comx>0:
                right_monomers_num+=1
                right_monomers_mol_list.append(monomer)
        except:
            continue

    bonds = set(bonds)  # Convert list to set
    dimer = False
    if (1,2) in bonds or (2,1) in bonds:
        dimer = True

    if print_mode:
        print('Step:', step, '- Frame:', step/1000)
        print('Monomers left in left side:', monomers_free_num)
        print()

    return left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters

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

def run(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal,seed, dump_steps, eps, delta_L=5):#delta_L was 12 for size 5, using 6 for 4x4 for now. 9 for 5X5
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
    #start_time = time.time()  # Record the start time
    #try: 
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])  # Instantiate LAMMPS eithout output

    #lmp = lammps.lammps()

    # Read file
    #lmp.file('in.setup')
    setup_simulation(lmp, sim_dir, seed, dump_steps, eps)

    # Get box size
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")

    # Get positions, molecules and ids
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    ids = lmp.numpy.extract_atom("id")

    #find number of monomers
    molecule_number = 2

    tethered = 0

    # Find the size of the monomer
    monomer_size = len(pos[mol == 1])

    types = lmp.numpy.extract_atom("type")

    # Initial graph
    step = lmp.get_thermo('step')
    left_monomers_num, right_monomers_num,dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
    counter = 1
    new_len = 13

    # Timeout configuration
    timeout_duration = 3000  # Maximum runtime (e.g., 5 hours = 18000 seconds)
    stall_timeout = 3000  # Time limit for detecting stalls (e.g., 10 minutes = 600 seconds)
    dump_file_path = sim_dir + '/output_com.xyz'

    #current_mol_number = 10 #random big number, higher than 2 to make the bug of fix deopsit not depoisting at the exact step of polymerization disappear

    # Initialize the simulation start time and track the last timestep
    start_time = time.time()
    last_timestep = 0
    last_timestep_time = time.time()  # Time when the last timestep was checked

    # Iterate through loops
    for i in range(loops):
    #for i in tqdm(range(loops)):
        # Run for run_interval steps
        # Check if the total elapsed time exceeds the timeout
        
        elapsed_time = time.time() - start_time
        """
        if elapsed_time > timeout_duration:
            print(f"Timeout reached after {elapsed_time} seconds at loop {i}. Terminating LAMMPS simulation at " + sim_dir, flush=True)
            lmp.command("quit")  # Gracefully terminate the simulation
            return False  # Indicate failure (simulation didn't finish successfully)

        # Check if the timestep has changed since the last iteration
        current_timestep = get_last_timestep(dump_file_path)
        #print('Current timestep' + str(current_timestep))
        if current_timestep == last_timestep:
            # If timestep hasn't changed, check if the stall time has been exceeded
            if time.time() - last_timestep_time > stall_timeout:
                print(f"Stalled for too long at loop {i}, terminating LAMMPS simulation at " + sim_dir, flush=True)
                lmp.command("quit")  # Gracefully terminate the simulation
                return False  # Indicate failure (simulation is stuck)
        else:
            # If timestep has advanced, update the last timestep and reset the stall timer
            last_timestep = current_timestep
            last_timestep_time = time.time()
        """
        lmp.command('run %d'%(run_interval))
        # Analyse the system
        L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        step = lmp.get_thermo('step')
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")
        types = lmp.numpy.extract_atom("type")
        molecule_number = max(mol)
        left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
        """
        if dimer and tethered == 0:
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x final %f %f \n\n' %(-new_len, new_len))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('run 1')
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            lmp.command('run 1')
            tethered = 1
        """
        if left_monomers_num < nMonsSol and dimer:
            nmonomers = nMonsSol-left_monomers_num
            #delta_L = ((L_end-init_region_L)/(max_len_goal-2))/2
            #delta_L = 3
            #current_region_L = new_len + delta_L*(molecule_number - 1)
            """
            if counter==1:
                lmp.command('unfix boundary')
            else:
                lmp.command('unfix boundary_%d'%(counter-1))
            lmp.command('fix                 boundary_%d all wall/reflect xlo %f xhi %f ylo %f yhi %f \n\n'%(counter,-current_region_L, current_region_L, -current_region_L, current_region_L))
            """
            lmp.command('unfix xwalls')
            #com_x_max, com_x_min = get_most_left_and_right_com(pos,mol)
            lmp.command('change_box          all x delta %f %f \n\n' %(-delta_L, 0))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            #lmp.command('fix remove_rot mol1and2 momentum 100 angular')
            lmp.command('run 1')
            # Extract box size
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo  # Lower bounds
            xhi, yhi, zhi = boxhi  # Upper bounds
        
            lmp.command("region              insertion_region_left_%d block %f %f -3 3 EDGE EDGE units box" %(counter,xlo+delta_L, 0))
            lmp.command('fix                 add_mons moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left_%d near 7'%(nmonomers,seed,counter))
            #lmp.command('minimize 1.0e-4 1.0e-6 10 100')
            counter +=1
            lmp.command('run 10')
        if right_monomers_num < nMonsSol and dimer:
            nmonomers = nMonsSol-right_monomers_num
            #delta_L = ((L_end-init_region_L)/(max_len_goal-2))/2
            #delta_L=3
            #current_region_L = new_len + delta_L*(molecule_number - 1)
            """
            if counter==1:
                lmp.command('unfix boundary')
            else:
                lmp.command('unfix boundary_%d'%(counter-1))
            lmp.command('fix                 boundary_%d all wall/reflect xlo %f xhi %f ylo %f yhi %f \n\n'%(counter,-current_region_L, current_region_L, -current_region_L, current_region_L))
            """
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x delta %f %f \n\n' %(0, delta_L))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            #lmp.command('fix remove_rot mol1and2 momentum 100 angular')
            lmp.command('run 1')
            # Unpacking values
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo  # Lower bounds
            xhi, yhi, zhi = boxhi  # Upper bounds
            
            lmp.command("region              insertion_region_right_%d block %f %f -3 3 EDGE EDGE units box" %(counter,0, xhi-delta_L))
            lmp.command('fix                 add_mons moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d near 7'%(nmonomers,seed,counter))
            #lmp.command('minimize            1.0e-4 1.0e-6 10 100')
            counter +=1
            lmp.command('run 10')
        """
        f dimer and monomers_free_num==0:
            delta_L = 10
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x delta %f %f \n\n' %(-delta_L, delta_L))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('run 10')
        """
        """
        if left_monomers_num > nMonsSol: #delete monomer if more monomers than the desired concentration on the left

            max_mol_left = max(left_monomers_mol_list) # remove the last added monomer

            lmp.command("group max_mol_left molecule %d" %(max_mol_left))
            lmp.command("delete_atoms group max_mol_left")

            lmp.command('run 10')

        if right_monomers_num > nMonsSol: #delete monomer if more monomers than the desired concentration on the left

            max_mol_right = max(right_monomers_mol_list) # remove the last added monomer

            lmp.command("group max_mol_right molecule %d" %(max_mol_right))
            lmp.command("delete_atoms group max_mol_right")

            lmp.command('run 10')
        """
    # Final graph
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    step = lmp.get_thermo('step')
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    ids = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")
    #molecule_number = max(mol)
    left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
    
    # Clean up
    lmp.close()
    
    print(f" LAMMPS simulation completed in {elapsed_time:.2f} seconds.")
    return True
    """
    except Exception as e:
        print(f"Error during LAMMPS simulation: {e}")
        return False
    """
    

def run_tryout_onepolymer(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal,seed, dump_steps, eps, delta_L=5):#delta_L was 12 for size 5, using 6 for 4x4 for now. 9 for 5X5
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
    #start_time = time.time()  # Record the start time
    #try: 
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])  # Instantiate LAMMPS eithout output

    #lmp = lammps.lammps()

    # Read file
    #lmp.file('in.setup')
    setup_simulation(lmp, sim_dir, seed, dump_steps, eps)

    # Get box size
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")

    # Get positions, molecules and ids
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    ids = lmp.numpy.extract_atom("id")

    #find number of monomers
    molecule_number = 2

    tethered = 0

    # Find the size of the monomer
    monomer_size = len(pos[mol == 1])

    types = lmp.numpy.extract_atom("type")

    # Initial graph
    step = lmp.get_thermo('step')
    left_monomers_num, right_monomers_num,dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
    counter = 1
    new_len = 13

    # Timeout configuration
    timeout_duration = 5000  # Maximum runtime (e.g., 5 hours = 18000 seconds)
    stall_timeout = 5000  # Time limit for detecting stalls (e.g., 10 minutes = 600 seconds)
    dump_file_path = sim_dir + '/output_com.xyz'

    #current_mol_number = 10 #random big number, higher than 2 to make the bug of fix deopsit not depoisting at the exact step of polymerization disappear

    # Initialize the simulation start time and track the last timestep
    start_time = time.time()
    last_timestep = 0
    last_timestep_time = time.time()  # Time when the last timestep was checked

    # Iterate through loops
    for i in range(loops):
    #for i in tqdm(range(loops)):
        # Run for run_interval steps
        # Check if the total elapsed time exceeds the timeout
        
        elapsed_time = time.time() - start_time
        """
        if elapsed_time > timeout_duration:
            print(f"Timeout reached after {elapsed_time} seconds at loop {i}. Terminating LAMMPS simulation at " + sim_dir, flush=True)
            lmp.command("quit")  # Gracefully terminate the simulation
            return False  # Indicate failure (simulation didn't finish successfully)

        # Check if the timestep has changed since the last iteration
        current_timestep = get_last_timestep(dump_file_path)
        #print('Current timestep' + str(current_timestep))
        if current_timestep == last_timestep:
            # If timestep hasn't changed, check if the stall time has been exceeded
            if time.time() - last_timestep_time > stall_timeout:
                print(f"Stalled for too long at loop {i}, terminating LAMMPS simulation at " + sim_dir, flush=True)
                lmp.command("quit")  # Gracefully terminate the simulation
                return False  # Indicate failure (simulation is stuck)
        else:
            # If timestep has advanced, update the last timestep and reset the stall timer
            last_timestep = current_timestep
            last_timestep_time = time.time()
        """
        lmp.command('run %d'%(run_interval))
        # Analyse the system
        L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        step = lmp.get_thermo('step')
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")
        types = lmp.numpy.extract_atom("type")
        molecule_number = max(mol)
        left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
        """
        if dimer and tethered == 0:
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x final %f %f \n\n' %(-new_len, new_len))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('run 1')
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            lmp.command('run 1')
            tethered = 1
        """
        if left_monomers_num < nMonsSol:
            nmonomers = nMonsSol-left_monomers_num
            #delta_L = ((L_end-init_region_L)/(max_len_goal-2))/2
            #delta_L = 3
            #current_region_L = new_len + delta_L*(molecule_number - 1)
            """
            if counter==1:
                lmp.command('unfix boundary')
            else:
                lmp.command('unfix boundary_%d'%(counter-1))
            lmp.command('fix                 boundary_%d all wall/reflect xlo %f xhi %f ylo %f yhi %f \n\n'%(counter,-current_region_L, current_region_L, -current_region_L, current_region_L))
            """
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x delta %f %f \n\n' %(-delta_L, 0))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            #lmp.command('fix remove_rot mol1and2 momentum 100 angular')
            lmp.command('run 1')
            # Extract box size
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo  # Lower bounds
            xhi, yhi, zhi = boxhi  # Upper bounds
        
            lmp.command("region              insertion_region_left_%d block %f %f -3 3 EDGE EDGE units box" %(counter,xlo+delta_L, 0))
            lmp.command('fix                 add_mons moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left_%d near 7'%(nmonomers,seed,counter))
            #lmp.command('minimize 1.0e-4 1.0e-6 10 100')
            counter +=1
            lmp.command('run 10')
        if right_monomers_num < nMonsSol:
            nmonomers = nMonsSol-right_monomers_num
            #delta_L = ((L_end-init_region_L)/(max_len_goal-2))/2
            #delta_L=3
            #current_region_L = new_len + delta_L*(molecule_number - 1)
            """
            if counter==1:
                lmp.command('unfix boundary')
            else:
                lmp.command('unfix boundary_%d'%(counter-1))
            lmp.command('fix                 boundary_%d all wall/reflect xlo %f xhi %f ylo %f yhi %f \n\n'%(counter,-current_region_L, current_region_L, -current_region_L, current_region_L))
            """
            lmp.command('unfix xwalls')
            lmp.command('change_box          all x delta %f %f \n\n' %(0, delta_L))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('fix put_dimer_in_middle mol1and2 recenter 0.0 0.0 0.0')
            #lmp.command('fix remove_rot mol1and2 momentum 100 angular')
            lmp.command('run 1')
            # Unpacking values
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo  # Lower bounds
            xhi, yhi, zhi = boxhi  # Upper bounds
            
            lmp.command("region              insertion_region_right_%d block %f %f -3 3 EDGE EDGE units box" %(counter,0, xhi-delta_L))
            lmp.command('fix                 add_mons moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right_%d near 7'%(nmonomers,seed,counter))
            #lmp.command('minimize            1.0e-4 1.0e-6 10 100')
            counter +=1
            lmp.command('run 10')

        # Analyse the system
        L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        step = lmp.get_thermo('step')
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")
        types = lmp.numpy.extract_atom("type")
        molecule_number = max(mol)
        left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, 2, L, types, ids,molecule_number, print_mode) #CUTOFF IS SET TO A HIGHER VALUE - 1.5 TO DELETE PARTICLES ONLY WHEN THEY REALLY ESCAPE ATTRACTION
        
        #getting number of polymers, free monomers from the clusters above with HIGHER CUTOFF
        polymer_number = sum(1 for sublist in clusters if len(sublist) > 1)
        #print('CLUSTERS')
        #print(clusters)
        #print('mols')
        #print(mol)
        ##print('step')
        #print(step)
        
        max_polymer_size = max(clusters, key=len) #if breaking occurs keep only the longest filament
        monomer_number = sum(1 for sublist in clusters if len(sublist) == 1)
        flattened_mols = [item for sublist in clusters for item in sublist]
        if len(flattened_mols) ==1: 
            monomer_number_max = 3
        else:
            monomer_number_max = 2

        molecules_to_delete = []
        if len(max_polymer_size) >1:
            if polymer_number>1 or monomer_number>monomer_number_max:
                for molecule in flattened_mols:
                    if molecule not in max_polymer_size: #if polymer num>1 == breaking or moonmer number>2 == depolymerization from the ends delete everything not in the longest polymer
                        molecules_to_delete.append(molecule)

                for mols in molecules_to_delete:
                    if mols in mol:
                        lmp.command('group mol_%d molecule %d'%(mols,mols))
                        lmp.command('delete_atoms group mol_%d'%(mols))
                        lmp.command('group mol_%d delete'%(mols)) #delete the group after deleting the atoms

                lmp.command('run 10')

                #after deleting the molecules, we decrease the size of the box to just about the size of the polymer

                pos = lmp.numpy.extract_atom("x")
                mol = lmp.numpy.extract_atom("molecule")

                com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(pos,mol)

                lmp.command('unfix xwalls') #make new walls to fit the remaining polymer
                x_walls_lo, x_walls_hi = com_x_min_mol - 10, com_x_max_mol + 10
                lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
                lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE" )
                lmp.command('run 1')
        
        
    # Final graph
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    step = lmp.get_thermo('step')
    pos = lmp.numpy.extract_atom("x")
    mol = lmp.numpy.extract_atom("molecule")
    ids = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")
    #molecule_number = max(mol)
    left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
    
    # Clean up
    lmp.close()
    
    print(f" LAMMPS simulation completed in {elapsed_time:.2f} seconds.")
    return True


def write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps):
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
    info_file.write(f'eps: {eps}\n')
    info_file.write(f'Lbox: {Lbox}\n')
    info_file.write(f'nMonsSol: {nMonsSol}\n')
    info_file.write(f'seeds: {seeds}\n')
    info_file.write(f'cutoff: {cutoff}\n')
    info_file.write(f'run_interval: {run_interval}\n')
    info_file.write(f'run_steps: {run_steps}\n')
    info_file.write(f'dump_steps: {dump_steps}\n')
    info_file.write(f'debug_mode: {debug_mode}\n')
    info_file.write(f'print_mode: {print_mode}\n')
    info_file.close()

def run_simulation_with_retries(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height,max_retries=50):
    attempt = 0
    while attempt < max_retries:
        try:
            # Set timeout (e.g., 5 hours)
            timeout_seconds = 2000
            signal.alarm(timeout_seconds)

            # Run your simulation
            single_seed(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height)# <-- your function

            # If successful, cancel alarm and exit
            signal.alarm(0)
            print("Simulation finished successfully.")
            return True

        except Exception as e:
            # If any exception occurs (including timeout), handle it
            attempt += 1
            print(f"Exception occurred: {str(e)}. Restarting simulation (attempt {attempt}/{max_retries})...")

    print("Maximum retries reached. Simulation failed.")
    return False

def run_simulation_with_retries_postanalysis(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height,max_retries=50):
    attempt = 0
    while attempt < max_retries:
        try:
            # Set timeout (e.g., 5 hours)
            timeout_seconds = 180000 #50 hours
            signal.alarm(timeout_seconds)

            # Run your simulation
            single_seed_postanalysis(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height)# <-- your function

            # If successful, cancel alarm and exit
            signal.alarm(0)
            print("Simulation finished successfully.")
            return True

        except Exception as e:
            # If any exception occurs (including timeout), handle it
            attempt += 1
            print(f"Exception occurred: {str(e)}. Restarting simulation (attempt {attempt}/{max_retries})...")

    print("Maximum retries reached. Simulation failed.")
    return False

def single_seed(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height):
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
    #if not os.path.exists(spath):
    #    r = os.makedirs(spath, exist_ok=True)
    #print()
    #print('Simulation folder:', spath)

    # Create monomer template
    #print('THIS IS THE PATH TO IND TEXT FILE')
    #print(path_to_ind_text_file)

    success = False
    while not success:
        ind,random_n_type = read_individual_txt_file(os.path.join(path_to_ind_text_file, 'individual.txt'))
        #write_monomer_template(spath)
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps_one_monomer(ind,spath, init_region_L_width, init_region_L_height, random_n_type)

        # Setup the simulation and write the configuration.dat file
        #print()
        #print('Starting simulation setup...')
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps(ind,gpath = spath, box_width=init_region_L_width, box_height=init_region_L_height, number_of_monomers = 2,random_n=random_n_type)
        #print('Setup done, 2 monomers made and cofiguratoin.dat file written!')
        #print()

        # Write lammps input file
        '''
        print()
        print('Writing input file...')
        write_input_file(spath, seed, eps, dump_steps, init_region_L/2)
        print('Input file written!')
        print()
        '''

        # Run simulation
        #print()
        #print('Starting simulation...')
        #print()
        #start_time = time.time()
        success = run(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal, seed, dump_steps, eps)


    print("Simulation finished successfully.")
    
def single_seed_postanalysis(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode,path_to_ind_text_file,max_len_goal,init_region_L_width, init_region_L_height):
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
    #if not os.path.exists(spath):
    #    r = os.makedirs(spath, exist_ok=True)
    #print()
    #print('Simulation folder:', spath)

    # Create monomer template
    #print('THIS IS THE PATH TO IND TEXT FILE')
    #print(path_to_ind_text_file)

    success = False
    while not success:
        ind,random_n_type = read_individual_txt_file(os.path.join(path_to_ind_text_file, 'individual.txt'))
        #write_monomer_template(spath)
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps_one_monomer(ind,spath, init_region_L_width, init_region_L_height, random_n_type)

        # Setup the simulation and write the configuration.dat file
        #print()
        #print('Starting simulation setup...')
        write_lammps_more_monomers_3_types_vary_bond_stiffness.write_lammps(ind,gpath = spath, box_width=init_region_L_width, box_height=init_region_L_height, number_of_monomers = 2,random_n=random_n_type)
        #print('Setup done, 2 monomers made and cofiguratoin.dat file written!')
        #print()

        # Write lammps input file
        '''
        print()
        print('Writing input file...')
        write_input_file(spath, seed, eps, dump_steps, init_region_L/2)
        print('Input file written!')
        print()
        '''

        # Run simulation
        #print()
        #print('Starting simulation...')
        #print()
        #start_time = time.time()
        success = run_tryout_onepolymer(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal, seed, dump_steps, eps)


    print("Simulation finished successfully.")


    '''
    elapsed_time = end_time-start_time
    if elapsed_time > 3600:
        print('Elapsed time:', int(np.floor(elapsed_time/3600)), 'hours', int(np.floor((elapsed_time%3600)/60)), 'minutes', elapsed_time%60, 'seconds.')
    elif elapsed_time > 60:
        print('Elapsed time:', int(np.floor(elapsed_time/60)), 'minutes', elapsed_time%60, 'seconds.')
    else:
        print('Elapsed time:', elapsed_time, 'seconds.')
    print()
    '''
"""
def parse_arguments(): #done, adapted
    parser = argparse.ArgumentParser(description='Run the LAMMPS simulation of the chevron-like molecules')
    parser.add_argument('--Lbox', type=float, default=60, help='Length of the box in the x and y directions')
    parser.add_argument('--init_region_L', type=float, default=20, help='Length of the box in the x and y directions')
    parser.add_argument('--max_length_goal', type=int, default=10, help='Length of the end monomer')
    parser.add_argument('--nMonsSol', type=int, default=1, help='Number of monomers in solution (per side of the box or reservoir)')
    parser.add_argument('--seed', type=int, default=23451, help='Seed for the random number generator')
    parser.add_argument('--cutoff', type=float, default=1.05, help='Distance cutoff criterion for the graph')
    parser.add_argument('--run_interval', type=int, default=100, help='Number of steps to run the simulation for each iteration')
    parser.add_argument('--run_steps', type=int, default=50000000, help='Total number of steps to run the simulation')
    parser.add_argument('--dump_steps', type=int, default=1000, help='Dumping interval in simulation steps')
    parser.add_argument('--debug_mode', action='store_true', help='Flag to enable or disable LAMMPS outputs')
    parser.add_argument('--eps', type=float, default=10.0, help='Binding constant for the monomer-monomer interaction')
    parser.add_argument('--print_mode', action='store_true', help='Flag to enable or disable print statements')
    parser.add_argument('--Nseeds', type=int, default=1, help='Number of seeds to run')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory where the simulation folders will be created')
    parser.add_argument('--path_ind', type=str, required=True, help='Path to the directory where the individual.txt file is')
    return parser.parse_args()

if __name__ == '__main__': #only this one left
    # Parse arguments from command line
    args = parse_arguments()

    # Define the seeds to run
    if args.Nseeds == 1:
        seeds = [args.seed]
    else:
        np.random.seed(args.seed)
        seeds = np.random.randint(10000, 99999, args.Nseeds)
        while len(np.unique(seeds)) < args.Nseeds:
            seeds = np.random.randint(10000, 99999, args.Nseeds)

    # Create the simulation folders
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # Write the information file
    write_info_file(args.path, args.Lbox, args.nMonsSol, seeds, args.cutoff, args.run_interval, args.run_steps, args.dump_steps, args.debug_mode, args.print_mode, args.eps)

    # Define number of loops
    loops = args.run_steps//args.run_interval

    # Run the simulations
    for seed in seeds:
        spath = f'{args.path}/sd{seed}'
        single_seed(spath, args.Lbox, args.nMonsSol, seed, args.cutoff, loops, args.run_interval, args.debug_mode, args.dump_steps, args.eps, args.print_mode, args.path_ind,args.max_length_goal,args.init_region_L)
"""

def run_simulation(path, path_ind,Lbox=60, max_length_goal=10, nMonsSol=1, seed=23451, cutoff=1.1, run_interval=100, run_steps=2000000, dump_steps=10000, debug_mode=True,eps=30, print_mode=False, Nseeds=1, init_region_L_width = 15, init_region_L_height=15):#was 15x15 for size 4, 20X15 for size 5
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")
    success_of_run = False
    while not success_of_run:
        seeds = np.random.randint(1, 999999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(1, 999999, Nseeds)

        seed = seeds[0]

        # Create the simulation folders
        #if not os.path.exists(path):
        #    os.makedirs(path, exist_ok=True)

        # Write the information file
        write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps)

        # Define number of loops
        loops = run_steps//run_interval

        # Run the simulations
        # Define custom exception
        class TimeoutException(Exception):
            pass

        # Define signal handler
        def timeout_handler(signum, frame):
            raise TimeoutException()

        # Set the signal handler once
        signal.signal(signal.SIGALRM, timeout_handler)

        success_of_run = run_simulation_with_retries(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)
        #single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)


def run_simulation_postanalysis(path, path_ind,Lbox=60, max_length_goal=10, nMonsSol=1, seed=23451, cutoff=1.1, run_interval=100, run_steps=100000000, dump_steps=10000, debug_mode=True,eps=30, print_mode=False, Nseeds=1, init_region_L_width = 15, init_region_L_height=15):
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")
    success_of_run = False
    while not success_of_run:
        seeds = np.random.randint(1, 999999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(1, 999999, Nseeds)

        seed = seeds[0]

        # Create the simulation folders
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Write the information file
        write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps)

        # Define number of loops
        loops = run_steps//run_interval

        # Run the simulations
        # Define custom exception
        class TimeoutException(Exception):
            pass

        # Define signal handler
        def timeout_handler(signum, frame):
            raise TimeoutException()

        # Set the signal handler once
        signal.signal(signal.SIGALRM, timeout_handler)

        success_of_run = run_simulation_with_retries_postanalysis(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)
        #single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)

#run_simulation('/nfs/scistore26/saricgrp/mkrstic/ga_code/stiffness_setup_test', path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/run_vary_stiffness2/deme_0/gen_0/ind_0/seed_22365')