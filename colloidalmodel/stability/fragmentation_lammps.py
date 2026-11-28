import lammps
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import ast
import signal

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

def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds,which_simulation):
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
                elif which_simulation == 'littleguys':
                    if (atom_type1 == 1 and atom_type2 == 2) or (atom_type1 == 2 and atom_type2 == 1) or (atom_type1 == 6 and atom_type2 == 7) or (atom_type1 == 7 and atom_type2 == 6): #make sure to have full binding
                        return True
                elif which_simulation == 'Model0':
                    if (atom_type1 == 4 and atom_type2 == 6) or (atom_type1 == 6 and atom_type2 == 4): #top types interacting
                        return True
    return False

def get_bonds(positions, molecules,types,atom_ids,Lbox,cutoff_distance,which_simulation):
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
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff_distance, box_bounds, which_simulation): #if theyre touching move
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

    # center of mass of the molecule with max index?
    com_x_max_mol = np.mean(max_mol_x)
    # center of mass of the molecule with min index?
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

def graph_new(positions, molecules,types,atom_ids,Lbox,cutoff_distance,num_of_molecules,which_simulation):
    bonds,atoms_data,_ = get_bonds(positions, molecules,types,atom_ids,Lbox,cutoff_distance,which_simulation)
    clusters = find_clusters(range(1,num_of_molecules+1),bonds)
    
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

def setup_simulation_littleguys(lmp, spath, seed, dump_steps, eps, extra_chevron_bond,epsC,epsT):
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary p p p",
        #f"log {spath}/log.txt",
        f"read_data {spath}/restart_config.dat",
        "group moving id > 0",
        "group none empty",
        "group mol1and2 molecule 1 2",
        f"variable seed equal {seed}",
        f"variable dump_steps equal {dump_steps}",
        f"variable eps equal {eps}",
        f"variable eps_C equal {epsC}", # was opposite - 3/6
        f"variable eps_T equal {epsT}",
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
        #"pair_style soft 10.0",
        #"pair_coeff * * 0.0",
        #"run 2000",  # soften initial overlaps
        # then switch back to your real pair_style & coefficients
        "pair_style hybrid/overlay zero 2.00 cosine/squared 2.00",
        "pair_coeff * * cosine/squared 0.00 1.00 1.10",
        "pair_coeff * * zero 2.00",
        "pair_coeff * * cosine/squared 1.00 1.00 1.00 wca",
        #LITTLE GUYS SETUP - extra chevron bond
        #"pair_coeff 2 7 cosine/squared ${eps_T} 1.00 ${bcoff} wca",
        "pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca",
        "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
        "pair_coeff 6 7 cosine/squared ${eps} 1.00 ${bcoff} wca",
        #LITTLE GUYS SETUP - specific binding sites + ([
        #"pair_coeff 3 4 cosine/squared ${eps_C} 1.00 ${bcoff} wca"
    #] + (
    #["pair_coeff 2 7 cosine/squared ${eps_T} 1.00 ${bcoff} wca"]
    #if not extra_chevron_bond else []
    #) + [
    #    "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
    #    "pair_coeff 6 7 cosine/squared ${eps} 1.00 ${bcoff} wca",
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
        #"fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
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
        "boundary p p p",
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
        #"fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        #"fix ywalls all wall/reflect ylo EDGE yhi EDGE",
        #'fix put_polymer_in_middle all recenter 0.0 0.0 0.0',
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

def run_simulation_with_retries(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps,extra_chevron_bond, print_mode,which_simulation,epsC,epsT,max_retries=1):
    attempt = 0
    while attempt < max_retries:
        try:
            # Set timeout (e.g., 5 hours)
            timeout_seconds = 18000
            signal.alarm(timeout_seconds)

            # Run your simulation
            single_seed(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, extra_chevron_bond, print_mode,which_simulation,epsC,epsT)

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

def run(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps, eps,extra_chevron_bond,  which_simulation,epsC,epsT):
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
    if which_simulation == 'GA':
        setup_simulation_GA(lmp, sim_dir, seed, dump_steps, eps)
    elif which_simulation == 'littleguys':
        setup_simulation_littleguys(lmp, sim_dir, seed, dump_steps, eps, extra_chevron_bond,epsC,epsT)
    elif which_simulation == 'Model0':
        lmp.file("in.devtest") #assumes an input script exists with interactions/angles definitions like the ones in the setup functions here

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
    #left_monomers_num, right_monomers_num,dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(step, pos, mol, cutoff, L, types, ids,molecule_number, print_mode)
    counter = 1
    new_len = 13

    # Timeout configuration
    timeout_duration = 1200  # Maximum runtime (e.g., 5 hours = 18000 seconds)
    stall_timeout = 120  # Time limit for detecting stalls (e.g., 10 minutes = 600 seconds)
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
        
        lmp.command('run %d'%(run_interval))
        # Analyse the system
        L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        step = lmp.get_thermo('step')
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")
        types = lmp.numpy.extract_atom("type")
        molecule_number = max(mol)
        clusters,atoms_data,bonds = graph_new(pos, mol, types,ids,L,cutoff, molecule_number,which_simulation)
        if len(clusters)>1: #means there is fragmentation
            with open(sim_dir +"/fragmented_clusters.txt", "w") as file: #write clusters at moment of fragmentation 
                file.write('Fragmented cluster at step ' + str(step) + ' with interaction ' + str(eps) +':\n')
                file.write(str(clusters) + '\n')
            #lmp.command(f"dump dump_end all custom 1 {sim_dir}/output_com_lastframe.xyz id mol type x y")
            #lmp.command('dump_modify dump_end format line "%d %d %d %.2f %.2f " first yes')
            #lmp.command('run 1')
            #lmp.command('quit') #stop further lammps execution
            #break
    
    # Clean up
    lmp.close()

    print(f" LAMMPS simulation completed in {elapsed_time:.2f} seconds.")
    return True

def write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps,epsC,epsT):
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
    info_file.write(f'epsC: {epsC}\n')
    info_file.write(f'epsT: {epsT}\n')
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

def single_seed(spath, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, extra_chevron_bond, print_mode, which_simulation,epsC,epsT):
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

        # Run simulation
        #print()
        #print('Starting simulation...')
        #print()
        #start_time = time.time()
        
        success = run(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode, seed, dump_steps, eps, extra_chevron_bond, which_simulation,epsC,epsT)
    

def run_simulation(path, Lbox=60,  nMonsSol=1, seed=23451, cutoff=1.2, run_interval=10, run_steps=100000000, dump_steps=10000, debug_mode=True,eps=5, print_mode=False, Nseeds=1,extra_chevron_bond=False, which_simulation='littleguys',epsC=3,epsT=7):
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")

    success_of_run = False
    while not success_of_run:
        #seeds = np.random.randint(1, 999999, Nseeds)
        #while len(np.unique(seeds)) < Nseeds:
        #    seeds = np.random.randint(1, 999999, Nseeds)

        #seed = seeds[0]

        # Create the simulation folders
        #if not os.path.exists(path):
        #    os.makedirs(path, exist_ok=True)

        # Write the information file
        write_info_file(path, Lbox, nMonsSol, seed, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps,epsC,epsT)

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

        success_of_run = run_simulation_with_retries(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, extra_chevron_bond,print_mode, which_simulation,epsC,epsT)
        #single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)


def run_simulation_postanalysis(path, path_ind,Lbox=60, max_length_goal=10, nMonsSol=1, seed=23451, cutoff=1.05, run_interval=100, run_steps=100000000, dump_steps=10000, debug_mode=True,eps=50, print_mode=False, Nseeds=1, extra_chevron_bond=False, init_region_L_width = 15, init_region_L_height=10):
    # Define the seeds to run
    #start_time = time.time()
    #print(f"Python started at {time.strftime('%X')}")

    seeds = np.random.randint(1, 999999, Nseeds)
    while len(np.unique(seeds)) < Nseeds:
        seeds = np.random.randint(1, 999999, Nseeds)

    #seed = seeds[0]

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

    run_simulation_with_retries(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, extra_chevron_bond, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)
    #single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal, init_region_L_width,init_region_L_height)

#run_simulation('/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/little_guys_fragmentation/eps_5')