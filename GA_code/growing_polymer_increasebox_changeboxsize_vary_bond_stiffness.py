import lammps
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import write_lammps_more_monomers_3_types_vary_bond_stiffness
import ast
import get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer

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

def graph_new(positions, molecules,types,atom_ids,Lbox,cutoff_distance,num_of_molecules):
    bonds,_,_ = get_bonds(positions, molecules,types,atom_ids,Lbox,cutoff_distance)
    clusters = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.find_clusters(range(1,num_of_molecules+1),bonds)
    return clusters

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

def setup_simulation(lmp, spath, seed, dump_steps, eps, init_region_L):
    high = init_region_L/2
    low = -init_region_L/2
    # Define LAMMPS commands
    commands = [
        "units lj",
        "atom_style molecular",
        "dimension 2",
        "boundary f f p",
       # f"log {spath}/log.txt",
        f"read_data {spath}/configuration.dat",
        f"molecule monomer_template {spath}/monomer.dat",
        "region insertion_region block -5 5 -5 5 EDGE EDGE units box",
        "group moving id > 0",
        "group none empty",
        f"variable seed equal {seed}",
        f"variable dump_steps equal {dump_steps}",
        f"variable eps equal {eps}",
        "variable icoff equal 1.41421356237",
        "variable Kbond1 equal 10.000000",
        "variable Kbond2 equal 50.000000",
        "variable Kbond3 equal 100.000000",
        "variable ecoff equal 1.200000",
        "variable bcoff equal 1.100000",
        "variable rdist equal 1.000000",
        "variable tstep equal 0.010000",
        "special_bonds lj 0.0 1.0 1.0",
        "bond_style harmonic",
        "bond_coeff 1 ${Kbond1} 1.0",
        "bond_coeff 2 ${Kbond2} 1.0",
        "bond_coeff 3 ${Kbond3} 1.0",
        "pair_style hybrid/overlay zero 2.00 cosine/squared 2.00",
        "pair_coeff * * cosine/squared 0.00 1.00 1.10",
        "pair_coeff * * zero 2.00",
        "pair_coeff * * cosine/squared 1.00 1.00 1.00 wca",
        "pair_coeff 1 2 cosine/squared ${eps} 1.00 ${bcoff} wca",
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
        "fix twodim all enforce2d"
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
    clusters = graph_new(pos, mol, types,atom_ids,L,cutoff, num_of_molecules)
    max_length = max([len(sublist) for sublist in clusters])
    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)
    if print_mode:
        print('Step:', step, '- Frame:', step/1000)
        print('Monomers left in left side:', monomers_free_num)
        print()
    return monomers_free_num, max_length

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

def run(sim_dir, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal,seed, dump_steps, eps,init_region_L,L_end):
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
    start_time = time.time()  # Record the start time
    try: 
        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])  # Instantiate LAMMPS eithout output

        #lmp = lammps.lammps()

        # Read file
        #lmp.file('in.setup')
        setup_simulation(lmp, sim_dir, seed, dump_steps, eps,init_region_L)

        # Get box size
        #L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
        current_region_L = init_region_L/2

        # Get positions, molecules and ids
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")

        #find number of monomers
        molecule_number = 2

        # Find the size of the monomer
        monomer_size = len(pos[mol == 1])

        types = lmp.numpy.extract_atom("type")

        # Initial graph
        step = lmp.get_thermo('step')
        monomers_left,max_length = analyse_simulation(step, pos, mol, cutoff, L_end, types, ids,molecule_number, print_mode)
        counter = 1
        #current_mol_number = 10 #random big number, higher than 2 to make the bug of fix deopsit not depoisting at the exact step of polymerization disappear

        # Iterate through loops
        for i in range(loops):
        #for i in tqdm(range(loops)):
            # Run for run_interval steps
            elapsed_time = time.time() - start_time
            if elapsed_time > 18000:#if it takes more than 5h to run, quit
                print(f"Timeout reached after {timeout} seconds at loop {i}. Terminating LAMMPS simulation.")
                lmp.command("quit")  # Gracefully terminate the LAMMPS simulation
                return False  # Indicate that the simulation was terminated due to timeout

            lmp.command('run %d'%(run_interval))
            # Analyse the system
            step = lmp.get_thermo('step')
            pos = lmp.numpy.extract_atom("x")
            mol = lmp.numpy.extract_atom("molecule")
            ids = lmp.numpy.extract_atom("id")
            types = lmp.numpy.extract_atom("type")
            molecule_number = max(mol)
            monomers_left,max_length = analyse_simulation(step, pos, mol, cutoff, L_end, types, ids,molecule_number, print_mode)
            if monomers_left < nMonsSol:
                nmonomers = nMonsSol-monomers_left
                delta_L = ((L_end-init_region_L)/(max_len_goal-2))/2
                current_region_L = init_region_L/2 + delta_L*(molecule_number - 1)
                """
                if counter==1:
                    lmp.command('unfix boundary')
                else:
                    lmp.command('unfix boundary_%d'%(counter-1))
                lmp.command('fix                 boundary_%d all wall/reflect xlo %f xhi %f ylo %f yhi %f \n\n'%(counter,-current_region_L, current_region_L, -current_region_L, current_region_L))
                """
                lmp.command('unfix xwalls')
                lmp.command('unfix ywalls')
                lmp.command('change_box          all x final %f %f y final %f %f\n\n' %(-current_region_L, current_region_L, -current_region_L, current_region_L))
                lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
                lmp.command("fix ywalls all wall/reflect ylo EDGE yhi EDGE")
                lmp.command('fix                 add_mons moving deposit %d 0 1000000000 ${seed} mol monomer_template region insertion_region near 7'%(nmonomers))
                counter +=1


        # Final graph
        step = lmp.get_thermo('step')
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")
        ids = lmp.numpy.extract_atom("id")
        types = lmp.numpy.extract_atom("type")
        #molecule_number = max(mol)
        monomers_left,max_length = analyse_simulation(step, pos, mol, cutoff, L_end, types, ids,molecule_number, print_mode)
        
        # Clean up
        lmp.close()

        print(f" LAMMPS simulation completed in {elapsed_time:.2f} seconds.")
        return True

    except Exception as e:
        print(f"Error during LAMMPS simulation: {e}")
        return False
    

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
    last_timestep = 0
    goal_last_timestep = 2000000

    success = False
    while not success and last_timestep!=goal_last_timestep:
        success = run(spath, nMonsSol, cutoff, loops, run_interval, debug_mode, print_mode,max_len_goal, seed, dump_steps, eps, init_region_L,Lbox)
        last_timestep = get_last_timestep(dump_file='output_com.xyz')
    #end_time = time.time()
    #print()
    #print('Simulation done!')


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

def run_simulation(path, path_ind,Lbox=60, max_length_goal=10, nMonsSol=1, seed=23451, cutoff=1.05, run_interval=100, run_steps=2000000, dump_steps=10000, debug_mode=True,eps=10, print_mode=False, Nseeds=1, init_region_L_width=20, init_region_L_height=20):
    # Define the seeds to run
    if Nseeds == 1:
        seeds = [seed]
    else:
        np.random.seed(seed)
        seeds = np.random.randint(1, 999999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(1, 999999, Nseeds)

    # Create the simulation folders
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Write the information file
    write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps)

    # Define number of loops
    loops = run_steps//run_interval

    # Run the simulations
    single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal,init_region_L_width, init_region_L_height)


def run_simulation_postanalysis(path, path_ind,Lbox =100, max_length_goal = 30,nMonsSol= 1,seed= 23451,cutoff= 1.05, run_interval=100, run_steps=10000000, dump_steps=10000, debug_mode=True,eps=10, print_mode=False,Nseeds= 1,  init_region_L=30):
    # Define the seeds to run
    
    if Nseeds == 1:
        seeds = [seed]
    else:
        np.random.seed(seed)
        seeds = np.random.randint(1, 999999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(1, 999999, Nseeds)
    """
    # Create the simulation folders
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    """
    # Define number of loops
    loops = run_steps//run_interval

    #for seed in seeds:
     #   spath = f'{path}/sd{seed}'
     #   os.makedirs(spath, exist_ok=True)
    # Write the information file
    write_info_file(path, Lbox, nMonsSol, seeds, cutoff, run_interval, run_steps, dump_steps, debug_mode, print_mode, eps)

    # Run the simulations
    single_seed(path, Lbox, nMonsSol, seed, cutoff, loops, run_interval, debug_mode, dump_steps, eps, print_mode, path_ind,max_length_goal,init_region_L)


#run_simulation('/nfs/scistore15/saricgrp/mkrstic/tryout_test', path_ind = '/nfs/scistore15/saricgrp/mkrstic/ga_code/run_vary_stiffness/hof_0')