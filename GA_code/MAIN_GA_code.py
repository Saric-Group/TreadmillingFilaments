import random
from deap import base, creator, tools
from copy import deepcopy
import numpy as np
import write_lammps_more_monomers_3_types_vary_bond_stiffness
from functools import partial
from collections import defaultdict, deque
#import get_com_from_lammps_output_3types
import get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer
import subprocess
import os
import shutil
import multiprocessing
from itertools import count
import shutil
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from lammps import lammps
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import glob
import write_slurm_array_script_in6
import get_deme_folders
from subprocess import Popen
import sys
from mpi4py import MPI
#import signac_runlammps_project_increasebox
import signac_runlammps_project_increasebox_vary_bond_stiffness
#import signac_increasebox_growingpolymer_restartlammps
from multiprocessing import Manager
import growing_polymer_increasebox_changeboxsize_vary_bond_stiffness
import growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox
import pickle

# Path to the main directory
MAIN_DIR = "/nfs/scistore26/saricgrp/mkrstic/ga_code"
RUN_DIR = '/nfs/scistore26/saricgrp/mkrstic/ga_code/run_vary_stiffness_rectbox_1_elite_justpolarity_10_seeds_per_ind_check_directionality_monomers_eval_all_eps_50' 
CXPB = 0.9 
MUTPB=0.1
ngen=50
migrate_interval=5 
num_demes=10
pop_size=50
crosspb=0.5#was 0.1 in last run - with lower crosspb and mutpb
indpb= 0.5#was 0.01 in last run - with lower crosspb and mutpb
tournsize=5 #was 10 in last run - with lower crosspb and mutpb
num_migrate=5 #was 10 in last run - with lower crosspb and mutpb
ELITE_SIZE = 1

import pickle
import os

def clear_directory(directory):
    """
    Deletes all files and subdirectories in the specified directory.
    
    Args:
    - directory (str): The path to the directory to be cleared.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Iterate over all files and directories in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Define the problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, history_index=None, type_rnd_number=-1)  # Use the new FitnessMin class

#Custom initialization function-full grid with all bonds, varying bond stiffness
def custom_init_hexagonal_grid(ind_cls, size):
    ind = ind_cls([])

    # Establish bonds with first neighbors in a hexagonal lattice
    for i in range(size):
        row = []
        for j in range(size):
            point = {"value": 1, "bonds": {}} #this was 1
            row.append(point)
        ind.append(row)

    for i in range(size):
        for j in range(size):
            point = ind[i][j]
            point_tuple = (i,j)
            # Calculate neighbors based on hexagonal lattice
            neighbors = get_hexagonal_neighbors(i,j)

            for neighbor in neighbors:
                    x, y = neighbor
                    if 0 <= x < size and 0 <= y < size:
                        neighbor_point = ind[x][y]    
                        point["bonds"][neighbor] = np.random.choice(2) #make 4 types of bonds? - type 0 is 0 stiffness!
                        neighbor_point["bonds"][point_tuple] = point["bonds"][neighbor]
    
    ind = remove_unbound_particles(ind)#remove particles without at least 1 bond with non-zero stiffness
    
    num, components = count_subgraphs(ind)
    ind = connect_components(ind,components)

    random_number_type = random.random() #initialize ind with defined polarity - up/down or left/right
    ind.type_rnd_number = random_number_type

    return ind


# Function to get hexagonal lattice neighbors
def get_hexagonal_neighbors(i, j):
    neighbors = []
    if i % 2 == 1:
        neighbors.extend([(i-1, j-1), (i-1, j), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j)])
    else:
        neighbors.extend([(i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j), (i+1, j+1)])
    return neighbors

#ensure neighbours have the same bond
def neighbour_bond(i,j,ind):
    neighs = get_hexagonal_neighbors(i, j)
    for ni,nj in neighs:
        if 0 <= ni < len(ind) and 0 <= nj < len(ind[0]):
            ind[ni][nj]["bonds"][(i, j)] = ind[i][j]["bonds"][(ni, nj)]
    return ind

def remove_unbound_particles(ind):#remove particle if it doenst have at least 1 stiff bond
    for i1 in range(len(ind)):
        for i2 in range(len(ind[i1])):
            non_zero_bonds = 0
            for bond_tuple, bond_value in ind[i1][i2]['bonds'].items():
                non_zero_bonds+=bond_value

            if non_zero_bonds==0:
                ind[i1][i2]['value']=0 #remove particles that have no non stiff bonds
            elif non_zero_bonds!=0:
                ind[i1][i2]['value']=1 # add particles that maybe didnt have a stiff bond before but got it now

    return ind

# Custom mutation function - should randomly delete particles
def custom_mutate(individual, indpb, number_of_diff_bond_stiffnesses=2):
    
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            # Mutate bonds 
            if random.random() < indpb: 
                neigh = get_hexagonal_neighbors(i, j)
                random_index = random.randint(0, len(neigh)-1) #choose one neighbour
                ni,nj = neigh[random_index] #chosen neighbour
                if 0 <= ni < len(individual) and 0 <= nj < len(individual[0]):
                    individual[i][j]["bonds"][(ni, nj)] = np.random.choice(number_of_diff_bond_stiffnesses)  # Set bond to 1 or 0 randomly
                    individual[ni][nj]['bonds'][(i,j)] = individual[i][j]["bonds"][(ni, nj)]
    
    individual = remove_unbound_particles(individual)#remove particles without at least 1 bond with non-zero stiffness

    num, components = count_subgraphs(individual)
    individual = connect_components(individual,components)

    return individual,

# Custom crossover function
def custom_crossover(ind1, ind2,crosspb):

    # Perform crossover on bonds and values
    for i in range(len(ind1)):
        for j in range(len(ind1[i])):
            # Swap 0 and 1 values
            if random.random() < crosspb:
                ind1[i][j]["bonds"], ind2[i][j]["bonds"] = ind2[i][j]["bonds"], ind1[i][j]["bonds"]
                ind1 = neighbour_bond(i,j,ind1)#make sure bonds are changed pairwise
                ind2 = neighbour_bond(i,j,ind2)

    ind1 = remove_unbound_particles(ind1)
    ind2 = remove_unbound_particles(ind2)

    num1, components1 = count_subgraphs(ind1)
    ind1 = connect_components(ind1,components1)

    num2, components2 = count_subgraphs(ind2)
    ind2 = connect_components(ind2,components2)
        
    return ind1, ind2

def get_neighbors(point, matrix):
    i, j = point
    m, n = len(matrix), len(matrix[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if i % 2 == 1:
        directions.extend([(1, -1), (-1, -1)])
    else:
        directions.extend([(1, 1), (-1, 1)])
    neighbors = []
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            if (ni, nj) in matrix[i][j]['bonds']:
                if matrix[i][j]['bonds'][(ni, nj)] == 1:
                    neighbors.append((ni, nj))
    #print(f"Neighbors of {point}: {neighbors}")  # Debug statement
    return neighbors

def is_in_grid(q, r, grid):
    """Check if the (q, r) coordinate is within grid bounds."""
    is_it = False
    if 0<= q <len(grid) and 0<= r <len(grid[0]):
        is_it = True
    return  is_it # Dictionary-based check for grid bounds

def triangular_lattice_distance(x1,y1,x2,y2):
    """Calculate the distance between two points in a triangular lattice."""
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def get_position_in_xy(i,j): #transform positions on grid into positions in the xy plane
    x = j
    y = i*np.sqrt(3)/2 #squish the y axis so that its sqrt(3):1
    if i%2 == 1: #hexagonal lattice - shift in every even row of the grid to the left fro half a grid lengthscale size 
        x -= 1/2 #shift x to the left to get hexagonal lattice

    return x,y

def find_closest_pair(component1, component2, grid):
    """Find the closest boundary points between two components."""
    boundary1 = component1
    boundary2 = component2 
    
    if not boundary1 or not boundary2:
        raise ValueError("No valid boundary points found in one or both components.")
    
    min_distance = float('inf')
    closest_pair = None
    
    for p1 in boundary1:
        xp1,yp1 = get_position_in_xy(p1[0],p1[1])
        for p2 in boundary2:
            xp2,yp2 = get_position_in_xy(p2[0],p2[1])
            distance = triangular_lattice_distance(xp1,yp1,xp2,yp2)
            if distance < min_distance:
                min_distance = distance
                closest_pair = (p1, p2)
    
    if closest_pair is None:
        raise ValueError("Could not find a valid closest pair.")
    
    return closest_pair, min_distance

def add_points_along_path(p1, p2, grid):
    """Add points between two positions to create a path in a triangular lattice."""
    q1, r1 = p1
    q2, r2 = p2

    # Move step by step from p1 to p2 using axial coordinates
    while (q1, r1) != (q2, r2):
        # Determine direction to move in (try q-axis first, then r-axis)
        if q1 < q2:
            nq, nr = q1 + 1, r1
        elif q1 > q2:
            nq, nr = q1 - 1, r1
        elif r1 < r2:
            nq, nr = q1, r1 + 1
        elif r1 > r2:
            nq, nr = q1, r1 - 1

        # Add the point if it's empty
        if (nq, nr) not in grid:
            grid[nq][nr]['value'] = 1

        # Update bonds between current and next point
        grid[q1][r1]['bonds'][(nq, nr)] = 1
        grid[nq][nr]['bonds'][(q1, r1)] = 1

        # Move to the next point
        q1, r1 = nq, nr
        
    return grid

def connect_components(grid, components):
    """Connect all components by adding bonds or new particles based on distance in a triangular lattice."""
    while len(components) > 1:
        # Take two components
        component1 = components[0]
        component2 = components[1]
        
        # Find the closest pair of points between them and their distance
        (p1, p2), min_distance = find_closest_pair(component1, component2, grid)
        
        # If the distance is 1 (first neighbors), just add a bond
        if min_distance < 1.1:
            grid[p1[0]][p1[1]]['bonds'][p2] = 1
            grid[p2[0]][p2[1]]['bonds'][p1] = 1
        else:
            # Otherwise, add new points along the path to connect them
            grid = add_points_along_path(p1, p2, grid)
        
        # Get new components
        num,components = count_subgraphs(grid)
        
    return grid

def dfs2(matrix, node, visited):
        stack = [node]
        cluster = [node]
        
        while stack:
            current = stack.pop()
            x, y = current
            for neighbor, bond in matrix[x][y]['bonds'].items():
                nx, ny = neighbor
                if bond == 1 and (nx, ny) not in visited and matrix[nx][ny]['value'] == 1:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    cluster.append((nx,ny))
                    
        return cluster

def count_subgraphs(matrix):

    m, n = len(matrix), len(matrix[0])
    visited = set()
    subgraph_count = 0
    clusters = []
    
    for i in range(m):
        for j in range(n):
            node = (i, j)
            if node not in visited and matrix[i][j]['value'] == 1:
                visited.add(node)
                cluster = dfs2(matrix, node, visited)
                subgraph_count += 1
                clusters.append(cluster)
                
    
    return subgraph_count, clusters

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

def build_adjacency_list(edges):
    """Builds an adjacency list from a list of edges."""
    adjacency_list = defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)  # Since it's an undirected graph
    return adjacency_list

def detect_branching_from_edges(edges):
    """Detects if the graph is branched using a list of edges."""
    adjacency_list = build_adjacency_list(edges)
    #print(adjacency_list)
    
    # Calculate the degree of each node
    degrees = {node: len(neighbors) for node, neighbors in adjacency_list.items()}
    
    # Detect branching: If any node has a degree greater than 2, it's branched
    branched_nodes = [node for node, degree in degrees.items() if degree > 2]
    
    return branched_nodes


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", custom_init_hexagonal_grid, creator.Individual, size=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def sum_all_bonds(matrix):
    total_bonds = 0
    
    for row in matrix:
        for element in row:
            total_bonds += sum(element['bonds'].values())
    
    return total_bonds


def eval_lattice(individual,path):#write out a configuration.dat for the given choice - maybe make them all the same type? expect the middle ones, then run lammps
    value_sum = sum(point["value"] for row in individual for point in row)#box size was 50
    #deme, generation, individual_id = individual.gen_id  # Assuming individual contains deme,generation,ID,seed
    #sim_dir = os.path.join(RUN_DIR, f"deme_{deme}",f"gen_{generation}", f"ind_{individual_id}",f"seed_{seed_num}")
    sim_dir = path

    num_vertices = 10
    linear_chain_bonds = num_vertices-1
    num_subgraphs, _ = count_subgraphs(individual)
    
    #if num_subgraphs ==1 and value_sum>1:
    #subprocess.run(lammps_command)
    #run_lammps_2('in6.local')
    # Define the path to the output file
    com_dump_file_positions = os.path.join(sim_dir, "output_com.xyz")
    
    os.chdir(MAIN_DIR)
    
    bonds,atoms_data,box_bounds = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_bonds(com_dump_file_positions,timestep_index=-1)
    num_of_bonds = len(bonds)
    
    num_of_subgraphs_com = count_subgraphs_com(num_vertices=num_vertices, bonds=bonds)
            
    max_shift, cluster_dynamics, clusters_over_time = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_relative_com_1_particle(com_dump_file_positions, dump_freq=10000)
    annealing_count, breaking_count = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_number_of_annealing_events(cluster_dynamics)
    branching_nodes = detect_branching_from_edges(bonds)
    
    annealing_penalty = 1000*annealing_count
    breaking_penalty = 1000*breaking_count

    final_clusters = clusters_over_time[-1]
    mol_number = max(max(sublist) for sublist in final_clusters)
    end_monomers, fully_connected, just_end_monomers = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.find_end_monomers(final_clusters, bonds)

    pair_distances = []
    for pair in just_end_monomers:
        atoms_data, (xlo, xhi, ylo, yhi) = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.read_lammps_dump(com_dump_file_positions,timestep_index=-1)
        com_mol_1 = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.center_of_mass(atoms_data, target_mol_id=pair[0]) 
        com_mol_2 = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.center_of_mass(atoms_data, target_mol_id=pair[1])
        com_dist = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.com_distances(com_mol_1,com_mol_2)
        pair_distances.append(com_dist)

    penalty_subgraphs_com = 1000 
    #penalty_linear_filament = 1000
    asymmetry_penalty = 1000
    penalty_complexity = 1
    penalty_linearity = 10**2 * len(branching_nodes)

    #if max_shift> 0.001:
    asymmetry_term = asymmetry_penalty*np.exp(-max_shift/10)
    #else:
    #    asymmetry_term = 1000000# big penalty if no shift
        
    if len(pair_distances) == 0:
        penalty_end_to_end_distance = 10**5
    else:
        penalty_end_to_end_distance = 10000/sum(pair_distances)

    penalty_nonbinding = 10**3*np.abs((10-mol_number))

    #fitness = asymmetry_term + penalty_subgraphs_com*np.abs((num_of_subgraphs_com-1))
    individual.term1 += penalty_complexity*value_sum
    individual.term2 += penalty_linearity
    individual.term3 += asymmetry_term
    individual.term4 += annealing_penalty
    individual.term5 += breaking_penalty
    individual.term6 += penalty_end_to_end_distance
    individual.term7 += penalty_nonbinding
    
    #fitness = penalty_complexity*value_sum + penalty_linearity + asymmetry_term + annealing_penalty + penalty_subgraphs_com*np.abs((num_of_subgraphs_com-1)) + penalty_end_to_end_distance
    #fitness = penalty_complexity*value_sum + penalty_linearity + asymmetry_term+ annealing_penalty + breaking_penalty + penalty_end_to_end_distance + penalty_nonbinding
    #fitness = penalty_linearity + asymmetry_term
    fitness = asymmetry_term

    print('One of the seeds FITNESS FOR IND IN WORK DIR' + str(sim_dir) +' IS:' + str(fitness), flush=True)

    return fitness

def eval_lattice_all_seeds(individual):
    main_sim_dir = individual.work_dir #main work directory
    #avg_max_shift = 0
    fitness = 0
    all_folders = glob.glob(os.path.join(main_sim_dir,'seed_*')) #find paths to all the seeds
    #print('FOUND FOLDERS:')
    #print(all_folders)
    #print('FOR PATH' + str(main_sim_dir))
    for path in all_folders:
        #avg_max_shift += eval_lattice(individual,path)
        fitness += eval_lattice(individual,path)
        #print('FITNESS IS:')
        #print(fitness)
    #avg_max_shift = avg_max_shift/len(all_folders)
    #fitness = 1000*np.exp(-avg_max_shift/10)
    fitness = fitness/len(all_folders)
    print('FINAL FITNESS FOR IND IN WORK DIR' + str(main_sim_dir) +' IS:' + str(fitness), flush=True)
    
    return fitness,

# Parallel evaluation function
def parallel_evaluate(population, max_workers=5):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(toolbox.evaluate, ind) for ind in population]
        fitnesses = [f.result() for f in concurrent.futures.as_completed(futures)]
    return fitnesses


def save_generation_to_file(filename, population, generation):
    with open(filename, 'a') as f:  # Open the file in append mode to append the content
        f.write(f"Generation {generation}:\n")
        for ind in population:
            f.write("Matrix of values:\n")
            for i, row in enumerate(ind):
                for j, point in enumerate(row):
                    f.write(f"{point['value']} ")
                f.write("\n")
            f.write("\n")

            f.write("Bonds:\n")
            written_bonds = set()  # To keep track of written bonds
            for i, row in enumerate(ind):
                for j, point in enumerate(row):
                    if point['value'] == 1:
                        for bond, value in point['bonds'].items():
                            # Sort the bond tuple to avoid duplicates
                            sorted_bond = tuple(sorted([(i, j), bond]))
                            if sorted_bond not in written_bonds:
                                f.write(f"({i}, {j}) <-> ({bond[0]}, {bond[1]}): {value}\n")
                                written_bonds.add(sorted_bond)
            f.write("\n")

        # Get the best individual from the population
        best_individual = tools.selBest(population, 1)[0]
        f.write("Best individual:\n")
        for i, row in enumerate(best_individual):
            for j, point in enumerate(row):
                f.write(f"{point['value']} ")
            f.write("\n")
        f.write("\n")

toolbox.register("evaluate", eval_lattice_all_seeds)
toolbox.register("mate", custom_crossover, crosspb=crosspb )  # No crossover on bonds
toolbox.register("mutate", custom_mutate, indpb=indpb)  # Using custom mutation function
toolbox.register("select", tools.selTournament, tournsize=tournsize)

# Define the function to initialize the population
def initialize_population(population, deme_index, generation_number):
    for ind_id, ind in enumerate(population):
        ind.gen_id = (deme_index, generation_number, ind_id)
        ind.work_dir = f"{RUN_DIR}/deme_{deme_index}/gen_{generation_number}/ind_{ind_id}"
        ind.history_index = None  # Initialize history_index if needed
        ind.term1 = 0
        ind.term2 = 0
        ind.term3 = 0
        ind.term4 = 0
        ind.term5 = 0
        ind.term6 = 0
        ind.term7 = 0
        print(f'Initializing individual {ind_id} in deme {deme_index} with work directory {ind.work_dir} ' )
     
# Define migration scheme
def migrate(populations, k, selection, replacement):
    migrants = []
    for pop in populations:
        selected = selection(pop, k)
        migrants.append(selected)
    
    for i in range(len(populations)):
        source_pop = populations[i]
        target_pop = populations[(i + 1) % len(populations)]
        replace_individuals = replacement(target_pop, k)
        
        for j in range(k):
            target_pop[target_pop.index(replace_individuals[j])] = migrants[i][j]

toolbox.register("migrate", migrate, k = num_migrate, selection=tools.selBest, replacement=tools.selWorst)
# Register History
history = tools.History()
#toolbox.decorate("mate", history.decorator)
#toolbox.decorate("mutate", history.decorator)
#toolbox.decorate("select", history.decorator)
"""
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
"""
# Register statistics for each objective (f1 to f6)
stats_f1 = tools.Statistics(key=lambda ind: ind.term1)
stats_f2 = tools.Statistics(key=lambda ind: ind.term2)
stats_f3 = tools.Statistics(key=lambda ind: ind.term3)
stats_f4 = tools.Statistics(key=lambda ind: ind.term4)
stats_f5 = tools.Statistics(key=lambda ind: ind.term5)
stats_f6 = tools.Statistics(key=lambda ind: ind.term6)
stats_f7 = tools.Statistics(key=lambda ind: ind.term7)

# Combine the statistics for all objectives into a MultiStatistics object
mstats = tools.MultiStatistics(f1=stats_f1, f2=stats_f2, f3=stats_f3, f4=stats_f4, f5=stats_f5, f6=stats_f6, f7 = stats_f7)

# Register the required statistics (min, max, avg, std) for each objective

mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

def check_job_completion(job_id):
    try:
        result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True, check=True)
        # Check if the job ID exists in the output, indicating it is still in the queue or running
        if job_id in result.stdout:
            return False
        else:
            return True  # Job not found in squeue output means it has completed
    except subprocess.CalledProcessError:
        # Handle subprocess errors if squeue command fails
        return False

# Function to run each batch of processes
def run_batch(batch_size, start_index,file_path):
    os.environ['PYTHONUNBUFFERED'] = "1"  # Disable Python output buffering
    mpirun_command = ["mpirun", "-np", str(batch_size), "python",os.path.join(MAIN_DIR,"lammps_run_growingpolymer.py"),file_path, str(batch_size), str(start_index)]
    #print(f"Running command: {' '.join(mpirun_command)}")  # Print the command to be executed

    process = Popen(mpirun_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #time.sleep(10)
    # Capture the output and error streams
    stdout, stderr = process.communicate()

    # Print out the output from the command
    #print(stdout.decode())
    #print(stderr.decode())

    process.wait()  # Wait for the process to finish
    return process

def evaluate_deme(deme, deme_index, gen, CXPB, MUTPB, hof,full,best_individuals_per_gen,worst_individuals_per_gen,logbook,restart):
    print(f'Evaluating DEME {deme_index}')
    if restart==False:
        if full==True:
            
            # Apply genetic operators
            #print("LENTGH")
            #print(len(deme))
            #print(type(len(deme)))
            #elites = tools.selBest(deme, ELITE_SIZE)
            elites = list(map(toolbox.clone, tools.selBest(deme, ELITE_SIZE)))

            offspring = toolbox.select(deme, len(deme)-ELITE_SIZE)
            #offspring = toolbox.select(deme, len(deme))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        elif full == False:
            offspring = deme

        # Parallel evaluation using multiprocessing
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = offspring
        initialize_population(invalid_ind, deme_index, gen+1)
        job_ids = []
        submit_dir = os.path.join(RUN_DIR,'submission_files')
        os.makedirs(submit_dir,exist_ok = True)
        os.chdir(submit_dir)
        for ind_index, ind in enumerate(invalid_ind):
            deme_ind, generation_ind, individual_id = ind.gen_id #for every individual 10 seeds
            main_sim_dir = os.path.join(RUN_DIR, f"deme_{deme_ind}",f"gen_{generation_ind}", f"ind_{individual_id}")
            os.makedirs(main_sim_dir,exist_ok = True)
            #clear_directory(os.path.join(RUN_DIR, f"deme_{deme_ind}",f"gen_{generation_ind}")) #clear out directory from previous results
            
            seeds = np.random.choice(1000000, size=10, replace=False)
            random_number_type = ind.type_rnd_number 
            for seed_num in seeds:
                sim_dir = os.path.join(main_sim_dir,f"seed_{seed_num}")
                
                # Create the directory if it doesn't exist
                os.makedirs(sim_dir, exist_ok=True)

                write_lammps_more_monomers_3_types_vary_bond_stiffness.write_individual_txt_file(gpath=sim_dir, individual=ind,random_num = random_number_type)
                #print("SIDE CHOSEN" + str(ind.type_rnd_number))

        signac_folder = get_deme_folders.make_signac_folder(gen,deme_index,RUN_DIR)#writes a paths text file for all lammps runs
                    
        # RUN ALL LAMMPS FILES 
        signac_runlammps_project_increasebox_vary_bond_stiffness.run_signac_lammps(signac_folder)

        print("All jobs completed.")

    elif restart==True:
        # Parallel evaluation using multiprocessing
        offspring = deme
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = offspring

    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    """
    # Analyze the output files to get fitness values
    for ind_index, ind in enumerate(invalid_ind):
        fitness = 0
        all_folders = glob.glob(os.path.join(ind.work_dir,'seed_*')) #find paths to all the seeds
        print('FOUND FOLDERS:')
        print(all_folders)
        print('FOR PATH' + str(ind.work_dir))
        for path in all_folders:
            fitness += eval_lattice(ind,path)
            print('FITNESS IS:')
            print(fitness)
        print('FINAL FITNESS IS:')
        fitness/=len(all_folders)
        ind.fitness.values = fitness,
    """
    # Clear the directory
    #clear_directory(RUN_DIR) #deactivate for saving all

    # Update deme with offspring
    
    if full==True and restart==False:
        deme[:] = offspring + elites
    else:
        deme[:] = offspring
    """
    deme[:] = offspring 
    """
    # Find and store the best individual in this generation
    best_individual = tools.selBest(deme, 1)[0]
    best_objectives = best_individual.fitness.values
    best_sim_dir = best_individual.work_dir  # Access work_dir if it exists

    # Find and store the worst individual in this generation
    worst_individual = tools.selWorst(deme, 1)[0]
    worst_objectives = worst_individual.fitness.values
    worst_sim_dir = worst_individual.work_dir  # Access work_dir if it exists

    # Append generation data
    best_individuals_per_gen.append((gen, deme_index ,best_individual, best_objectives, best_sim_dir))
    worst_individuals_per_gen.append((gen, deme_index ,worst_individual, worst_objectives, worst_sim_dir))

    #print('DEME IS:')
    #print(deme)

    # Update history and hall of fame if needed
    hof.update(deme)
    #print('HALL OF FAME')
    #print(hof)
    
    record = mstats.compile(deme)
    record["Generation"] = gen
    record["DemeIndex"] = deme_index
    print("INDIVIDUAL RECORD")
    print(record) 
    logbook.record(**record)
    print('LOGBOOK CURRENT')
    print(logbook)
    """
    # Compile the statistics for this generation
    record = stats.compile(deme)

    # Print the record to verify what is being tracked
    print(record)

    # Record the generation's statistics in the logbook
    logbook.record(gen=gen, evals=deme_index, **record)

    # Optional: Print the logbook stream to see the output
    print(logbook)
    """
    #print best individuals
    #print('BEST INDIVIDUALS')
    #print(best_individuals_per_gen)
    
    print('EVALUATION OF DEME AND UPDATE OF LOGBOOK AND HALL OF FAME SUCCESSFULLY COMPLETED!')

def check_mean_fitness_stagnation(logbook, tolerance_generations, epsilon):
    mean_stagnation_count = 0

    # Loop through the logbook starting from the second generation
    for gen in range(len(logbook)):
        # Get current and previous generation's mean fitness
        prev_gen_mean = logbook[gen - 1]['avg']
        curr_gen_mean = logbook[gen]['avg']

        # Calculate the improvement in mean fitness
        mean_fitness_improvement = abs(curr_gen_mean - prev_gen_mean)

        # Check if the improvement is smaller than the threshold (epsilon)
        if mean_fitness_improvement < epsilon:
            mean_stagnation_count += 1
        else:
            mean_stagnation_count = 0  # Reset if there's significant improvement

        # If mean fitness has stagnated for tolerance_generations, stop
        if mean_stagnation_count >= tolerance_generations:
            print(f"Algorithm stopped at generation {gen} due to mean fitness saturation.")
            return True

    return False  # Continue evolving if no stopping criterion is met


# Define the genetic algorithm
def main():

    # Ensure the main run directory exists
    os.makedirs(RUN_DIR, exist_ok=True)

    (populations, hof, logbook, start_gen, 
    best_individuals_per_gen, worst_individuals_per_gen) = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.load_deme_checkpoint(filename = RUN_DIR + "/checkpoint_demes.pkl")

    if populations is None:
        # fresh run
        clear_directory(RUN_DIR)
        best_individuals_per_gen = []
        worst_individuals_per_gen = []
        logbook = tools.Logbook()
        logbook.header = ['Generation', 'DemeIndex'] + mstats.fields
        hof = tools.HallOfFame(maxsize=20)
        populations = [toolbox.population(n=pop_size) for _ in range(num_demes)]

        # Initial evaluation...
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_demes) as executor:
            futures = [executor.submit(evaluate_deme, deme, deme_index, -1, CXPB, MUTPB,
                                    hof, False, best_individuals_per_gen, worst_individuals_per_gen, logbook, False)
                    for deme_index, deme in enumerate(populations)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
        start_gen = 0
        restart_loaded = False
    else:
        print(f"Resuming from generation {start_gen} with {len(best_individuals_per_gen)} best individuals logged.")
        restart_loaded =True



    # Main evolution loop
    for gen in range(start_gen,ngen):
        
        print('Evaluating GEN ' + str(gen))

        if gen==start_gen and restart_loaded:
            restart=True
        else:
            restart=False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_demes) as executor:
            futures = [executor.submit(evaluate_deme, deme, deme_index, gen,CXPB, MUTPB,hof,True,best_individuals_per_gen,worst_individuals_per_gen,logbook, restart) for deme_index, deme in enumerate(populations)]
            for future in concurrent.futures.as_completed(futures):
                #try:
                result = future.result()  # Ensure all demes are evaluated
                #except Exception as e:
                #   print(f"Evaluation of deme raised an exception: {e}")

        # Apply migration at specified intervals
        if gen % migrate_interval == 0:
            toolbox.migrate(populations)
        """
        if check_mean_fitness_stagnation(logbook, tolerance_generations=5, epsilon=10):
            print(f"Stopped after {gen} generations due to mean fitness saturation.")
            break
        """

        # Save checkpoint after each generation
        get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.save_deme_checkpoint(populations,hof,logbook,gen,best_individuals_per_gen,worst_individuals_per_gen, filename = RUN_DIR + "/checkpoint_demes.pkl")
    
    with open(os.path.join(RUN_DIR,"logbook.txt"), "w") as f:
        f.write(str(logbook))
        
    # Write all the best individuals to a file at once
    with open(os.path.join(RUN_DIR,"best_individuals.txt"), "w") as f:
        f.write("Generation, Best Individual, Fitness, sim_dir\n")
        for gen_i, deme_i, individual_v, fitness_v, sim_dir_i in best_individuals_per_gen:
            f.write(f"{gen_i}, {deme_i},{individual_v}, {fitness_v}, {sim_dir_i}\n")

    # Write all the worst individuals to a file at once
    with open(os.path.join(RUN_DIR,"worst_individuals.txt"), "w") as f:
        f.write("Generation, Worst Individual, Fitness, sim_dir\n")
        for gen_i2, deme_i2, individual_v2, fitness_v2, sim_dir_i2 in worst_individuals_per_gen:
            f.write(f"{gen_i2}, {deme_i2},{individual_v2}, {fitness_v2}, {sim_dir_i2}\n")
    
    # Gather the best individual from each deme
    best_individuals = [tools.selBest(pop, 1)[0] for pop in populations]
    
    for l,ind in enumerate(best_individuals):
        #print(f"Best Individual - Fitness: {ind.fitness.values}")
        #print(f"Best Individual - Details: {ind}")

        # Create a 'best' directory to save the best individual's files
        best_dir = os.path.join(RUN_DIR, "best_deme_" + str(l))
        os.makedirs(best_dir, exist_ok=True)
        source_folders = ind.work_dir
        all_seeds = glob.glob(os.path.join(source_folders, 'seed_*'))
        first_seed = all_seeds[0]
        source_file = os.path.join(first_seed, 'individual.txt')

        shutil.copy(source_file, best_dir)
    
        # Command to run LAMMPS with input script
        growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox.run_simulation(best_dir,best_dir)
        #command = 'python /nfs/scistore15/saricgrp/mkrstic/ga_code/growing_polymer.py --path %s --path_ind %s'%(best_dir,best_dir)

        # Run the command
        #subprocess.run(command)
            
        with open(os.path.join(best_dir,'info.txt'), 'w') as file:
            file.write('BEST INDIVIDUAL: \n')
            file.write(str(ind))
            file.write('\n')
            file.write('FITNESS: \n')
            file.write(str(ind.fitness.values))

    #for hall of fame make lammps files
    hof_fitness = 0
    for m,ind in enumerate(hof):
        #print(f"Best Individual - Fitness: {ind.fitness.values}")
        #print(f"Best Individual - Details: {ind}")
        hof_fitness += ind.fitness.values[0]

        # Create a 'best' directory to save the best individual's files
        best_dir = os.path.join(RUN_DIR, "hof_" + str(m))
        os.makedirs(best_dir, exist_ok=True)
        source_folders = ind.work_dir
        all_seeds = glob.glob(os.path.join(source_folders, 'seed_*'))
        first_seed = all_seeds[0]
        source_file = os.path.join(first_seed, 'individual.txt')

        shutil.copy(source_file, best_dir)
    
        # Command to run LAMMPS with input script
        growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox.run_simulation(best_dir,best_dir)
        #command = 'python /nfs/scistore15/saricgrp/mkrstic/ga_code/growing_polymer.py --path %s --path_ind %s'%(best_dir,best_dir)

        # Run the command
        #subprocess.run(command)

        with open(os.path.join(best_dir,'info.txt'), 'w') as file:
            file.write('BEST INDIVIDUAL: \n')
            file.write(str(ind))
            file.write('\n')
            file.write('FITNESS: \n')
            file.write(str(ind.fitness.values))

    print(logbook)
            
    return hof_fitness

hof_fitness = main()

