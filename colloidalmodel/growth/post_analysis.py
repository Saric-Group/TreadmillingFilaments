import growing_polymer_increasebox_changeboxsize
import get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from collections import defaultdict, deque
import ast
import glob
import shutil
import signac
import signac_runlammps_project_increasebox_postanalysis
import signac_runlammps_project_increasebox_postanalysis_squarebox
import signac_runlammps_project_increasebox_anglepotential_postanalysis
import signac_runlammps_project_increasebox_postanalysis_rectbox
import signac_runlammps_project_increasebox_anglepotential_postanalysis_rectbox
import signac_runlammps_project_increasebox_postanalysis_rectbox_switch
import signac_runlammps_project_increasebox_postanalysis_rectbox_stiffness_binding
import signac_runlammps_project_increasebox_postanalysis_rectbox_switch_binding
import signac_runlammps_project_increasebox_const_conc_growth_selective_walls

import signac_runlammps_project_increasebox_const_conc_growth


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

def read_individual_txt_file(file_path): #done, fine
    
    #ind_path = os.path.join(file_path,'individual.txt')

    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Reads all lines into a list
        #print(lines)
        
    individual = lines[1]
    #random_number_for_type = lines[3]
    #print(random_number_for_type)
    
    individual = ast.literal_eval(individual)
    #random_number_for_type = float(ast.literal_eval(random_number_for_type))
    
    return individual#, random_number_for_type

def make_signac_folder(sim_path):

    paths = glob.glob(sim_path +"/sd*")
    submit_folder = os.path.join(sim_path,'submission_files')
    project = signac.init_project(submit_folder)
    for p in paths:
        sp = {"path": p}
        job = project.open_job(sp).init()

    return submit_folder

def classify_monomers(bonds, monomer1, monomer2):
    # Step 1: Build the adjacency list (graph representation)
    graph = defaultdict(set)
    for a, b in bonds:
        graph[a].add(b)
        graph[b].add(a)

    # Step 2: Identify branching points (monomers with 3+ bonds)
    branching_points = {node for node, neighbors in graph.items() if len(neighbors) >= 3}

    # Step 3: BFS function that stops at branch points
    def bfs_stop_at_branch(start, exclude):
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                # Stop at branch points
                queue.extend(graph[node] - visited - {exclude} - branching_points)
        return visited

    connected_to_1 = bfs_stop_at_branch(monomer1, monomer2)
    connected_to_2 = bfs_stop_at_branch(monomer2, monomer1)

    return {
        "connected_to_1": connected_to_1,
        "connected_to_2": connected_to_2,
        "branching_points": branching_points,
        "direct_neighbors_1": graph[monomer1],
        "direct_neighbors_2": graph[monomer2]
    }



for k in range(1):
    print(k)
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/volume_exclusion_rectbox_higherbox_longer'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/check_individuals_volume_exclusion'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_postanalysis_newruns/stiffness_and_binding_GA_deme0_gen23_ind46_seed38432_longer'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/run_vary_stiffness_and_binding_types_10_seeds_3_binding_types_elites_1_polarity_and_linearity_rectbox_eps30_100ind/deme_0/gen_23/ind_46/seed_38432'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_inds_from_capper_sims_postanalysis/ind5'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_inds_from_capper_sims/ind5'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/little_guys_trianglestate_full_longer_deletion'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code'#/run_vary_stiffness_rectbox_1_elite_justpolarity_size5_longerlammps_10_seeds_per_ind_check_directionality_monomers_trial2_eval_all/deme_0/gen_6/ind_32/seed_166096'
    #sim_path ='/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_stiffness_and_switch/deme5_gen13_ind66_noswitch'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/run_switch_rectbox_elitism_10_seeds_100ind/deme_5/gen_13/ind_66/seed_145319'
    
    
    sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/littleguys_eps8_detailed_balance_moreoutput_getmolidfromatomid'
    path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/littleguys_triangle_fullinterface_nomiddlemonomer_2/sd12396'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/LG_withextrabond_kineticpolarity'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration/GA_capper_longer'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/check_individuals_volume_exclusion'
    #sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/GA_tail_nomiddlemonomer_2'
    #path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/check_individual_tail_hindering'
    
    # sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/extra_GA_designs_forasymmetry/ind6'
    # path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/extra_GA_designs_forasymmetry/ind6_retractinghand'
    
    # sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/GA_designs_rigid_assembly/capper'
    # path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/GA_capper/sd11983'
    Nseeds = 300
    seed = 189
    angle = False
    rectbox = True
    stiffness = True
    switch = False
    binding = False
    const_conc_growth = True
    selective_walls = False

    
    if Nseeds == 1:
        seeds = [seed]
    else:
        np.random.seed(seed)
        seeds = np.random.randint(10000, 99999, Nseeds)
        while len(np.unique(seeds)) < Nseeds:
            seeds = np.random.randint(10000, 99999, Nseeds)

    # Create the simulation folders
    if not os.path.exists(sim_path):
        os.makedirs(sim_path, exist_ok=True)

    for seed in seeds:
        spath = f'{sim_path}/sd{seed}'
        os.makedirs(spath, exist_ok=True)
        #shutil.copy(os.path.join(path_ind,'individual_make_inside_more_rigid.txt'), os.path.join(spath,'individual.txt'))
        shutil.copy(os.path.join(path_ind,'individual.txt'), spath)
        #shutil.copy(path_ind, spath)


    signac_folder = make_signac_folder(sim_path)
    
    if const_conc_growth == True and selective_walls == False:
        signac_runlammps_project_increasebox_const_conc_growth.run_signac_lammps(signac_folder)
    elif const_conc_growth == True and selective_walls == True:
        signac_runlammps_project_increasebox_const_conc_growth_selective_walls.run_signac_lammps(signac_folder)
    elif const_conc_growth == False: 
        if rectbox == True: 
            if stiffness==True and binding==False and switch==False:
                signac_runlammps_project_increasebox_postanalysis_rectbox.run_signac_lammps(signac_folder)
            elif stiffness== True and switch==True and binding==False:
                signac_runlammps_project_increasebox_postanalysis_rectbox_switch.run_signac_lammps(signac_folder)
            elif stiffness==True and binding==True and switch==False:
                signac_runlammps_project_increasebox_postanalysis_rectbox_stiffness_binding.run_signac_lammps(signac_folder)
            elif switch==True and binding==True:
                signac_runlammps_project_increasebox_postanalysis_rectbox_switch_binding.run_signac_lammps(signac_folder)
        elif rectbox ==False:
            signac_runlammps_project_increasebox_postanalysis_squarebox.run_signac_lammps(signac_folder)

    """
    if rectbox == True: 
        if angle==True and stiffness==True:
            signac_runlammps_project_increasebox_anglepotential_postanalysis_rectbox.run_signac_lammps(signac_folder)
        elif angle==False and stiffness==True:
            signac_runlammps_project_increasebox_postanalysis_rectbox.run_signac_lammps(signac_folder)
        elif switch==True:
            signac_runlammps_project_increasebox_postanalysis_rectbox_switch.run_signac_lammps(signac_folder)

    elif rectbox ==False:
        if angle==True:
            signac_runlammps_project_increasebox_anglepotential_postanalysis.run_signac_lammps(signac_folder)
        else:
            signac_runlammps_project_increasebox_postanalysis.run_signac_lammps(signac_folder)
    """
    print("All jobs completed.")
    
    #run longer simulation
    #commented to not have lammps run again, just do analysis
    #growing_polymer_increasebox_changeboxsize.run_simulation_postanalysis(sim_path, path_ind,Lbox =100, max_length_goal = 30,nMonsSol= 1,seed= 23451,cutoff= 1.05, run_interval=100, run_steps=10000000, dump_steps=10000, debug_mode=True,eps=10, print_mode=False,Nseeds= 5,  init_region_L=30)
    """
    paths = glob.glob(os.path.join(sim_path,'sd*'))
    shift_total = 0
    for path in paths:
        com_dump_file_positions = os.path.join(path, "output_com.xyz")

        bonds,atoms_data,box_bounds = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_bonds(com_dump_file_positions,timestep_index=-1)#doesnt care about number of monomers
        num_of_bonds = len(bonds)
        num_of_molecules = np.max(np.array([tup[1] for tup in atoms_data]))
        #number of subgraphs
        num_of_subgraphs_com = count_subgraphs_com(num_vertices=num_of_molecules, bonds=bonds)
                
        max_shift, cluster_dynamics, clusters_over_time,time_of_cluster_change, total_delta_r, prefactors = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_relative_com_1_particle_postanalysis_new(com_dump_file_positions, dump_freq=10000)#adjusted to any monomer number
        shift_total +=max_shift
        com_over_time_monomer_1 = [row[0] for row in total_delta_r]#for asymmetry 
        com_over_time_monomer_1_signed = []
        for i, com in enumerate(com_over_time_monomer_1):
            com_over_time_monomer_1_signed.append(com*prefactors[i])
        #annealing count and breaking count can just use
        annealing_count, breaking_count = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.get_number_of_annealing_events(cluster_dynamics)#doesnt care about monomer number
        #branching nodes - linearity
        branching_nodes = detect_branching_from_edges(bonds) #doesnt care about monomer number

        final_clusters = clusters_over_time[-1]
        #mol_number = max(max(sublist) for sublist in final_clusters)
        end_monomers, fully_connected, just_end_monomers = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.find_end_monomers(final_clusters, bonds)#doesnt care about monomer number

        pair_distances = []
        for pair in just_end_monomers:
            atoms_data, (xlo, xhi, ylo, yhi) = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.read_lammps_dump(com_dump_file_positions,timestep_index=-1)
            com_mol_1 = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.center_of_mass(atoms_data, target_mol_id=pair[0]) 
            com_mol_2 = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.center_of_mass(atoms_data, target_mol_id=pair[1])
            com_dist = get_com_from_lammps_output_3types_rigidity_newannealing_growingpolymer.com_distances(com_mol_1,com_mol_2)
            pair_distances.append(com_dist)

        #individual, random_number_for_type = read_individual_txt_file(path_ind + '/individual.txt')
        individual = read_individual_txt_file(path_ind + '/individual.txt')

        # Initialize counters
        count_particle_1 = 0
        count_bonds_with_1 = 0

        # Iterate through the data
        for row in individual:
            for cell in row:
                # Count values == 1
                if cell['value'] == 1:
                    count_particle_1 += 1
                
                # Count bonds where any bond value is 1
                count_bonds_with_1 += sum(1 for bond in cell['bonds'].values() if bond == 1)

        #pair_distances = end to end distance - rigidity

        #write the numbers to a txt file:
        with open(os.path.join(path,'output_values.txt'), 'w') as file:
            file.write('Number of subgraphs: \n')
            file.write(str(num_of_subgraphs_com) + '\n')
            file.write('Number of times an annealing event occurs: \n')
            file.write(str(annealing_count) +'\n')
            file.write('Number of times a breaking event occurs: \n')
            file.write(str(breaking_count) +'\n')
            file.write('Number of monomers than experience branching out: \n')
            file.write(str(branching_nodes) + '\n')
            file.write('End to end distances: \n')
            file.write(str(pair_distances) + '\n')
            file.write('Number of particles: \n')
            file.write(str(count_particle_1) + '\n')
            file.write('Number of bonds: \n')
            file.write(str(count_bonds_with_1) + '\n')
            file.write('Cluster dynamics')
            file.write(str(cluster_dynamics) + '\n')
            file.write('Distance of COM plot values: \n')
            file.write(str(com_over_time_monomer_1) + '\n')


        plt.plot(range(len(com_over_time_monomer_1)),com_over_time_monomer_1)
        plt.title('Distance of com of monomer 1 to com of the polymer over time')
        plt.savefig(os.path.join(path,'com_of_monomer1_over_time.pdf'))
        plt.close()

    print("AVERAGE SHIFT FOR ALL SEEDS:")
    print(shift_total/len(paths))
    #write the numbers to a txt file:
    with open(os.path.join(sim_path,'average_shift.txt'), 'w') as file:
        file.write("AVERAGE SHIFT FOR ALL SEEDS: \n")
        file.write(str(shift_total/len(paths)))
    """
