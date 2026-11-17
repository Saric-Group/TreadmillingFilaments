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



sim_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/littleguys_eps8_detailed_balance_moreoutput_getmolidfromatomid'
path_ind = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/littleguys_triangle_fullinterface_nomiddlemonomer_2/sd12396'

Nseeds = 300
seed = 189


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
    shutil.copy(os.path.join(path_ind,'individual.txt'), spath)


signac_folder = make_signac_folder(sim_path)
signac_runlammps_project_increasebox_const_conc_growth.run_signac_lammps(signac_folder)
