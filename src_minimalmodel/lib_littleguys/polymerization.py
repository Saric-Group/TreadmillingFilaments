import lammps, sys 
import numpy as np
from collections import defaultdict, deque

def are_molecules_touching(molecule1, molecule2, cutoff, box_bounds):
    xlo, xhi, ylo, yhi = box_bounds
    for atom1 in molecule1:
        for atom2 in molecule2:
            x1, y1, atom_type1 = atom1[1], atom1[2], atom1[3]
            x2, y2, atom_type2 = atom2[1], atom2[2], atom2[3]
            
            dx = x1-x2
            dy = y1-y2

            # removed because we do not have PBC
            #dx = min(abs(x1 - x2), abs(x1 - x2 + (xhi - xlo)), abs(x1 - x2 - (xhi - xlo)))
            #dy = min(abs(y1 - y2), abs(y1 - y2 + (yhi - ylo)), abs(y1 - y2 - (yhi - ylo)))
            
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= cutoff:
                if (atom_type1 == 4 and atom_type2 == 6) or (atom_type1 == 6 and atom_type2 == 4): #top types interacting
                    return True
    return False

def get_bonds(cutoff, lmp):

    #step  = lmp.get_thermo('step')
    positions   = lmp.numpy.extract_atom("x")
    molecules   = lmp.numpy.extract_atom("molecule")
    atom_ids    = lmp.numpy.extract_atom("id")
    types       = lmp.numpy.extract_atom("type")
    dimensions  = get_box_properties(lmp)

    box_bounds = (dimensions['xlo'],dimensions['xhi'],dimensions['ylo'],dimensions['yhi'])
    atoms_data = np.column_stack((atom_ids, molecules, types))
    atoms_data = np.column_stack((atoms_data, positions))
    
    # find the molecules in the system
    molecules = {}
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y, z = atom
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append((atom_id, x, y,atom_type))
    #print(f"Current molecules: {molecules}")

    # Find touching molecules and add bonds
    connected_molecules = []
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff, box_bounds): #if theyre touching move
                    connected_molecules.append((mol_id1, mol_id2))
                    
    #print(f'MONOMERS CONNECTED: {connected_molecules}')
    return connected_molecules, atoms_data, len(molecules.keys())

def get_most_left_and_right_com(pos,mol,clusters):

    """
    Gets positions of most left and right monomers in polymer 
    """

    # if there is no polymer (very beginning of the simulation)
    is_there_polymer = False 
    for subgroup in clusters:
        if len(subgroup)>1:
            is_there_polymer = True
    
    if is_there_polymer:
        # create mask to track only molecules in the polymer - not free monomers
        mask = np.array([any((x in subgroup) and (len(subgroup) > 1) for subgroup in clusters)for x in mol])
        pos = pos[mask]
        mol = mol[mask]

        max_x_index = np.argmax(pos[:, 0])  # Index of atom with max x
        min_x_index = np.argmin(pos[:, 0])  # Index of atom with min x
        max_mol_id = mol[max_x_index]       # molecule to which this atom belongs to for max x
        min_mol_id = mol[min_x_index]       # molecule to which this atom belongs to for min x

    else:
        
        # if there is no polymer formed yet,
        # we return the ID of the first molecule (the seed)
        max_mol_id = 1
        min_mol_id = 1
    
    # Boolean masks for atoms in the selected molecule
    max_mol_mask = (mol == max_mol_id)
    min_mol_mask = (mol == min_mol_id)

    # extract x-positions and masses of the atoms in that molecule
    max_mol_x = pos[max_mol_mask, 0]  
    min_mol_x = pos[min_mol_mask, 0]

    # center of mass of the molecule with max index
    com_x_max_mol = np.mean(max_mol_x)
    # center of mass of the molecule with min index
    com_x_min_mol = np.mean(min_mol_x)

    return com_x_max_mol, com_x_min_mol

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

    #print(f"CLUSTERS: {clusters}")
    return clusters

def graph_new(cutoff, lmp):
    
    bonds, atoms_data, num_of_molecules = get_bonds(cutoff, lmp)
    clusters = find_clusters(range(1,num_of_molecules+1),bonds)
    return clusters, atoms_data, bonds

def get_box_properties(lmp):
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo 
    xhi, yhi, zhi = boxhi 
    return {'xlo':xlo, 'ylo':ylo, 'zlo':zlo, 'xhi':xhi, 'yhi':yhi, 'zhi':zhi}

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
    positions = [(x, y) for _, mol_id, _, x, y, z in atoms_data if int(mol_id) == target_mol_id]
    
    # Check if any positions were found
    if not positions:
        return None  # No atoms found for the given molecule ID
    
    # Calculate the center of mass
    x_com = sum(pos[0] for pos in positions) / len(positions)
    y_com = sum(pos[1] for pos in positions) / len(positions)
    
    return np.array([x_com, y_com])
    
def analyse_simulation_status(cutoff, lmp, print_mode = False, factor_cutoff = 0.9):
    
    """
    Analyze the system at a given step of the simulation
    Create a graph of the system and print the sizes of the connected components, 
    the largest molecule ID found in each graph, 
    and the number of monomers left in the left and right sides of the box
    """

    if factor_cutoff*cutoff<=1.0:
        print('Problem: It will never detect binding.')
        sys.exit(1)

    # generate graph of the sytem based on the status of the simulation
    clusters, atoms_data, bonds = graph_new(factor_cutoff*cutoff, lmp)
    
    free_monomers, left_monomers_mol_list, right_monomers_mol_list = [], [], []
    left_monomers_num, right_monomers_num = 0, 0

    max_length = max([len(sublist) for sublist in clusters])
    monomers_free_num = sum(1 for sublist in clusters if len(sublist) == 1)

    # determine all single-monomer/component clusters
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    #print(f"FREE MONOMERS: {free_monomers}")

    for monomer in free_monomers:
        try: 
            com = center_of_mass(atoms_data, target_mol_id = monomer)
            comx = com[0]
            # monomers to the left of the seed
            if comx<-2:
                left_monomers_num+=1
                left_monomers_mol_list.append(monomer)
            
            # monomers to the right of the seed
            elif comx>2:
                right_monomers_num+=1
                right_monomers_mol_list.append(monomer)
        except:
            continue

    # convert list to set
    bonds = set(bonds)  
    dimer = False
    if (1,2) in bonds or (2,1) in bonds:
        dimer = True

    if print_mode:
        print('Step:', step, '- Frame:', step/1000)
        print('Monomers left in left side:', monomers_free_num)
        print()

    return left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters

def equilibrium_polymerization_run(SimulationParameters, factor_increase = 2, factor_position = 1.3):
    
    '''
    Run a LAMMPS simulation with intervals at which the simulation box is analyzed,
    extended, and the concentration of monomers kept constant. 
    It is called 'equilibrium_polymerization_run' because we do not have hydrolysis.
    '''

    print("Starting simulation with growing box!")

    # instantiate LAMMPS (without printing any output)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])  

    # read the input file to set up the system
    lmp.file(SimulationParameters.inputfile)

    # get size and geometry of the simulation box
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo") 
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()

    # lower bounds
    xlo, ylo, zlo = boxlo  
    # upper bounds
    xhi, yhi, zhi = boxhi  

    print("Initial simulation bounds are: ")
    print(f"XLO: {xlo} XHI: {xhi}")
    print(f"YLO: {xlo} YHI: {xhi}")
    print(f"ZLO: {xlo} ZHI: {xhi}")

    xlo_init = xlo
    xhi_init = xhi

    # find number of monomers
    molecule_number = 1

    # get positions, molecules and ids
    step  = lmp.get_thermo('step')
    pos   = lmp.numpy.extract_atom("x")
    mol   = lmp.numpy.extract_atom("molecule")
    ids   = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")

    # location com of the system
    xcm = np.mean(pos[:, 0])
    ycm = np.mean(pos[:, 1])
    
    # analyze status simulation
    left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation_status(SimulationParameters.cutoff_top,lmp)

    #get max and min x position of polymer
    com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(pos,mol,clusters)

    # loops of MD runs
    counter, counter_left, counter_right = 1, 0, 0

    for i in range(SimulationParameters.loops):

        # run for a short interval
        lmp.command('run %d'%(SimulationParameters.run_interval))

        # retrieve system information
        pos = lmp.numpy.extract_atom("x")
        mol = lmp.numpy.extract_atom("molecule")

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        xlo, ylo, zlo = boxlo 
        xhi, yhi, zhi = boxhi 
        molecule_number = max(mol)

        # analyze status simulation
        left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)
        
        #print(f"MONOMERS ON THE LEFT: {left_monomers_num}")
        #print(f"MONOMERS ON THE RIGHT: {right_monomers_num}")
        #print(f"DIMER: {dimer}")
        #print(f"MONOMERS_FREE_NUM: {monomers_free_num}")
        
        # we need to add a monomer on the left
        if left_monomers_num < SimulationParameters.nMonsSol:
            
            #print("Add monomers to the left")

            nmonomers = SimulationParameters.nMonsSol-left_monomers_num

            # (we just did it above, no need to repeat?)
            # analyze status simulation
            # left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation_status(SimulationParameters.cutoff_top,lmp)
            
            # get max and min x position of polymer
            com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(pos,mol,clusters)
            #print("Current com_x_max_mol and com_x_min_mol: ", com_x_max_mol, com_x_min_mol)
            
            # update only low wall, as this is where we need the new monomer
            x_walls_lo = com_x_min_mol - factor_increase*SimulationParameters.bbscale
            x_walls_hi = xhi
            #print(f"Walls will be updated to: {x_walls_lo} : {x_walls_hi}")

            lmp.command('unfix xwalls')
            lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('run 10')

            # define an insertion point
            lmp.command("region              insertion_region_left block %f %f %f %f 0 0 units box" %(com_x_min_mol - factor_position*SimulationParameters.bbscale, com_x_min_mol - factor_position*SimulationParameters.bbscale, ycm, ycm))
            lmp.command('fix                 add_mons_left moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left id next'%(nmonomers,SimulationParameters.seed))
            
            counter +=1
            counter_left +=1
            lmp.command('run 10')

            lmp.command('unfix add_mons_left')
            lmp.command('region insertion_region_left delete')

            # NECESSARY!!
            # walls have been updated, reintroduce this information for python
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo 
            xhi, yhi, zhi = boxhi 

        if right_monomers_num < SimulationParameters.nMonsSol:
            
            #print("Add monomers to the right")

            nmonomers = SimulationParameters.nMonsSol-right_monomers_num

            # (no need to analyze again since it has already
            # been done above) analyze status simulation
            # left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation_status(SimulationParameters.cutoff_top,lmp)
            
            #get max and min x position of polymer
            com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(pos,mol,clusters)
            #print("Current com_x_max_mol and com_x_min_mol: ", com_x_max_mol, com_x_min_mol)

            # update only upper wall, as this is where we need the new monomer
            x_walls_lo = xlo
            x_walls_hi = com_x_max_mol + factor_increase*SimulationParameters.bbscale
            #print(f"Walls will be updated to: {x_walls_lo} : {x_walls_hi}")

            lmp.command('unfix xwalls')
            lmp.command('change_box          all x final %f %f \n\n' %(x_walls_lo, x_walls_hi))
            lmp.command("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
            lmp.command('run 10')
            
            lmp.command("region              insertion_region_right block %f %f %f %f 0 0 units box" %(com_x_max_mol+factor_position*SimulationParameters.bbscale, com_x_max_mol+factor_position*SimulationParameters.bbscale, ycm, ycm))
            lmp.command('fix                 add_mons_right moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right id next'%(nmonomers,SimulationParameters.seed+1))

            counter +=1
            counter_right+=1
            lmp.command('run 10')

            lmp.command('unfix add_mons_right')
            lmp.command('region insertion_region_right delete')
        
            # walls have been updated, reintroduce this information for python
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo 
            xhi, yhi, zhi = boxhi 

        # .... delete monomer if more monomers than the desired concentration on the left
        if left_monomers_num > SimulationParameters.nMonsSol:

            # remove the last added monomer
            max_mol_left = max(left_monomers_mol_list) 
            lmp.command("group max_mol_left molecule %d" %(max_mol_left))
            lmp.command("delete_atoms group max_mol_left")
            lmp.command('run 10')
            # delete group to avoid problems
            lmp.command("group max_mol_left delete")

        # .... delete monomer if more monomers than the desired concentration on the right
        if right_monomers_num > SimulationParameters.nMonsSol: 
            
            # remove the last added monomer
            max_mol_right = max(right_monomers_mol_list)
            lmp.command("group max_mol_right molecule %d" %(max_mol_right))
            lmp.command("delete_atoms group max_mol_right")
            lmp.command('run 10')

            # delete group to avoid problems
            lmp.command("group max_mol_right delete")

    # final graph
    L     = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo")
    step  = lmp.get_thermo('step')
    pos   = lmp.numpy.extract_atom("x")
    mol   = lmp.numpy.extract_atom("molecule")
    ids   = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")

    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo 
    xhi, yhi, zhi = boxhi 

    #molecule_number = max(mol)
    left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, clusters = analyse_simulation(SimulationParameters.cutoff, lmp)
    
    # clean up
    lmp.close()

def write_polymerization_program(dirname):

    to_write="""
from lib_littleguys import polymerization, parameters
import json

# load the json file in the directory
with open(f"parameters.json", "r") as f:
    data = json.load(f)

# read the parameters for the simulation
Simulation = parameters.SimulationParameters(**data)

# launch the simulation
polymerization.equilibrium_polymerization_run(Simulation)
    """

    with open(dirname+"/directional_growth.py", 'w') as f:
        f.write(to_write)


if __name__=="__main__":
    
    # ----------------------------------#
    #  RUN A DIRECTIONAL GROWTH PROGRAM #
    # ----------------------------------#

    # parse arguments from command line
    args = parse_arguments()

    # load the json file in the directory
    with open(f"{args.path}/parameters.json", "r") as f:
        data = json.load(f)

    # read the parameters for the simulation
    Simulation = parameters.SimulationParameters(**data)

    # launch the simulation
    equilibrium_polymerization_run(Simulation)
