import lammps 
import numpy as np
from collections import defaultdict, deque
import random, sys
from lib_littleguys import analysis_functions

def hydrolysis_happens(hydrolysis_probability): #returns true or false, DONE
    # Generate a random number between 0 and 1
    U = random.random()
    # Check if the event occurs
    return U < hydrolysis_probability

def get_bonds(cutoff, lmp):

    positions   = lmp.numpy.extract_atom("x")
    mols        = lmp.numpy.extract_atom("molecule")
    atom_ids    = lmp.numpy.extract_atom("id")
    types       = lmp.numpy.extract_atom("type")
    #dimensions  = get_box_properties(lmp)
    #box_bounds = (dimensions['xlo'], dimensions['xhi'], dimensions['ylo'], dimensions['yhi'])
    
    # group all atom properties
    atoms_data = np.column_stack((atom_ids, mols, types))
    atoms_data = np.column_stack((atoms_data, positions))
    num_molecules = np.max(mols)
    
    # find the molecules in the system
    molecules    = defaultdict(list)
    molecule_ids = []
    for atom in atoms_data:
        atom_id, mol_id, atom_type, x, y, z = atom
        # do not look at wall particles (type 8)
        if atom_type != 8: 
            molecules[mol_id].append((atom_id, atom_type, x, y))

    # Find touching molecules and add bonds
    connected_molecules = []
    bonded_molecules    = []
    for mol_id1 in molecules:
        for mol_id2 in molecules:
            if mol_id1 < mol_id2:
                if analysis_functions.are_molecules_touching(molecules[mol_id1], molecules[mol_id2], cutoff, model0_condition = True):
                    connected_molecules.append((mol_id1, mol_id2))
                    if mol_id1 not in bonded_molecules:
                        bonded_molecules.append(mol_id1)
                    if mol_id2 not in bonded_molecules:
                        bonded_molecules.append(mol_id2)
                    
    return connected_molecules, atoms_data, num_molecules, bonded_molecules, mols, positions

def find_clusters(particles, bonds):#DONE
    
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

def graph_new(cutoff, lmp):
    
    # extract system information
    bonds, atoms_data, num_of_molecules, bonded_molecules, mols, pos = get_bonds(cutoff, lmp)

    # create graph
    clusters = find_clusters(range(1,num_of_molecules+1), bonds)
    #print("These are the clusters: ", clusters)
    
    # Keep clusters if:
    # - they are longer than 1 (assumed always valid), or
    # - the single element is in mol
    # this is due to deletion of monomers
    filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1 or cluster[0] in mols]
    #print("These are the filtered clusters ", filtered_clusters)

    return filtered_clusters, atoms_data, bonds,bonded_molecules, mols, pos

def get_box_properties(lmp):
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo 
    xhi, yhi, zhi = boxhi 
    return {'xlo':xlo, 'ylo':ylo, 'zlo':zlo, 'xhi':xhi, 'yhi':yhi, 'zhi':zhi}

def center_of_mass(atoms_data, target_mol_id):#DONE
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

    clusters, atoms_data, bonds, bonded_molecules, mols, pos = graph_new(factor_cutoff*cutoff, lmp)
    # get the longest cluster within the connected components
    max_cluster = max(clusters, key=len)
    # create boolean mask to find molecules within largest cluster
    mask = np.isin(mols, max_cluster)
    # get positions of atoms in molecules in largest cluster
    pos_selected = pos[mask]
    x_positions = pos_selected[:, 0]

    # Get min and max x-values of the longest cluster
    x_min = x_positions.min()
    x_max = x_positions.max()

    # number of currently free monomers
    free_monomers = []
    for sublist in clusters:
        if len(sublist)==1:
            free_monomers.append(sublist[0])
    monomers_free_num = len(free_monomers)

    flag_allmonomers = False
    if monomers_free_num == len(clusters):
        flag_allmonomers = True

    """
    # if all the monomers are free
    if monomers_free_num == len(clusters):
        print("Everything is monomers")
        # find which monomer is the minimum
        min_x = 1000000000
        max_x = -1000000000
        for monomer in free_monomers:
            if monomer in mols: 
                com = center_of_mass(atoms_data, target_mol_id = monomer)
                comx = com[0]
                # monomers to the left of the leftmost in the cluster in consideration
                if comx<min_x:
                    min_x = comx 
                    index_min = monomer
                if comx>max_x:
                    max_x = comx 
                    index_max = monomer 

        x_min = 1000000000
        x_max = -1000000000 
        for monomer in free_monomers:
            if monomer in mols: 
                if monomer!= index_min:
                    com = center_of_mass(atoms_data, target_mol_id = monomer)
                    comx = com[0]
                    # monomers to the left of the leftmost in the cluster in consideration
                    if comx<x_min:
                        x_min = comx 
                        index_min = monomer
                    if comx>x_max:
                        x_max = comx 
                        index_max = monomer 
        x_min += 2
        x_max += 2
    """

    # if there is some bonded particle
    left_monomers_mol_list, right_monomers_mol_list = [], []
    left_monomers_num, right_monomers_num = 0, 0
    #print(f"XMIN : {x_min} XMAX: {x_max}")

    for monomer in free_monomers:
        if monomer in mols: 
            com = center_of_mass(atoms_data, target_mol_id = monomer)
            comx = com[0]
            #print("Monomer ", monomer, " XCOM ", comx)
            # monomers to the left of the leftmost in the cluster in consideration
            if comx<x_min:
                left_monomers_num+=1
                left_monomers_mol_list.append(monomer)
            
            # monomers to the right of the rightmost in the cluster in consideration
            elif comx>x_max:
                right_monomers_num+=1
                right_monomers_mol_list.append(monomer)

    # convert list to set
    bonds = set(bonds)  
    dimer = False
    if (1,2) in bonds or (2,1) in bonds:
        dimer = True

    if print_mode:
        print('Step:', step, '- Frame:', step/1000)
        print('Monomers left in left side:', monomers_free_num)
        print()

    #print(f"LEFT {left_monomers_num} ({left_monomers_mol_list}) RIGHT {right_monomers_num} ({right_monomers_mol_list})")
    return atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers

def get_most_left_and_right_com(lmp, clusters):
    #gets positions of most left and right monomers in polymer for insertion of new monomers

    all_pos   = lmp.numpy.extract_atom("x")
    all_mol   = lmp.numpy.extract_atom("molecule")
    types     = lmp.numpy.extract_atom("type")
    
    mask = types != 8
    pos = all_pos[mask] # dont consider wall particles
    mol = all_mol[mask] # dont consider wall particles

    # create mask to track only molecules in the polymer - not free monomers
    mask = np.array([any((x in subgroup) and (len(subgroup) > 1) for subgroup in clusters) for x in mol])
    
    # if all the molecules in the system are free monomers
    if not (any(mask) == True):
        # current molecules
        print("No filament notion")
        #wait = input()
        num_clusters = len(clusters)
        # identify which is the max, which is the min -- and remove one of the two
        # to redefine the system from there
        xmin = 10000000000
        xmax = -10000000000
        for subgroup in clusters:
            moleculeID = subgroup[0]
            index_pos  = np.where(mol == moleculeID)[0][0]
            xy = pos[index_pos, 0]
            x  = np.mean(xy)
            if x<xmin:
                xmin = x
                index_min = moleculeID
            if x>xmax:
                xmax = x
                index_max = moleculeID
        
        xmin = 10000000000
        xmax = -10000000000
        # we choose to ignore the min
        for subgroup in clusters:
            moleculeID = subgroup[0]
            if moleculeID != index_min:
                index_pos  = np.where(mol == moleculeID)[0][0]
                xy = pos[index_pos, 0]
                x  = np.mean(xy)
                if x<=xmin:
                    xmin = x
                if x>=xmax:
                    xmax = x

        return xmax, xmin

    # if there is some notion of a filament
    else:
        pos = pos[mask]
        mol = mol[mask]

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

def equilibrium_treadmilling_run(SimulationParameters):
    
    '''
    Run a LAMMPS simulation with intervals at which the simulation box is analyzed,
    extended, and the concentration of monomers kept constant. 
    The run has hydrolysis implemented, which means that the interface weakens.
    '''

    print("Starting simulation with growing box!")

    # instantiate LAMMPS (without printing any output)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])  
    
    # read the input file to set up the system
    lmp.file(SimulationParameters.inputfile)

    # keep track of hydrolyzed molecules
    #list_of_hydrolyzed_molecules = dict()
    unfixed_mols = []

    # get size and geometry of the simulation box
    L = lmp.extract_global("boxxhi") - lmp.extract_global("boxxlo") 
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
    xlo, ylo, zlo = boxlo  
    xhi, yhi, zhi = boxhi  

    # find number of monomers
    molecule_number = 1

    # get positions, molecules and ids
    step  = lmp.get_thermo('step')
    pos   = lmp.numpy.extract_atom("x")
    mol   = lmp.numpy.extract_atom("molecule")
    ids   = lmp.numpy.extract_atom("id")
    types = lmp.numpy.extract_atom("type")

    # initial position of the com of the system
    oldcmx = np.mean(pos[:, 0])
    oldcmy = np.mean(pos[:, 1])

    # place to save the com in time
    fcom = open(SimulationParameters.comfile, 'w')

    # distance between the com of the most left/right monomer in polymer and the left/right box edge
    delta_L = SimulationParameters.faci*SimulationParameters.bbscale

    # analyze status simulation
    atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)

    # get com of the leftmost and rightmost monomers
    com_x_max_mol,com_x_min_mol = get_most_left_and_right_com(lmp,clusters)

    # add initial walls to left and right of inserted polymer
    x_walls_lo = com_x_min_mol - SimulationParameters.faci*SimulationParameters.bbscale
    x_walls_hi = com_x_max_mol + SimulationParameters.faci*SimulationParameters.bbscale
    lmp.command("fix walls all wall/reflect xlo %f xhi %f" %(x_walls_lo, x_walls_hi))
    lmp.command('create_atoms 8 single %d %d 0'%(x_walls_lo + 0.5, oldcmy))
    lmp.command('create_atoms 8 single %d %d 0'%(x_walls_hi - 0.5, oldcmy))

    # save for updates
    left_8 = x_walls_lo
    right_8 = x_walls_hi

    # loops of MD runs
    counter, counter_left, counter_right = 1, 0, 0

    i, round_counter = -1, 1

    print("Starting treadmilling loop...")

    # to make really long simulations
    while True:
    #for i in range(SimulationParameters.loops):
        round_counter+=1

        # .............................
        # ON THE FLY ANALYSIS FOR SIMS
        # current position of the com 
        # (after all the updates, before running more)
        # (it also includes unbound monomers -- it's fine, minor error)
        xcmcurrent = np.mean(pos[:, 0])
        ycmcurrent = np.mean(pos[:, 1])

        savecurrentxcm = xcmcurrent
        savecurrentycm = ycmcurrent

        diff_cm_x = xcmcurrent-oldcmx
        diff_cm_y = ycmcurrent-oldcmy
        
        oldcmx = xcmcurrent
        oldcmy = ycmcurrent

        # if COM displaces more than two monomers, jump?
        if np.abs(diff_cm_x)>(2*SimulationParameters.bbscale):
            # we therefore correct and place the com in the previous
            # to neglect such a large jump 
            xcmcurrent = xcmcurrent - oldcmx
            ycmcurrent = ycmcurrent - oldcmy
        
        if round_counter%SimulationParameters.round_frequency == 0:
            # xcm current includes a potentially updated xcm that ignores jumps
            # savecurrentcm is the true position of the com
            # oldcm is the one from the previous step
            current_step  = lmp.get_thermo('step')
            fcom.writelines(f"{current_step} {xcmcurrent} {ycmcurrent} {savecurrentxcm} {savecurrentycm} {oldcmx} {oldcmy}\n")
            fcom.flush()
        # .............

        # run the simulation forward
        lmp.command('run %d'%(SimulationParameters.run_interval))
        pos   = lmp.numpy.extract_atom("x")
        mol   = lmp.numpy.extract_atom("molecule")
        types = lmp.numpy.extract_atom("type")
        step  = lmp.get_thermo('step')

        # get largest molecule ID
        molecule_number = max(mol)

        # simulation box information
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        xlo, ylo, zlo = boxlo 
        xhi, yhi, zhi = boxhi 

        # analyze status of the simulation
        atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)
        # get left and rightmost ends of the cluster of interest
        com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
        # get list of molecules that have been hydrolyzed
        #hydrolyzed_monomers = [key for key in list_of_hydrolyzed_molecules]

        # if they are all monomers, do not do any updates in system until something binds
        if not flag_allmonomers:

            # -----------------------------------------
            # ENSURING CONSTANT MONOMER CONCENTRATION
            # -----------------------------------------
            if left_monomers_num < SimulationParameters.nMonsSol:
                
                #print("Adding a monomer on the left!")
                nmonomers = SimulationParameters.nMonsSol-left_monomers_num

                # only update one side of the filament
                x_walls_lo = com_x_min_mol - delta_L
                left_8     = x_walls_lo
                x_walls_hi = right_8

                lmp.command('unfix walls')
                lmp.command('delete_atoms group type_8_dynamic')
                lmp.command('run 10')
                lmp.command("fix walls all wall/reflect xlo %f xhi %f" %(x_walls_lo, x_walls_hi))
                lmp.command('run 10')
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_lo + 0.5, oldcmy))
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_hi - 0.5, oldcmy))
                lmp.command("region  insertion_region_left block %f %f %f %f 0 0 units box" %(com_x_min_mol - SimulationParameters.facp*SimulationParameters.bbscale, com_x_min_mol - SimulationParameters.facp*SimulationParameters.bbscale, oldcmy, oldcmy))
                lmp.command('fix     add_mons_left moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_left '%(nmonomers,SimulationParameters.seed))
                lmp.command('run 10')
                lmp.command('unfix add_mons_left')
                lmp.command('region insertion_region_left delete')
                
                counter +=1
                counter_left +=1

            if right_monomers_num < SimulationParameters.nMonsSol:

                #print("Adding a monomer to the right!")
                nmonomers = SimulationParameters.nMonsSol-right_monomers_num
                
                # only update one side of the filament
                x_walls_lo = left_8
                x_walls_hi = com_x_max_mol+delta_L
                right_8    = x_walls_hi

                lmp.command('unfix walls')
                lmp.command('delete_atoms group type_8_dynamic')
                lmp.command('run 10')
                lmp.command("fix walls all wall/reflect xlo %f xhi %f" %(x_walls_lo, x_walls_hi))
                lmp.command('run 10') 
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_lo + 0.5, oldcmy))
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_hi - 0.5, oldcmy))
                lmp.command('run 10') 
                lmp.command("region              insertion_region_right block %f %f %f %f 0 0 units box" %(com_x_max_mol+SimulationParameters.facp*SimulationParameters.bbscale, com_x_max_mol+SimulationParameters.facp*SimulationParameters.bbscale, oldcmy, oldcmy))
                lmp.command('fix                 add_mons_right moving deposit %d 0 1000000000 %d mol monomer_template region insertion_region_right '%(nmonomers,SimulationParameters.seed+1))
                lmp.command('run 10')
                lmp.command('unfix add_mons_right')
                lmp.command('region insertion_region_right delete')
                
                counter +=1
                counter_right+=1
            # -----------------------------------------

            """
            # --------------------------------------------------------------
            #   APPLY HYDROLYSIS (AT THE MOMENT IT DOES NOT WORK LIKE THIS)
            # --------------------------------------------------------------

            # analyze status of the simulation
            atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)
            # get left and rightmost ends of the cluster of interest
            com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(lmp,clusters)

            considered_molecules = []
            for molecule in bonded_molecules:
                if molecule not in considered_molecules:
                    if (molecule not in hydrolyzed_monomers):
                        # if molecule hasnt already been hydrolysed and if hydrolysis happens, weaken bond
                        if hydrolysis_happens(SimulationParameters.hydrolysis_probability): 
                            lmp.command('group mol_%d molecule %d' %(molecule, molecule))
                            lmp.command('variable reduce_epsilon_top_%d equal %f'%(molecule,SimulationParameters.epsilon_top_final)) #ADD PARAMETERS TO THE JSON FILE
                            lmp.command('fix    hydrolyze_top_%d mol_%d adapt 0 pair cosine/squared eps 4 6 v_reduce_epsilon_top_%d reset no scale no' %(molecule, molecule,molecule))
                            lmp.command('fix    dump_v1_%d all print 1 "Epsilon_top: ${reduce_epsilon_top_%d}" file eps_top_%d.txt screen no'%(molecule,molecule,molecule))
                            list_of_hydrolyzed_molecules[molecule] = step
                # do not consider molecule again
                considered_molecules.append(molecule)

            # run briefly to input commands
            lmp.command('run 10')

            # clean-up fixes and ensure that we don't try to unfix same molecules again
            for molecule, hydrolysis_step in list_of_hydrolyzed_molecules.items():
                if molecule not in unfixed_mols:
                    lmp.command('unfix hydrolyze_top_%d' %(molecule))
                    lmp.command('unfix dump_v1_%d' %(molecule)) 
                    lmp.command('group mol_%d delete' %(molecule))
                    unfixed_mols.append(molecule)
            
            # run briefly to input commands
            lmp.command('run 10')
            """

            # -----------------------------------------
            #      DELETE BROKEN PARTS OF SYSTEM
            # -----------------------------------------

            # analyze status of the simulation
            atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)
            # get left and rightmost ends of the cluster of interest
            com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
            
            # getting number of polymers, free monomers from the clusters above with HIGHER CUTOFF
            polymer_number   = sum(1 for sublist in clusters if len(sublist) > 1)
            #if breaking occurs keep only the longest filament
            max_polymer_size = max(clusters, key=len) 
            # number of free monomers (previously calculated)
            monomer_number = monomers_free_num
            # determine how many free monomers we allow in the system
            flattened_mols = [item for sublist in clusters for item in sublist]
            if len(max_polymer_size) ==1:
                monomer_number_max = 3 #if only one monomer=seed - 3 monomers are ok
            elif len(max_polymer_size)>1:
                monomer_number_max = 2

            molecules_to_delete = []
            #if more than one polymer exists (fragmentation) delete other polymers+ monomers on that side only
            if polymer_number>1: 
                #print('Polymer number:' + str(polymer_number))
                #print('Monomer number:' + str(monomer_number))
                for molecule in flattened_mols:
                    # if the molecule being considered is not part of the polymer
                    if molecule not in max_polymer_size:
                        # for the connected components in the system
                        for cluster in clusters:
                            # if this is not a monomer, and the molecule is in it
                            if len(cluster)>1 and molecule in cluster:
                                # determine where this particule is
                                comx, comy = center_of_mass(atoms_data, target_mol_id = molecule)
                                # determine where the filament is
                                com_longest_filament_x, com_longest_filament_y = center_of_mass(atoms_data, target_mol_id = max_polymer_size[0])
                                # molecule must be deleted
                                molecules_to_delete.append(molecule)
                                if comx<com_longest_filament_x:
                                    for left_mol in left_monomers_mol_list:
                                        molecules_to_delete.append(left_mol)
                                elif comx>com_longest_filament_x:
                                    for right_mol in right_monomers_mol_list:
                                        molecules_to_delete.append(right_mol)
            
            #print(f"DELETING--->MONOMER_NUMBER: {monomer_number} MONOMER_NUMBER_MAX {monomer_number_max}")
            # if too many free monomers exist (end depolymerisation), delete all monomers on this ONE side
            if monomer_number>monomer_number_max: 
                if left_monomers_num>SimulationParameters.nMonsSol:
                    #print("LEFT MONOMERS: ", left_monomers_mol_list)
                    for mols in left_monomers_mol_list:
                        molecules_to_delete.append(mols)
                elif right_monomers_num>SimulationParameters.nMonsSol:
                    #print("RIGHT MONOMERS: ", right_monomers_mol_list)
                    for mols in right_monomers_mol_list:
                        molecules_to_delete.append(mols)
                    
            mol  = lmp.numpy.extract_atom("molecule")
            deleted = False
            for mols in molecules_to_delete:
                if mols in mol:
                    deleted = True
                    lmp.command('group mol_%d molecule %d'%(mols,mols))
                    lmp.command('delete_atoms group mol_%d'%(mols))
                    lmp.command('group mol_%d delete '%(mols))
                    #deleted_atoms.append(mols)
                    #if mols in list_of_binding_events: #since LAMMPS recycles molecule IDs, we remove the information we had on the previous monomer during its deletion
                    #    list_of_binding_events.pop(mols)
                    #if mols in list_of_hydrolyzed_molecules:
                    #    list_of_hydrolyzed_molecules.pop(mols)
                    if mols in unfixed_mols:
                        unfixed_mols.remove(mols)

            # run briefly to input commands
            lmp.command('run 10')

            # after deleting the molecules, we decrease the size of the box to just about the size of the polymer
            if deleted:
                
                # recompute system properties bc there have been updates
                pos   = lmp.numpy.extract_atom("x")
                mol   = lmp.numpy.extract_atom("molecule")
                types = lmp.numpy.extract_atom("type")

                # reanalyze the system again after all the deletions
                atoms_data, left_monomers_num, right_monomers_num, dimer, monomers_free_num, left_monomers_mol_list, right_monomers_mol_list, bonded_molecules, clusters, flag_allmonomers = analyse_simulation_status(SimulationParameters.cutoff_top, lmp)
                com_x_max_mol, com_x_min_mol = get_most_left_and_right_com(lmp,clusters)
                
                # update the information about the walls
                x_walls_lo = com_x_min_mol - SimulationParameters.faci*SimulationParameters.bbscale
                x_walls_hi = com_x_max_mol + SimulationParameters.faci*SimulationParameters.bbscale
                left_8  = x_walls_lo
                right_8 = x_walls_hi

                lmp.command('unfix walls') #make new walls to fit the remaining polymer
                lmp.command('delete_atoms group type_8_dynamic') #delete previous particles labeling wall position - type 8
                lmp.command("fix walls all wall/reflect xlo %f xhi %f" %(x_walls_lo, x_walls_hi))
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_lo + 0.5, oldcmy)) #create new wall labeling particles
                lmp.command('create_atoms 8 single %d %d 0'%(x_walls_hi - 0.5, oldcmy))
                lmp.command('run 10')

        else:
            step  = lmp.get_thermo('step')
            fopen = open("trackingfile.dat", "w")
            fopen.writelines(f"Dead filament at step = {step}")
            print("We lost all the monomers")
            sys.exit(1)

    # clean up
    lmp.close()
    fcom.close()

def write_treadmilling_program(dirname):

    to_write=f"""
from lib_littleguys import treadmilling, parameters
import json

# load the json file in the directory
with open(f"parameters.json", "r") as f:
    data = json.load(f)

# read the parameters for the simulation
Simulation = parameters.SimulationParameters(**data)

# launch the simulation
treadmilling.equilibrium_treadmilling_run(Simulation)
    """

    with open(dirname+"/treadmilling.py", 'w') as f:
        f.write(to_write)


if __name__=="__main__":
    
    # ---------------------------- #
    #  RUN A TREADMILLING PROGRAM  #
    # ---------------------------- #

    # parse arguments from command line
    args = parse_arguments()

    # load the json file in the directory
    with open(f"{args.path}/parameters.json", "r") as f:
        data = json.load(f)

    # read the parameters for the simulation
    Simulation = parameters.SimulationParameters(**data)

    # launch the simulation
    equilibrium_treadmilling_run(Simulation)
