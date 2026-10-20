import numpy as np 
from lib_littleguys.molecule_templates import generate_buildingblock_structure, generate_chevron_structure, generate_triangle_structure

def write_initconfig(SimulationParameters, is_seed = False, treadmilling = False):

    """
    Function that writes the initial configuration
    for a system with N building blocks, all in the
    closed configuration.
    """
    
    # place to save the coordinates
    coords = np.zeros((SimulationParameters.particles_per_monomer*SimulationParameters.numblocks, 2))

    # initialize displacement
    displace = [0, 0]

    if is_seed:
        is_open = False 
    else:
        is_open = True
    generate_buildingblock_structure(i = 0, coords = coords, SimulationParameters= SimulationParameters, open = is_open)
    coords[0:SimulationParameters.particles_per_monomer]*=SimulationParameters.bbscale
    
    # write coordinates seed in closed config
    for i in range(1, SimulationParameters.numblocks):
        generate_buildingblock_structure(i = i*SimulationParameters.particles_per_monomer, coords = coords, SimulationParameters= SimulationParameters, open = False)
        coords[i*SimulationParameters.particles_per_monomer:i*SimulationParameters.particles_per_monomer+SimulationParameters.particles_per_monomer]*=SimulationParameters.bbscale
        coords[i*SimulationParameters.particles_per_monomer:i*SimulationParameters.particles_per_monomer+SimulationParameters.particles_per_monomer,0]+=i*(SimulationParameters.bbscale+SimulationParameters.buffer)
    
    # recenter the polymer
    xcm = np.mean(coords[:, 0])
    coords[:, 0] -= xcm

    # update the initial boundaries of the system to fit the polymer (however many particles there are)
    if SimulationParameters.xlo is None:
        SimulationParameters.xlo = np.min(coords[:, 0]) - 1.2*SimulationParameters.bbscale
        SimulationParameters.xhi = np.max(coords[:, 0]) + 1.2*SimulationParameters.bbscale

    num_atom_types = SimulationParameters.particles_per_monomer+1
    if treadmilling:
        # there are three new atom types when the system allows for hydrolysis
        num_atom_types+=3

    to_write=f"""
# configuration file for the initial conditions of the Model 0 molecules simulations

{SimulationParameters.particles_per_monomer*SimulationParameters.numblocks} atoms
{SimulationParameters.bonds_per_monomer*SimulationParameters.numblocks} bonds
{SimulationParameters.angles_per_monomer*SimulationParameters.numblocks} angles
{num_atom_types} atom types 
3 bond types
4 angle types
{SimulationParameters.xlo} {SimulationParameters.xhi} xlo xhi
{SimulationParameters.ylo} {SimulationParameters.yhi} ylo yhi
-0.25 0.25 zlo zhi

Masses

""" 
    for i in range(num_atom_types):
        to_write+=f"""{i+1} 1\n"""

    to_write+="""\nAtoms\n\n"""
    
    molecule_counter = 0
    type_counter     = 1

    for i in range(SimulationParameters.particles_per_monomer*SimulationParameters.numblocks):

        # increase the counting of molecules
        if i % SimulationParameters.particles_per_monomer == 0:
            molecule_counter+=1

        to_write+=f"""{i+1} {molecule_counter} {type_counter} {coords[i, 0]} {coords[i, 1]} 0.00\n"""

        # increase the counting of types
        type_counter+=1

        # reset the type counter
        if type_counter == (SimulationParameters.particles_per_monomer+1):
            type_counter = 1

    to_write+="""\nBonds\n\n"""

    for i in range(SimulationParameters.numblocks):
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+1} 1 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2}\n"""
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+2} 2 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+3}\n"""
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+3} 1 {i*SimulationParameters.particles_per_monomer+4} {i*SimulationParameters.particles_per_monomer+5}\n"""
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+4} 2 {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+6}\n"""
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+5} 3 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5}\n"""
        to_write+=f"""{i*SimulationParameters.particles_per_monomer+6} 2 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+7}\n"""

    to_write+="""\nAngles\n\n"""

    if SimulationParameters.coupled_interfaces:
        for i in range(SimulationParameters.numblocks):
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+1} 1 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+3}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+2} 1 {i*SimulationParameters.particles_per_monomer+4} {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+6}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+3} 2 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+4} 2 {i*SimulationParameters.particles_per_monomer+3} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+5} 3 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+6}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+6} 4 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+4}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+7} 2 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+7}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+8} 2 {i*SimulationParameters.particles_per_monomer+3} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+7}\n"""
    else:
        for i in range(SimulationParameters.numblocks):
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+1} 1 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+3}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+3} 2 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+4} 2 {i*SimulationParameters.particles_per_monomer+3} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+5} 3 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+6}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+6} 4 {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+5} {i*SimulationParameters.particles_per_monomer+4}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+7} 2 {i*SimulationParameters.particles_per_monomer+1} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+7}\n"""
            to_write+=f"""{i*SimulationParameters.particles_per_monomer+8} 2 {i*SimulationParameters.particles_per_monomer+3} {i*SimulationParameters.particles_per_monomer+2} {i*SimulationParameters.particles_per_monomer+7}\n"""
    with open(SimulationParameters.configfile, 'w') as f:
        f.write(to_write)

def write_initconfig_chevrons(SimulationParameters):

    """
    Function that writes the initial configuration
    for a system with N chevrons
    """
    
    # place to save the coordinates (8 particles bc of hat)
    SimulationParameters.particles_per_monomer = 8
    coords = np.zeros((SimulationParameters.particles_per_monomer*SimulationParameters.numblocks, 2))

    # initialize displacement
    displace = [0, 0]

    generate_chevron_structure(i = 0, coords = coords, SimulationParameters= SimulationParameters)
    
    # write coordinates seed in closed config
    for i in range(1, SimulationParameters.numblocks):
        generate_chevron_structure(i = i*SimulationParameters.particles_per_monomer, coords = coords, SimulationParameters= SimulationParameters)
        coords[i*SimulationParameters.particles_per_monomer:i*SimulationParameters.particles_per_monomer+SimulationParameters.particles_per_monomer,0]+=i*2
    
    # recenter the polymer
    xcm = np.mean(coords[:, 0])
    coords[:, 0] -= xcm

    # update the initial boundaries of the system to fit the polymer (however many particles there are)
    if SimulationParameters.xlo is None:
        SimulationParameters.xlo = np.min(coords[:, 0]) - 1.2*SimulationParameters.bbscale
        SimulationParameters.xhi = np.max(coords[:, 0]) + 1.2*SimulationParameters.bbscale

    num_atom_types = SimulationParameters.particles_per_monomer+1

    to_write=f"""
# configuration file for the initial conditions of simulations with chevrons

{SimulationParameters.particles_per_monomer*SimulationParameters.numblocks} atoms
{num_atom_types} atom types 
{SimulationParameters.xlo} {SimulationParameters.xhi} xlo xhi
{SimulationParameters.ylo} {SimulationParameters.yhi} ylo yhi
-0.25 0.25 zlo zhi

Masses

""" 
    for i in range(num_atom_types):
        to_write+=f"""{i+1} 1\n"""

    to_write+="""\nAtoms\n\n"""
    
    molecule_counter = 0
    type_counter     = 1

    for i in range(SimulationParameters.particles_per_monomer*SimulationParameters.numblocks):

        # increase the counting of molecules
        if i % SimulationParameters.particles_per_monomer == 0:
            molecule_counter+=1

        to_write+=f"""{i+1} {molecule_counter} {type_counter} {coords[i, 0]} {coords[i, 1]} 0.00\n"""

        # increase the counting of types
        type_counter+=1

        # reset the type counter
        if type_counter == (SimulationParameters.particles_per_monomer+1):
            type_counter = 1

    with open(SimulationParameters.configfile, 'w') as f:
        f.write(to_write)

def write_initconfig_triangles(SimulationParameters):

    """
    Function that writes the initial configuration
    for a system with N triangles
    """
    
    # place to save the coordinates (8 particles bc of hat)
    SimulationParameters.particles_per_monomer = 8
    coords = np.zeros((SimulationParameters.particles_per_monomer*SimulationParameters.numblocks, 2))

    # initialize displacement
    displace = [0, 0]

    generate_triangle_structure(i = 0, coords = coords, SimulationParameters= SimulationParameters)
    
    # write coordinates seed in closed config
    for i in range(1, SimulationParameters.numblocks):
        generate_triangle_structure(i = i*SimulationParameters.particles_per_monomer, coords = coords, SimulationParameters= SimulationParameters)
        coords[i*SimulationParameters.particles_per_monomer:i*SimulationParameters.particles_per_monomer+SimulationParameters.particles_per_monomer,0]+=i*2
    
    # recenter the polymer
    xcm = np.mean(coords[:, 0])
    coords[:, 0] -= xcm

    # update the initial boundaries of the system to fit the polymer (however many particles there are)
    if SimulationParameters.xlo is None:
        SimulationParameters.xlo = np.min(coords[:, 0]) - 1.2*SimulationParameters.bbscale
        SimulationParameters.xhi = np.max(coords[:, 0]) + 1.2*SimulationParameters.bbscale

    num_atom_types = SimulationParameters.particles_per_monomer+1

    to_write=f"""
# configuration file for the initial conditions of simulations with chevrons

{SimulationParameters.particles_per_monomer*SimulationParameters.numblocks} atoms
{num_atom_types} atom types 
{SimulationParameters.xlo} {SimulationParameters.xhi} xlo xhi
{SimulationParameters.ylo} {SimulationParameters.yhi} ylo yhi
-0.25 0.25 zlo zhi

Masses

""" 
    for i in range(num_atom_types):
        to_write+=f"""{i+1} 1\n"""

    to_write+="""\nAtoms\n\n"""
    
    molecule_counter = 0
    type_counter     = 1

    for i in range(SimulationParameters.particles_per_monomer*SimulationParameters.numblocks):

        # increase the counting of molecules
        if i % SimulationParameters.particles_per_monomer == 0:
            molecule_counter+=1

        to_write+=f"""{i+1} {molecule_counter} {type_counter} {coords[i, 0]} {coords[i, 1]} 0.00\n"""

        # increase the counting of types
        type_counter+=1

        # reset the type counter
        if type_counter == (SimulationParameters.particles_per_monomer+1):
            type_counter = 1

    with open(SimulationParameters.configfile, 'w') as f:
        f.write(to_write)


def write_python_function(SimulationParameters):

    # extend conservative cutoff to be able to compute the neighbour list
    SimulationParameters.conservative_cutoff_top = SimulationParameters.cutoff_top + 0.5

    to_write=f"""

def max_inter_mol_46(lmpptr):

    import math
    import numpy as np
    from lammps import lammps
    lmp = lammps(ptr=lmpptr)

    # pull local pair data
    dist = lmp.extract_compute("pl", 2, 1)       # local vector (distances)
    nrow = lmp.extract_compute("pl", 2, 0)       # number of local entries
    ids  = lmp.extract_compute("ids", 2, 2)      # local array: patom1, patom2, ptype1, ptype2
    step = lmp.get_thermo('step')

    for k in range(nrow):
        i  = int(ids[k][0]);  j = int(ids[k][1])
        ti = int(ids[k][2]);  tj = int(ids[k][3])

        # if particles are of the desired type, and not of the same molecule, and we are considering consecutive molecules
        if ( (ti == 4 and tj == 6) or (ti == 6 and tj == 4) ):
            if dist[k] > {SimulationParameters.cutoff_top}:
                f = open('record_breakage.dat', 'w')
                f.writelines(str(step) + ' ' + str(dist[k]) + ' ' +str(i) + ' ' + str(j))
                f.close()
                # this means the bond has broken
                return 1

    # return the haltflag
    return 0
"""

    with open(SimulationParameters.pythonfile, 'w') as f:
        f.write(to_write)