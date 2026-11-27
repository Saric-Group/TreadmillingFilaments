import numpy as np 

def generate_buildingblock_structure(i, coords, SimulationParameters, rails = False, open = False, intermediate = False):

    if open:
        coords[i, 0] = 0
        coords[i, 1] = 0
        #coords[i, 2] = 0

        coords[i+1, 0] = SimulationParameters.x
        coords[i+1, 1] = 0
        #coords[i+1, 2] = 0

        coords[i+2, 0] = 1.0
        coords[i+2, 1] = 0.0
        #coords[i+2, 2] = 0.0

        coords[i+3, 0] = (SimulationParameters.x)*np.cos((SimulationParameters.angle_open+180)*np.pi/180) + SimulationParameters.x
        coords[i+3, 1] = (SimulationParameters.x)*np.sin((SimulationParameters.angle_open+180)*np.pi/180) + 1.0
        #coords[i+3, 2] = 0

        coords[i+4, 0] = SimulationParameters.x
        coords[i+4, 1] = 1.0
        #coords[i+4, 2] = 0

        coords[i+5, 0] = (1-SimulationParameters.x)*np.cos(SimulationParameters.angle_open*np.pi/180.0) + SimulationParameters.x
        coords[i+5, 1] = (1-SimulationParameters.x)*np.sin(SimulationParameters.angle_open*np.pi/180.0) + 1.0
        #coords[i+5, 2] = 0

        coords[i+6, 0] = SimulationParameters.x
        coords[i+6, 1] = 1-SimulationParameters.x
        #coords[i+SimulationParameters.particles_per_monomer, 2] = 0

        return

        """
        # OUTDATED!
        if rails:
            coords[i+6, 0] = SimulationParameters.x
            coords[i+6, 1] = 0
            coords[i+6, 2] = 1
            
            coords[i+7, 0] = SimulationParameters.x
            coords[i+7, 1] = 0
            coords[i+7, 2] = -1
        """

    if intermediate:
        coords[i, 0] = 0
        coords[i, 1] = 0
        #coords[i, 2] = 0

        coords[i+1, 0] = SimulationParameters.x
        coords[i+1, 1] = 0
        #coords[i+1, 2] = 0

        coords[i+2, 0] = 1.0
        coords[i+2, 1] = 0.0
        #coords[i+2, 2] = 0.0

        coords[i+3, 0] = (SimulationParameters.x)*np.cos((SimulationParameters.angle_open*0.5+180)*np.pi/180) + SimulationParameters.x
        coords[i+3, 1] = (SimulationParameters.x)*np.sin((SimulationParameters.angle_open*0.5+180)*np.pi/180) + 1.0
        #coords[i+3, 2] = 0

        coords[i+4, 0] = SimulationParameters.x
        coords[i+4, 1] = 1.0
        #coords[i+4, 2] = 0

        coords[i+5, 0] = (1-SimulationParameters.x)*np.cos(SimulationParameters.angle_open*0.5*np.pi/180.0) + SimulationParameters.x
        coords[i+5, 1] = (1-SimulationParameters.x)*np.sin(SimulationParameters.angle_open*0.5*np.pi/180.0) + 1.0
        #coords[i+5, 2] = 0

        coords[i+6, 0] = SimulationParameters.x
        coords[i+6, 1] = 1-SimulationParameters.x
        #coords[i+SimulationParameters.particles_per_monomer, 2] = 0

        return
    else:
        coords[i, 0] = 0
        coords[i, 1] = 0
        #coords[i, 2] = 0

        coords[i+1, 0] = SimulationParameters.x
        coords[i+1, 1] = 0
        #coords[i+1, 2] = 0

        coords[i+2, 0] = 1.0
        coords[i+2, 1] = 0.0
        #coords[i+2, 2] = 0.0

        coords[i+3, 0] = coords[i, 0]
        coords[i+3, 1] = 1.0
        #coords[i+3, 2] = 0.0

        coords[i+4, 0] = SimulationParameters.x
        coords[i+4, 1] = 1.0
        #coords[i+4, 2] = 0.0

        coords[i+5, 0] = 1.0
        coords[i+5, 1] = 1.0
        #coords[i+5, 2] = 0.0

        coords[i+6, 0] = SimulationParameters.x
        coords[i+6, 1] = 1-SimulationParameters.x
        #coords[i+SimulationParameters.particles_per_monomer, 2] = 0

        return
        
        """
        # OUTDATED
        if rails:
            coords[i+6, 0] = SimulationParameters.x
            coords[i+6, 1] = 0
            coords[i+6, 2] = 1
            
            coords[i+7, 0] = SimulationParameters.x
            coords[i+7, 1] = 0
            coords[i+7, 2] = -1
        """

def generate_chevron_structure(i, coords, SimulationParameters):

    # PLEASE NOTE THAT THIS CHEVRON HAS A 'HAT'

    coords[i, 0] = 0
    coords[i, 1] = 0
    #coords[i, 2] = 0

    coords[i+1, 0] = 1
    coords[i+1, 1] = 0
    #coords[i+1, 2] = 0

    coords[i+2, 0] = -0.5
    coords[i+2, 1] = np.sqrt(3)/2
    #coords[i+2, 2] = 0.0

    coords[i+3, 0] = 0.5
    coords[i+3, 1] = np.sqrt(3)/2
    #coords[i+3, 2] = 0

    coords[i+4, 0] = 0
    coords[i+4, 1] = np.sqrt(3)
    #coords[i+4, 2] = 0

    coords[i+5, 0] = 1
    coords[i+5, 1] = np.sqrt(3)
    #coords[i+5, 2] = 0

    coords[i+6, 0] = 0.5
    coords[i+6, 1] = 3*np.sqrt(3)/2
    #coords[i+5, 2] = 0

    coords[i+7, 0] = 0.5
    coords[i+7, 1] = -np.sqrt(3)/2
    #coords[i+5, 2] = 0

    return

def generate_triangle_structure(i, coords, SimulationParameters):

    # PLEASE NOTE THAT THIS CHEVRON HAS A 'HAT'

    coords[i, 0] = 0
    coords[i, 1] = 0
    #coords[i, 2] = 0

    coords[i+1, 0] = 1
    coords[i+1, 1] = 0
    #coords[i+1, 2] = 0

    coords[i+2, 0] = 2
    coords[i+2, 1] = 0
    #coords[i+2, 2] = 0.0

    coords[i+3, 0] = 0.5
    coords[i+3, 1] = np.sqrt(3)/2
    #coords[i+3, 2] = 0

    coords[i+4, 0] = 1.5
    coords[i+4, 1] = np.sqrt(3)
    #coords[i+4, 2] = 0

    coords[i+5, 0] = 1
    coords[i+5, 1] = np.sqrt(3)
    #coords[i+5, 2] = 0

    coords[i+6, 0] = -0.5
    coords[i+6, 1] = np.sqrt(3)/2
    #coords[i+5, 2] = 0

    coords[i+7, 0] = 2.5
    coords[i+7, 1] = np.sqrt(3)/2
    #coords[i+5, 2] = 0

    return

def write_monomer_template(SimulationParameters):

    """
    Function that writes the initial configuration
    for a system with N building blocks, all in the
    closed configuration. The goal is to study the
    stability of the polymer if the system has or
    not a conformation switch.

    Please note that the monomer is initialized the open state!
    """
    
    # place to save the coordinates
    coords = np.zeros((SimulationParameters.particles_per_monomer, 2))

    # initialize displacement
    displace = [0, 0]

    # write coordinates seed in closed config
    generate_buildingblock_structure(i = 0, coords = coords, SimulationParameters = SimulationParameters, open = True)
    coords[0:SimulationParameters.particles_per_monomer]*=SimulationParameters.bbscale

    xcm = np.mean(coords[:, 0])
    coords[:, 0] -= xcm

    to_write=f"""
# configuration file for the initial conditions of the Model 0 molecules simulations

{SimulationParameters.particles_per_monomer} atoms
{SimulationParameters.bonds_per_monomer} bonds
{SimulationParameters.angles_per_monomer} angles

Coords

"""
    for i in range(SimulationParameters.particles_per_monomer):

        to_write+=f"""{i+1} {coords[i, 0]} {coords[i, 1]} 0.00\n"""

    to_write+="""\nTypes\n\n"""

    molecule_counter = 0
    type_counter     = 1

    for i in range(SimulationParameters.particles_per_monomer):

        # increase the counting of molecules
        if i % SimulationParameters.particles_per_monomer == 0:
            molecule_counter+=1

        to_write+=f"""{i+1} {type_counter}\n"""

        # increase the counting of types
        type_counter+=1

        # reset the type counter
        if type_counter == (SimulationParameters.particles_per_monomer+1):
            type_counter = 1

    to_write+="""\nBonds\n\n"""

    to_write+="""1 1 1 2\n"""
    to_write+="""2 2 2 3\n"""
    to_write+="""3 1 4 5\n"""
    to_write+="""4 2 5 6\n"""
    to_write+="""5 3 2 5\n"""
    to_write+="""6 2 2 7\n"""

    to_write+="""\nAngles\n\n"""

    if SimulationParameters.coupled_interfaces:
        to_write+=f"""1 1 1 2 3\n"""
        to_write+=f"""2 1 4 5 6\n"""
        to_write+=f"""3 2 1 2 5\n"""
        to_write+=f"""4 2 3 2 5\n"""
        to_write+=f"""5 3 2 5 6\n"""
        to_write+=f"""6 4 2 5 4\n"""
        to_write+=f"""7 2 1 2 7\n"""
        to_write+=f"""8 2 3 2 7\n"""
    else:
        to_write+=f"""1 1 1 2 3\n"""
        to_write+=f"""2 2 1 2 5\n"""
        to_write+=f"""3 2 3 2 5\n"""
        to_write+=f"""4 3 2 5 6\n"""
        to_write+=f"""5 4 2 5 4\n"""
        to_write+=f"""6 2 1 2 7\n"""
        to_write+=f"""7 2 3 2 7\n"""

    with open(SimulationParameters.monomerfile, 'w') as f:
        f.write(to_write)

def generate_colloidal_littleguy(closed = True):

    # IN DEVELOPMENT!

    x0, y0, ang, id0 = 0, 0, 0, 0
    X, Y, I = np.empty(8), np.empty(8), np.empty(8)

    if closed:
        das = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, np.pi/2, 3*np.pi/2]
        rs = [0, 1, 1, 1, 1, 1, np.sqrt(3), np.sqrt(3)]
        Bs = id0-1+np.array([[1,2],[1,3],[1,5],[1,6],[2,3],[3,4],[4,5],[5,6],[2,7],[3,7],[5,8],[6,8],[1,4]])
    else:
        das = [0, np.pi/2, 5*np.pi/6, np.pi, 7*np.pi/6, 3*np.pi/2, 2*np.pi/3, 4*np.pi/3]
        rs = [0, 1, 1, np.sqrt(3), 1, 1, np.sqrt(3), np.sqrt(3)]
        Bs = id0-1+np.array([[1,2],[1,3],[1,5],[1,6],[2,3],[3,4],[4,5],[5,6],[2,7],[3,7],[5,8],[6,8]])
    
    for i in range(8):
        I[i] = i+id0
        X[i] = x0+rs[i]*np.cos(ang+das[i])
        Y[i] = y0+rs[i]*np.sin(ang+das[i])


    T = np.array([9,8,5,6,7,10,11,11])
    if kill == 'T' or kill == 't' or kill == 'tip' or kill == 'Tip' or kill == 'TIP':
        if form == 'C' or form == 'c' or form == 'closed' or form ==  'Closed' or form == 'CLOSED':#this is for the seed
            T = np.array([9,8,11,11,11,10,11,11])
        else:
            T = np.array([9,8,2,1,2,10,11,11])
    elif kill == 'B' or kill == 'b' or kill == 'back' or kill == 'Back' or kill == 'BACK':
        if form == 'C' or form == 'c' or form == 'closed' or form ==  'Closed' or form == 'CLOSED':#this is for the seed
            T = np.array([11,11,5,6,7,11,11,11])
        else:
            T = np.array([1,11,5,6,7,11,11,11])

    return X,Y,I,T,Bs

