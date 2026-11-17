import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
#3 atom types? for epsC, epsT and eps for the rest

def periodic_boundary_conditions(x,L):                
        return x-np.round(x/L)*L

def hexagonal_positions(grid): #transform positions on grid into positions in the xy plane
    positions = []
    x_positions = []
    y_positions = []
    n = len(grid)

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                # Calculate x, y coordinates based on hexagonal lattice - deltax = L/2, deltay = L/2 * sqrt(3)/2
                x = j
                y = i*np.sqrt(3)/2 #squish the y axis so that its sqrt(3):1
                if i%2 == 1: #hexagonal lattice - shift in every even row of the grid to the left fro half a grid lengthscale size 
                    x -= 1/2 #shift x to the left to get hexagonal lattice
                x_positions.append(x)
                y_positions.append(y)
                positions.append((x, y))

    return positions,x_positions,y_positions

    """
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0]
    """

def grid_position_to_xy(grid_pos):
    i,j = grid_pos

    x = j
    y = i*np.sqrt(3)/2 #squish the y axis so that its sqrt(3):1
    if i%2 == 1: #hexagonal lattice - shift in every even row of the grid to the left fro half a grid lengthscale size 
        x -= 1/2 

    return (x,y)


# Example usage:
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0]
]
positions,xs,ys = hexagonal_positions(grid)
#print("Positions of inputs of 1 in the hexagonal lattice with center in the middle:")
#print(positions)

def count_ones(grid): #counts all the 1s in the grid and prints out mapping from (i,j): count as well as total numbers of 1s in grid
    counts = {}
    rows = len(grid)
    cols = len(grid[0])

    count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                count +=1 
                counts[(i, j)] = count

    return counts, count

    """
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
    """

# Example grid
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0]
    
]

# Get the counts dictionary
counts_dict,count = count_ones(grid)

# Print the result
#print(counts_dict)

def transform_output(output, counts):
    transformed_list = []

    for pair, stiffness_type in output.items():
        pos1, pos2 = pair.split("<->")
        
        # Remove any non-digit characters before converting to integers
        pos1 = tuple(int(''.join(filter(str.isdigit, coord))) for coord in pos1.strip("()").split(","))
        pos2 = tuple(int(''.join(filter(str.isdigit, coord))) for coord in pos2.strip("()").split(","))
        
        # Get the counts for pos1 and pos2 from the dictionary
        count1 = counts.get(pos1, None)
        count2 = counts.get(pos2, None)

        # Check if counts exist and count is '1'
        if count1 is not None and count2 is not None and stiffness_type>0:
            transformed_list.append((count1, count2, stiffness_type))

    return transformed_list

# Example dictionary inputs
counts = {(2, 3): 1, (3, 3): 2, (3, 4): 3, (4, 3): 4, (4, 4): 5, (4, 5): 6, (5, 1): 7, (5, 2): 8, (5, 3): 9, (5, 4): 10, (5, 5): 11, (6, 2): 12, (6, 3): 13, (6, 4): 14, (6, 5): 15, (6, 6): 16, (7, 2): 17, (7, 3): 18, (7, 4): 19, (7, 5): 20, (7, 6): 21}
output = {
    "(2, 3) <-> (3, 3)": 1,
    "(3, 3) <-> (3, 4)": 0,  # Note: count is 0
    "(3, 4) <-> (4, 3)": 1
}

# Transform the output
transformed_list = transform_output(output, counts)

# Print the result
#for pair in transformed_list:
 #   print(pair)
    
    
def assign_types(grid, molecule_num):
    rows = len(grid)
    cols = len(grid[0])
    assigned_values = np.array([molecule_num]*(rows*cols))
    #assigned_values = np.ones(rows*cols)

    return assigned_values

# Example grid
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0]
]

# Assign values
assigned_values = assign_types(grid,1)

# Print the result
#print(assigned_values)

def apply_periodic_boundary_conditions(positions, box_width, box_height):
    bounded_positions = []
    for x, y in positions:
        bounded_x = periodic_boundary_conditions(x,box_height)
        bounded_y = periodic_boundary_conditions(y,box_height)
        bounded_positions.append((bounded_x, bounded_y))
    return bounded_positions

"""
def place_objects(initial_positions, box_width, box_height, number_of_monomers):
    m = len(initial_positions)
    n = len(initial_positions[1])
    obj_positions = np.zeros((number_of_monomers,m,n))
    # Generate random translation and rotation for object 1
    tx1 = random.uniform(-box_width / 2, box_width / 2)
    ty1 = random.uniform(-box_height / 2, box_height / 2)
    angle1 = random.uniform(0, 360)

    # Apply translation and rotation to each point of object 1
    obj1_positions = [(x + tx1, y + ty1) for x, y in initial_positions]
    obj1_positions = rotate_object(obj1_positions, angle1)
    obj1_positions = apply_periodic_boundary_conditions(obj1_positions, box_width, box_height)
    
    obj_positions[0] = obj1_positions
    #print(obj_positions)
    #print(obj1_positions)

    for i in np.arange(number_of_monomers-1):
        # Generate random translation and rotation for object 2
        txi = random.uniform(-box_width / 2 , box_width / 2)
        tyi = random.uniform(-box_height / 2, box_height / 2)
        anglei = random.uniform(0, 360)

        # Apply translation and rotation to each point of object 2
        obji_positions = [(x + txi, y + tyi) for x, y in initial_positions]
        obji_positions = rotate_object(obji_positions, anglei)
        obji_positions = apply_periodic_boundary_conditions(obji_positions, box_width, box_height)
        
        obj_positions[i+1] = obji_positions 

        # Check if objects overlap
        overlap = 0
        for j in np.arange(i+1):  
            curr_obj = obj_positions[j]
            if check_overlap(obji_positions, curr_obj, box_width, box_height):
                overlap+=1
                    
        if overlap==0:
            continue
        else:
            i=i-1   #repeat till no overlap  
        
    #print(len(obj_positions))
    #print(len(obj_positions[0]))
    #print(obj_positions.shape)
    return obj_positions
"""
def place_objects(initial_positions, box_width, box_height, number_of_monomers,displacement = 10):
    m = len(initial_positions)
    n = len(initial_positions[0])
    obj_positions = np.zeros((number_of_monomers, m, n))
    
    for i in range(number_of_monomers):
        overlap = True
        inside = False
        while overlap or not inside:
            # Generate random translation and rotation
            tx = random.uniform(-box_width / 2, box_width / 2)
            ty = random.uniform(-box_height / 2, box_height / 2)
            angle = random.uniform(0, 360)

            # Apply translation and rotation to initial positions
            obj_positions[i] = [(x + tx, y + ty) for x, y in initial_positions]
            obj_positions[i] = rotate_object(obj_positions[i], angle)
            #obj_positions[i] = apply_periodic_boundary_conditions(obj_positions[i], box_width, box_height)

            # Check for overlap with previously placed objects
            overlap = False
            for j in range(i):
                if check_overlap(obj_positions[i], obj_positions[j], box_width, box_height):
                    overlap = True
                    break

            #FOR NON PERIODIC BOUNDARY CONDITIONDS ADDING EXTRA CONDITION - NOT PLACE MONOMERS AT THE EDGES
            x_all = [t[0] for t in obj_positions[i]]
            y_all = [t[1] for t in obj_positions[i]]

            all_inside_x = all(-box_width / 2 + 0.5 <= x <=  box_width / 2 - 0.5 for x in x_all)
            all_inside_y = all(-box_height / 2 + 0.5 <= y <=  box_height / 2 - 0.5 for y in y_all)

            inside = False
            if all_inside_x == True and all_inside_y == True:
                inside = True

    return obj_positions

def distance_with_periodic_boundary(pos1, pos2, box_width, box_height):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])

    # Apply periodic boundary conditions
    dx = periodic_boundary_conditions(dx,box_height)
    dy = periodic_boundary_conditions(dy,box_height)

    # Calculate Euclidean distance
    distance = math.sqrt(dx**2 + dy**2)
    return distance

def check_overlap(obj1_positions, obj2_positions, box_width, box_height, min_distance=2.5):
    for pos1 in obj1_positions:
        for pos2 in obj2_positions:
            distance = distance_with_periodic_boundary(pos1, pos2, box_width, box_height)
            if distance < min_distance:
                return True
    return False

def rotate_object(positions, angle):
    rotated_positions = []
    for x, y in positions:
        # Translate to origin, rotate, and translate back
        new_x = (x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle)))
        new_y = (x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)))
        rotated_positions.append((new_x, new_y))
    return rotated_positions

def plot_objects(obj1_positions, obj2_positions, box_width, box_height):
    plt.figure(figsize=(8, 6))
    plt.xlim(-box_width / 2, box_width / 2)
    plt.ylim(-box_height / 2, box_height / 2)

    # Plot object 1 in red
    for x, y in obj1_positions:
        plt.scatter(x, y, color='red')

    # Plot object 2 in blue
    for x, y in obj2_positions:
        plt.scatter(x, y, color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Two monomers in a box')
    #plt.grid(True)
    #plt.show()
    plt.savefig('initial_configuration')

# Example usage
"""
initial_positions = [(1, 1), (1, 2),(1,3),(1,4),(2,4),(2,3),(2,2),(3,3)]
box_width = 50
box_height = 50
rect_width = 0.5
rect_height = 0.5

obj1_positions, obj2_positions = place_objects(positions, box_width, box_height, 4)
print(obj1_positions)
print([x for x, _ in obj1_positions])
plot_objects(obj1_positions, obj2_positions, box_width, box_height)
"""
def write_individual_txt_file(gpath, individual, random_num = -1):

    g = open('%s/individual.txt'%(gpath), 'w')
    g.write('INDIVIDUAL\n')
    g.write(str(individual))
    g.write('\n SIDE CHOSEN(if number<0.5 its xmin and xmax, else ymin ans ymax) \n')
    """
    if random_num == -1:
        random_num = random.random()
        individual.type_rnd_number = random_num
    """
    g.write(str(random_num))
    g.close()

def write_lammps(individual,gpath, box_width, box_height, number_of_monomers,random_n):
    rows = len(individual)
    cols = len(individual[0])
    
    grid = np.zeros((rows,cols))
    dict_bonds = {}
    written_bonds = set()  # To keep track of written bonds
    for i, row in enumerate(individual):
        for j, point in enumerate(row):
            grid[i][j] = point['value'] #fill up grid
            if point['value'] == 1:
                for bond, value_bond_stiffness_type in point['bonds'].items():
                    # Sort the bond tuple to avoid duplicates
                    sorted_bond = tuple(sorted([(i, j), bond]))
                    if sorted_bond not in written_bonds:
                        dict_bonds[f'({i}, {j}) <-> ({bond[0]}, {bond[1]})'] =  value_bond_stiffness_type
                        written_bonds.add(sorted_bond)
    #print('GRID IS:')
    #print(grid)
    
    positions,X_pos,Y_pos = hexagonal_positions(grid)
    #print('POSITIONS:')
    #print(positions)
    #print('POSITIONS x')
    #print(X_pos)
    counts_dict,count = count_ones(grid)
    transformed_list = transform_output(dict_bonds, counts_dict)
    num_of_bonds = len(transformed_list)
    #assigned_types = assign_types(grid,1) #all are the same type
    #center_of_mass = calculate_center_of_mass(positions, box_width)
    #print("POSITIONS ARE:")
    #print(positions)

    types_little_guys = [5,3,7,6,2,4,5,1]
    #THIS IS FOR ADDED BOND TO FIX LG IN CHEVRON SHAPE
    #types_little_guys = [5,2,3,1,4,7,6,5]

    assigned_types = types_little_guys #assign_types(grid,1) #all are the same type
    #assigned_types,random_num = find_particle_types(X_pos,Y_pos, positions,neighbor_distance=1.1,random_n=random_n) #type 1 edge type 2 bulk
    #assigned_types = label_particles(grid=grid, n=random_n)
    obji_positions = place_objects(positions, box_width, box_height, number_of_monomers)
    
    #mols = np.ones(2*count) #what molecule it is 
    #mols[count:] = 2
    #L = dyn_region_L #further think about this
    
    f = open('%s/configuration.dat'%(gpath), 'w')
    f.write("# Configuration file for the initial conditions of the chevron-like molecules simulations\n\n")
    f.write("%d atoms\n" %(number_of_monomers*count))
    f.write("%d bonds\n" %(number_of_monomers*num_of_bonds))
    #f.write("%d atom types\n" %number_of_monomers)
    f.write("7 atom types\n")
    f.write("1 bond types\n")
    f.write("-%.1f %.1f xlo xhi\n"%(box_width/2.0,box_width/2.0))
    f.write("-%.1f %.1f ylo yhi\n"%(box_height/2.0,box_height/2.0))
    f.write("-0.25 0.25 zlo zhi\n\n")
    f.write("Masses\n\n")
    """
    for i in np.arange(1,number_of_monomers):
        f.write("%d 1\n"%i)
    f.write("%d 1\n\n"%(number_of_monomers))
    """
    f.write("1 1\n")
    f.write("2 1\n")
    f.write("3 1\n")
    f.write("4 1\n")
    f.write("5 1\n")
    f.write("6 1\n")
    f.write("7 1\n\n")
    f.write("Atoms\n\n")
    for j in np.arange(number_of_monomers):
        Xi = [x for x, _ in obji_positions[j]]
        Yi = [y for _ , y in obji_positions[j]]
        for i in range(len(positions)): 
            f.write("%d %d %d %f %f 0.0\n"%(i+1+j*count,j+1,assigned_types[i],Xi[i],Yi[i]))
    f.write('\n')
    if num_of_bonds>0:
        f.write("Bonds\n\n")
        for j in range(number_of_monomers):
            for i in range(num_of_bonds):
                f.write("%d %d %d %d \n"%(j*num_of_bonds+i+1,transformed_list[i][2],transformed_list[i][0] + j*count,transformed_list[i][1]+ j*count))
        f.close()
        
def write_lammps_one_monomer(individual,gpath, box_width, box_height, random_n):
    rows = len(individual)
    cols = len(individual[0])
    
    grid = np.zeros((rows,cols))
    dict_bonds = {}
    written_bonds = set()  # To keep track of written bonds
    for i, row in enumerate(individual):
        for j, point in enumerate(row):
            grid[i][j] = point['value'] #fill up grid
            if point['value'] == 1:
                for bond, value_bond_stiffness_type in point['bonds'].items():
                    # Sort the bond tuple to avoid duplicates
                    sorted_bond = tuple(sorted([(i, j), bond]))
                    if sorted_bond not in written_bonds:
                        dict_bonds[f'({i}, {j}) <-> ({bond[0]}, {bond[1]})'] =  value_bond_stiffness_type
                        written_bonds.add(sorted_bond)
    #print('GRID IS:')
    #print(grid)
    
    positions,X_pos,Y_pos = hexagonal_positions(grid)
    #print('POSITIONS:')
    #print(positions)
    #print('POSITIONS x')
    #print(X_pos)
    counts_dict,count = count_ones(grid)
    transformed_list = transform_output(dict_bonds, counts_dict)
    num_of_bonds = len(transformed_list)
    #assigned_types = assign_types(grid,1) #all are the same type
    #center_of_mass = calculate_center_of_mass(positions, box_width)
    #print("POSITIONS ARE:")
    #print(positions)

    types_little_guys = [5,3,7,6,2,4,5,1]
    #THIS IS FOR ADDED BOND TO FIX LG IN CHEVRON SHAPE
    #types_little_guys = [5,2,3,1,4,7,6,5]

    assigned_types = types_little_guys #assign_types(grid,1) #all are the same type
    #assigned_types,random_num = find_particle_types(X_pos,Y_pos, positions,neighbor_distance=1.1,random_n=random_n) #type 1 edge type 2 bulk

    #mols = np.ones(2*count) #what molecule it is 
    #mols[count:] = 2
    L = box_height #further think about this
    
    f = open('%s/monomer.dat'%(gpath), 'w')
    f.write("# Configuration file for the initial conditions of the chevron-like molecules simulations\n\n")
    f.write("%d atoms\n" %(count))
    f.write("%d bonds\n" %(num_of_bonds))
    """
    #f.write("%d atom types\n" %number_of_monomers)
    #f.write("3 atom types\n")
    #f.write("1 bond types\n")
    #f.write("-%.1f %.1f xlo xhi\n"%(L/2.0,L/2.0))
    #f.write("-%.1f %.1f ylo yhi\n"%(L/2.0,L/2.0))
    #f.write("-0.25 0.25 zlo zhi\n\n")
    #f.write("Masses\n\n")

    for i in np.arange(1,number_of_monomers):
        f.write("%d 1\n"%i)
    f.write("%d 1\n\n"%(number_of_monomers))
    
    f.write("1 1\n")
    f.write("2 1\n")
    f.write("3 1\n\n")
    """
    f.write("Coords\n\n")
    #for j in np.arange(number_of_monomers):
    for i in range(len(positions)): 
        f.write("%d %f %f 0.0\n"%(i+1,X_pos[i],Y_pos[i]))
    f.write('\n')
    f.write("Types\n\n")
    for i in range(len(positions)): 
        f.write("%d %d \n"%(i+1,assigned_types[i]))
    f.write('\n')
    if num_of_bonds>0:
        f.write("Bonds\n\n")
        for i in range(num_of_bonds):
            f.write("%d %d %d %d \n"%(i+1,transformed_list[i][2],transformed_list[i][0],transformed_list[i][1]))
    f.close()
    
    
#functions to get com w respect to the pbc to detect particles on the edge
def apply_pbc(x, y,L):
    """Apply periodic boundary conditions to coordinates."""
    xhi = L/2
    xlo = -L/2
    yhi = L/2
    ylo = -L/2
    
    if x < xlo:
        x += L
    elif x >= xhi:
        x -= L

    if y < ylo:
        y += L
    elif y >= yhi:
        y -= L
    
    return x, y
    

def calculate_center_of_mass(ind_positions, L):
        
    # Extract positions of the molecule
    xs, ys = zip(*ind_positions)
    xs = list(xs)
    ys = list(ys)
    
    num_atoms=len(xs)
    
    borderx = False #check if breaks on x border
    for j,x1 in enumerate(xs):
        for x2 in xs[j+1:]:
            dist = np.abs(x1-x2)
            if dist>L/2:
                borderx = True
                
    bordery = False #check if breaks on y border
    for i,y1 in enumerate(ys):
        for y2 in ys[i+1:]:
            dist = np.abs(y1-y2)
            if dist>L/2:
                bordery = True       
    
    xs_list = list(xs)
    if borderx: #if breaks on x border shift the cut off ones by L either to left or right
        if sum(xs_list) >= 0:
            for k1,x in enumerate(xs_list):
                if x<0:
                    xs_list[k1] += L
        elif sum(xs_list) < 0:
            for k2,x in enumerate(xs_list):
                if x>0:
                    xs_list[k2] -= L
    xs = tuple(xs_list)
            
    ys_list = list(ys)
    if bordery: #if breaks on x border shift the cut off ones by L either to left or right
        if sum(ys_list) >= 0:
            for k3,y in enumerate(ys_list):
                if y<0:
                    ys_list[k3] += L
        elif sum(ys_list) < 0:
            for k4,y in enumerate(ys_list):
                if y>0:
                    ys_list[k4] -= L
    ys = tuple(ys_list)  
    
    # Calculate center of mass assuming virtual extended box
    com_x = sum(xs) / num_atoms
    com_y = sum(ys) / num_atoms
    
    # Apply periodic boundary conditions to COM
    com_x, com_y = apply_pbc(com_x, com_y, L )
    com = [com_x,com_y]
    
    return com

#finds left and right neighbours in the direction position - com and adds particle type 1 to particles on the edge and praticel type 2 to the particles inside the bulk
"""
def find_particle_types(positions, center_of_mass, neighbor_distance=1.1):
    # Convert positions to numpy array for easier manipulation
    positions = np.array(positions)
    center_of_mass = np.array(center_of_mass)

    # Create a KDTree for efficient neighbor searching
    tree = KDTree(positions)

    # Initialize type list
    type_list = []

    for pos in positions:
        # Compute the direction vector from the position to the center of mass
        direction = center_of_mass - pos
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

        # Define the right and left neighbor positions
        right_neighbor_pos = pos + direction * neighbor_distance
        left_neighbor_pos = pos - direction * neighbor_distance

        # Check if right and left neighbors exist within a small tolerance
        right_neighbors = tree.query_ball_point(right_neighbor_pos, neighbor_distance / 2)
        left_neighbors = tree.query_ball_point(left_neighbor_pos, neighbor_distance / 2)

        # Remove the current particle from neighbor lists if it's included
        right_neighbors = [i for i in right_neighbors if not np.array_equal(positions[i], pos)]
        left_neighbors = [i for i in left_neighbors if not np.array_equal(positions[i], pos)]

        # Determine the type based on the number of neighbors
        if len(right_neighbors) > 0 and len(left_neighbors) > 0:
            type_list.append(2)
        else:
            type_list.append(1)

    return type_list
"""

def find_particle_types(x_positions, y_positions,positions,neighbor_distance=1.0,random_n = -1):
    # Convert positions to numpy array for easier manipulation
    positions = np.array(positions)
    #print(positions)
    x_max = max(x_positions)
    x_min = min(x_positions)
    y_max = max(y_positions)
    y_min = min(y_positions)

    type_4 = [(1,1),(2,2)]
    type_5 = [(2,1),(1,2)]

    type_4_positions =[grid_position_to_xy(type_grid) for type_grid in type_4]
    type_5_positions = [grid_position_to_xy(type_grid) for type_grid in type_5]
   
    # Create a KDTree for efficient neighbor searching
    tree = KDTree(positions)

    # Define the relative positions of the neighbors in a hexagonal lattice
    relative_neighbors = np.array([
        [1, 0],
        [-1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
        [-0.5, -np.sqrt(3)/2]
    ]) * neighbor_distance

    # Initialize type list
    type_list = []

    happens_x_min = 0
    happens_x_max = 0
    happens_y_min = 0
    happens_y_max = 0
    if random_n == -1:
        random_n = random.random() #maybe make it just one now it has 2 for the full lattice
    
    for pos in positions:
        """
        neighbors_count = 0
        for rel_pos in relative_neighbors:
            neighbor_pos = pos + rel_pos
            # Find neighbors within a small distance from the expected neighbor position
            neighbors = tree.query_ball_point(neighbor_pos, neighbor_distance * 0.1)
            if neighbors:
                neighbors_count += 1

        # Determine the type based on the number of neighbors
        if neighbors_count < 6:
        """
        if random_n<0.5:
            if tuple(pos) in type_4_positions:
                type_list.append(4)
            elif tuple(pos) in type_5_positions:
                type_list.append(5) 
            elif pos[0] == x_max: #and happens_x_max == 0:
                happens_x_max +=1
                type_list.append(1)
            elif pos[0]== x_min:# and happens_x_min == 0:
                happens_x_min +=1
                type_list.append(2)
            else:
                type_list.append(3)
            
        elif random_n>=0.5:
            if tuple(pos) in type_4_positions:
                type_list.append(4)
            elif tuple(pos) in type_5_positions:
                type_list.append(5) 
            elif pos[1] ==y_max:# and happens_y_max == 0:
                happens_y_max +=1
                type_list.append(1)
            elif pos[1] ==y_min:# and happens_y_min == 0:
                happens_y_min +=1
                type_list.append(2)
            else:
                type_list.append(3)
            
        #else:
        #    type_list.append(3)

    return type_list,random_n


def label_particles(grid, n):
    types = []
    ones_positions = [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 1]

    if n < 0.5:
        # Row-based labeling
        row_groups = {}
        for r, c in ones_positions:
            row_groups.setdefault(r, []).append(c)

        row_extremes = {r: (min(cols), max(cols)) for r, cols in row_groups.items()}

        for r, c in ones_positions:
            leftmost, rightmost = row_extremes[r]
            if c == rightmost:
                types.append(1)
            elif c == leftmost:
                types.append(2)
            else:
                types.append(3)

    else:
        # Column-based labeling
        col_groups = {}
        for r, c in ones_positions:
            col_groups.setdefault(c, []).append(r)

        col_extremes = {c: (min(rows), max(rows)) for c, rows in col_groups.items()}

        for r, c in ones_positions:
            topmost, bottommost = col_extremes[c]
            if r == topmost:
                types.append(1)
            elif r == bottommost:
                types.append(2)
            else:
                types.append(3)

    return types


def read_individual_txt_file(file_path):
    
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Reads all lines into a list
        
    individual = lines[1]
    random_number_for_type = lines[3]

    individual = ast.literal_eval(individual)
    random_number_for_type  = ast.literal_eval(random_number_for_type)
    
    return individual, random_number_for_type
