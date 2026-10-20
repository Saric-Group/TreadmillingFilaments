def write_input_singlemonomer(SimulationParameters, add_run = False):

    """
    This function writes the LAMMPS input file for single monomer from model 0
    """

    # the actual input file
    to_write=f"""
# Simulation with filament using model 0 building blocks

units               lj
atom_style          molecular
dimension           2 
boundary            f f p
read_data           configuration.dat
group               moving molecule > 1  

# set reflections 
fix xwalls all wall/reflect xlo EDGE xhi EDGE
fix ywalls all wall/reflect ylo EDGE yhi EDGE

# designate bottom particles (they will move on a rail)
variable            bottom_group_atom_type atom (type==1)||(type==2)||(type==3) 
group               bottom dynamic all var bottom_group_atom_type every 1

# trap the building block at the bottom - 1D motion
variable force_trapping_bottom atom -100000*y
variable energy atom 50000*y*y
fix trap_bottom bottom addforce 0.0 v_force_trapping_bottom 0.0 every 1 energy v_energy

# ensure safe communication cutoff (to keep the shape of the building block correctly)
comm_modify cutoff {SimulationParameters.bond_1_rc + SimulationParameters.bond_3_rc + 0.5}

# .............
# bonds
bond_style          harmonic
bond_coeff          1 {SimulationParameters.bond_1_k} {SimulationParameters.bond_1_rc} 
bond_coeff          2 {SimulationParameters.bond_2_k} {SimulationParameters.bond_2_rc} 
bond_coeff          3 {SimulationParameters.bond_3_k} {SimulationParameters.bond_3_rc} 

# .............
# angles
angle_style          harmswitch
angle_coeff          1 {SimulationParameters.angle_1_k} 180 360 
angle_coeff          2 {SimulationParameters.angle_2_k} 90 360 
angle_coeff          3 {SimulationParameters.angle_3_k} {SimulationParameters.angle_3_eq} {SimulationParameters.angle_3_rc}
angle_coeff          4 {SimulationParameters.angle_4_k} {SimulationParameters.angle_4_eq} {SimulationParameters.angle_4_rc}

# .............
# interaction potentials          
pair_style          cosine/squared 2.0
pair_coeff          * * 0 0 1             

# ....... within building block
pair_coeff          4 7  10.0 {SimulationParameters.sigma_preventinversion} {SimulationParameters.sigma_preventinversion} wca

# ....... for polymerization
# bottom binding sites interact through volume exclusion
pair_coeff          1 3 {SimulationParameters.epsilon_bottom} {SimulationParameters.sigma} {SimulationParameters.sigma} wca
# top particles attract each other
pair_coeff          4 6 {SimulationParameters.epsilon_top} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca


# particles that are bonded do not interact with each other (except 4 and 7, which are 3rd neighbours)
special_bonds       lj 0.0 0.0 1.0

# dump trajectory
dump                1 all custom {SimulationParameters.dumpfreq} output.dump id mol type x y

# compute & dump bonds
compute             cBonds all property/local btype batom1 batom2
compute             cDistBonds all bond/local engpot dist
dump                2 all local {SimulationParameters.dumpfreq} bonds.dump c_cBonds[*] c_cDistBonds[*]

# compute & dump angles
compute             cAngles all property/local aatom1 aatom2 aatom3 atype
compute             ctheta all angle/local theta
dump                3 all local {SimulationParameters.dumpfreq} angles.dump c_cAngles[*] c_ctheta

# reduce simulation time 
neigh_modify check yes exclude type 1 1 exclude type 2 2 exclude type 3 3 exclude type 4 4 exclude type 5 5 exclude type 6 6 exclude type 7 7 

# timestep and running
thermo              {SimulationParameters.dumpfreq}
timestep            {SimulationParameters.dt}

# time integration
fix                 fLang moving langevin {SimulationParameters.temp} {SimulationParameters.temp} {SimulationParameters.damping} {SimulationParameters.seed}
fix                 fNVE moving nve
fix                 twodim all enforce2d

run {SimulationParameters.steps}

"""
    
    with open(SimulationParameters.inputfile, 'w') as f:
        f.write(to_write)

def write_input_polymerizingfilament(SimulationParameters, add_run = False):

    """
    This function writes the LAMMPS input file for a filament of building blocks.
    Each block corresponds to model 0 (angular version). 
    """

    # the actual input file
    to_write=f"""
# Simulation with filament using model 0 building blocks

units               lj
atom_style          molecular
dimension           2 
boundary            f f p
read_data           configuration.dat
molecule            monomer_template monomer.dat
group               moving molecule > 1
group               first_mol molecule 1

# set reflections 
fix xwalls all wall/reflect xlo EDGE xhi EDGE
fix ywalls all wall/reflect ylo EDGE yhi EDGE

# designate bottom particles (they will move on a rail)
variable            bottom_group_atom_type atom (type==1)||(type==2)||(type==3) 
group               bottom dynamic all var bottom_group_atom_type every 1

# trap the building block at the bottom - 1D motion
variable force_trapping_bottom atom -100000*y
variable energy atom 50000*y*y
fix trap_bottom bottom addforce 0.0 v_force_trapping_bottom 0.0 every 1 energy v_energy

# ensure safe communication cutoff (to keep the shape of the building block correctly)
comm_modify cutoff {SimulationParameters.bond_1_rc + SimulationParameters.bond_3_rc + 0.5}

# .............
# bonds
bond_style          harmonic
bond_coeff          1 {SimulationParameters.bond_1_k} {SimulationParameters.bond_1_rc} 
bond_coeff          2 {SimulationParameters.bond_2_k} {SimulationParameters.bond_2_rc} 
bond_coeff          3 {SimulationParameters.bond_3_k} {SimulationParameters.bond_3_rc} 

# .............
# angles
angle_style          harmswitch
angle_coeff          1 {SimulationParameters.angle_1_k} 180 360 
angle_coeff          2 {SimulationParameters.angle_2_k} 90 360 
angle_coeff          3 {SimulationParameters.angle_3_k} {SimulationParameters.angle_3_eq} {SimulationParameters.angle_3_rc}
angle_coeff          4 {SimulationParameters.angle_4_k} {SimulationParameters.angle_4_eq} {SimulationParameters.angle_4_rc}

# .............
# interaction potentials          
pair_style          cosine/squared 2.0
pair_coeff          * * 0 0 1             

# ....... within building block
pair_coeff          4 7  10.0 {SimulationParameters.sigma_preventinversion} {SimulationParameters.sigma_preventinversion} wca

# ....... for polymerization
# bottom binding sites interact through volume exclusion
pair_coeff          1 3 {SimulationParameters.epsilon_bottom} {SimulationParameters.sigma} {SimulationParameters.sigma} wca
# top particles attract each other
pair_coeff          4 6 {SimulationParameters.epsilon_top} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca

# particles that are bonded do not interact with each other (except 4 and 7, which are 3rd neighbours)
special_bonds       lj 0.0 0.0 1.0

# dump trajectory
dump                1 all custom {SimulationParameters.dumpfreq} output.dump id mol type x y

# compute & dump bonds
compute             cBonds all property/local btype batom1 batom2
compute             cDistBonds all bond/local engpot dist
dump                2 all local {SimulationParameters.dumpfreq} bonds.dump c_cBonds[*] c_cDistBonds[*]

# compute & dump angles
compute             cAngles all property/local aatom1 aatom2 aatom3 atype
compute             ctheta all angle/local theta
dump                3 all local {SimulationParameters.dumpfreq} angles.dump c_cAngles[*] c_ctheta

# timestep and running
thermo              {SimulationParameters.dumpfreq}
timestep            {SimulationParameters.dt}

# time integration
fix                 fLang moving langevin {SimulationParameters.temp} {SimulationParameters.temp} {SimulationParameters.damping} {SimulationParameters.seed}
fix                 fNVE moving nve
fix                 twodim all enforce2d

"""

    if add_run:
        to_write+= f"""run {SimulationParameters.steps}"""
    
    with open(SimulationParameters.inputfile, 'w') as f:
        f.write(to_write)

def write_input_treadmillingfilament(SimulationParameters, add_run = False):

    """
    This function writes the LAMMPS input file for a filament of building blocks.
    Each block corresponds to model 0 (angular version). 
    """
    
    # HARD CODE WHAT THE HYDROLYSIS CUTOFF IS
    SimulationParameters.rcut_hydrolysis = SimulationParameters.cutoff_top*0.8

    # the actual input file
    to_write=f"""
# Simulation with filament using model 0 building blocks

units               lj
atom_style          molecular
dimension           2 
boundary            f f p
read_data           configuration.dat
molecule            monomer_template monomer.dat
group               moving type < 8

# set reflections 
fix xwalls all wall/reflect xlo EDGE xhi EDGE
fix ywalls all wall/reflect ylo EDGE yhi EDGE

# designate bottom particles (they will move on a rail)
variable            bottom_group_atom_type atom (type==1)||(type==2)||(type==3) 
group               bottom dynamic all var bottom_group_atom_type every 1

variable            type_8 atom (type==8)
group               type_8_dynamic dynamic all var type_8 every 1

# trap the building block at the bottom - 1D motion
variable force_trapping_bottom atom -100000*y
variable energy atom 50000*y*y
fix trap_bottom bottom addforce 0.0 v_force_trapping_bottom 0.0 every 1 energy v_energy

# ensure safe communication cutoff (to keep the shape of the building block correctly)
comm_modify cutoff {SimulationParameters.bond_1_rc + SimulationParameters.bond_3_rc + 0.5}

# .............
# bonds
bond_style          harmonic
bond_coeff          1 {SimulationParameters.bond_1_k} {SimulationParameters.bond_1_rc} 
bond_coeff          2 {SimulationParameters.bond_2_k} {SimulationParameters.bond_2_rc} 
bond_coeff          3 {SimulationParameters.bond_3_k} {SimulationParameters.bond_3_rc} 

# .............
# angles
angle_style          harmswitch
angle_coeff          1 {SimulationParameters.angle_1_k} 180 360 
angle_coeff          2 {SimulationParameters.angle_2_k} 90 360 
angle_coeff          3 {SimulationParameters.angle_3_k} {SimulationParameters.angle_3_eq} {SimulationParameters.angle_3_rc}
angle_coeff          4 {SimulationParameters.angle_4_k} {SimulationParameters.angle_4_eq} {SimulationParameters.angle_4_rc}

# .............
# interaction potentials          
pair_style          cosine/squared 2.0
pair_coeff          * * 0 0 1             

# ....... within building block
pair_coeff          4 7  10.0 {SimulationParameters.sigma_preventinversion} {SimulationParameters.sigma_preventinversion} wca

# ....... for polymerization
# bottom binding sites interact through volume exclusion
pair_coeff          1 3 {SimulationParameters.epsilon_bottom} {SimulationParameters.sigma} {SimulationParameters.sigma} wca
# top particles attract each other
pair_coeff          4 6 {SimulationParameters.epsilon_top} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca
# interactions for when there is hydrolysis
pair_coeff          6 9 {SimulationParameters.epsilon_top_final} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca
pair_coeff          4 10 {SimulationParameters.epsilon_top_final} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca
pair_coeff          9 10 {SimulationParameters.epsilon_top_final} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca

# particles that are bonded do not interact with each other (except 4 and 7, which are 3rd neighbours)
special_bonds       lj 0.0 0.0 1.0

# ........... FIX BOND REACT PART
molecule pre_hydrolysis_1 ../../reactions/pre_hydrolysis_1.dat
molecule pre_hydrolysis_2 ../../reactions/pre_hydrolysis_2.dat
molecule pre_hydrolysis_3 ../../reactions/pre_hydrolysis_3.dat
molecule pre_hydrolysis_4 ../../reactions/pre_hydrolysis_4.dat

molecule post_hydrolysis_1 ../../reactions/post_hydrolysis_1.dat
molecule post_hydrolysis_2 ../../reactions/post_hydrolysis_2.dat
molecule post_hydrolysis_3 ../../reactions/post_hydrolysis_3.dat
molecule post_hydrolysis_4 ../../reactions/post_hydrolysis_4.dat

fix CHEMISTRY all bond/react reset_mol_ids no &
react Hydro1 all {SimulationParameters.hydrolysisfreq} 0 {SimulationParameters.rcut_hydrolysis} pre_hydrolysis_1 post_hydrolysis_1 ../../reactions/map_hydrolysis_1.dat prob {SimulationParameters.hydrolysis_probability} {SimulationParameters.seed+1} &
react Hydro2 all {SimulationParameters.hydrolysisfreq} 0 {SimulationParameters.rcut_hydrolysis} pre_hydrolysis_2 post_hydrolysis_2 ../../reactions/map_hydrolysis_2.dat prob {SimulationParameters.hydrolysis_probability} {SimulationParameters.seed+2} &
react Hydro3 all {SimulationParameters.hydrolysisfreq} 0 {SimulationParameters.rcut_hydrolysis} pre_hydrolysis_3 post_hydrolysis_3 ../../reactions/map_hydrolysis_3.dat prob {SimulationParameters.hydrolysis_probability} {SimulationParameters.seed+3} &
react Hydro4 all {SimulationParameters.hydrolysisfreq} 0 {SimulationParameters.rcut_hydrolysis} pre_hydrolysis_4 post_hydrolysis_4 ../../reactions/map_hydrolysis_4.dat prob {SimulationParameters.hydrolysis_probability} {SimulationParameters.seed+4}

# dump trajectory
dump                1 all custom {SimulationParameters.dumpfreq} output.dump id mol type x y

# compute & dump bonds
compute             cBonds all property/local btype batom1 batom2
compute             cDistBonds all bond/local engpot dist
dump                2 all local {SimulationParameters.dumpfreq} bonds.dump c_cBonds[*] c_cDistBonds[*]

# compute & dump angles
compute             cAngles all property/local aatom1 aatom2 aatom3 atype
compute             ctheta all angle/local theta
dump                3 all local {SimulationParameters.dumpfreq} angles.dump c_cAngles[*] c_ctheta

# timestep and running
thermo              {SimulationParameters.dumpfreq}
timestep            {SimulationParameters.dt}

# time integration
fix                 fLang moving langevin {SimulationParameters.temp} {SimulationParameters.temp} {SimulationParameters.damping} {SimulationParameters.seed}
fix                 fNVE moving nve
fix                 twodim all enforce2d

"""

    if add_run:
        to_write+= f"""run {SimulationParameters.steps}"""
    
    with open(SimulationParameters.inputfile, 'w') as f:
        f.write(to_write)

def write_input_stabilityexperiment(SimulationParameters, add_run = False):

    """
    This function writes the LAMMPS input file for a filament of building blocks.
    Each block corresponds to model 0 (angular version). 
    The LAMMPS run stops if the filament breaks, which means that an 
    interface disapears
    """

    # the actual input file
    to_write=f"""
# Simulation with filament using model 0 building blocks

units               lj
atom_style          molecular
dimension           2 
boundary            f f p
read_data           configuration.dat
molecule            monomer_template monomer.dat

# set reflections 
fix xwalls all wall/reflect xlo EDGE xhi EDGE
fix ywalls all wall/reflect ylo EDGE yhi EDGE

# designate bottom particles (they will move on a rail)
variable            bottom_group_atom_type atom (type==1)||(type==2)||(type==3) 
group               bottom dynamic all var bottom_group_atom_type every 1

# trap the building block at the bottom - 1D motion
variable force_trapping_bottom atom -100000*y
variable energy atom 50000*y*y
fix trap_bottom bottom addforce 0.0 v_force_trapping_bottom 0.0 every 1 energy v_energy

# ensure safe communication cutoff (to keep the shape of the building block correctly)
comm_modify cutoff {SimulationParameters.bond_1_rc + SimulationParameters.bond_3_rc + 0.5}

# .............
# bonds
bond_style          harmonic
bond_coeff          1 {SimulationParameters.bond_1_k} {SimulationParameters.bond_1_rc} 
bond_coeff          2 {SimulationParameters.bond_2_k} {SimulationParameters.bond_2_rc} 
bond_coeff          3 {SimulationParameters.bond_3_k} {SimulationParameters.bond_3_rc} 

# .............
# angles
angle_style          harmswitch
angle_coeff          1 {SimulationParameters.angle_1_k} 180 360 
angle_coeff          2 {SimulationParameters.angle_2_k} 90 360 
angle_coeff          3 {SimulationParameters.angle_3_k} {SimulationParameters.angle_3_eq} {SimulationParameters.angle_3_rc}
angle_coeff          4 {SimulationParameters.angle_4_k} {SimulationParameters.angle_4_eq} {SimulationParameters.angle_4_rc}

# .............
# interaction potentials          
pair_style          hybrid/overlay cosine/squared {SimulationParameters.cutoff_top} zero {SimulationParameters.conservative_cutoff_top}
pair_coeff          * * zero {SimulationParameters.conservative_cutoff_top}
pair_coeff          * * cosine/squared 0 0 1             

# ....... within building block
pair_coeff          4 7  cosine/squared 10.0 {SimulationParameters.sigma_preventinversion} {SimulationParameters.sigma_preventinversion} wca

# ....... for polymerization
# bottom binding sites interact through volume exclusion
pair_coeff          1 3 cosine/squared {SimulationParameters.epsilon_bottom} {SimulationParameters.sigma} {SimulationParameters.sigma} wca
# top particles attract each other
pair_coeff          4 6 cosine/squared {SimulationParameters.epsilon_top} {SimulationParameters.sigma} {SimulationParameters.cutoff_top} wca

# particles that are bonded do not interact with each other (except 4 and 7, which are 3rd neighbours)
special_bonds       lj 0.0 0.0 1.0

# dump trajectory
dump                1 all custom {SimulationParameters.dumpfreq} output.dump id mol type x y

# compute & dump bonds
compute             cBonds all property/local btype batom1 batom2
compute             cDistBonds all bond/local engpot dist
dump                2 all local {SimulationParameters.dumpfreq} bonds.dump c_cBonds[*] c_cDistBonds[*]

# compute & dump angles
compute             cAngles all property/local aatom1 aatom2 aatom3 atype
compute             ctheta all angle/local theta
dump                3 all local {SimulationParameters.dumpfreq} angles.dump c_cAngles[*] c_ctheta

# timestep and running
thermo              {SimulationParameters.dumpfreq}
timestep            {SimulationParameters.dt}

# time integration
fix                 fLang all langevin {SimulationParameters.temp} {SimulationParameters.temp} {SimulationParameters.damping} {SimulationParameters.seed}
fix                 fNVE all nve
fix                 twodim all enforce2d

# ....................................
# routine to halt the simulation when the polymer breaks
group all4 type 4
group all6 type 6

# local lists of near pairs and their attributes
compute pl all pair/local dist # per-pair distances within force cutoff
compute ids all property/local patom1 patom2 ptype1 ptype2

# Python function that returns max inter-molecular 4–6 distance
variable haltflag python max_inter_mol_46
python max_inter_mol_46 input 1 SELF return v_haltflag format pf  file max_inter_mol_46.py

"""

    to_write+="""
# stop the first time that the intermolecular distance exceeds threshold
fix stop all halt 10 v_haltflag == 1 message yes

"""

    if add_run:
        to_write+= f"""run {SimulationParameters.steps}"""
    
    with open(SimulationParameters.inputfile, 'w') as f:
        f.write(to_write)

