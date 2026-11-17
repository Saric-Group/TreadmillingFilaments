import restart_fragmentation_analysis
import fragmentation_lammps
import os
import numpy as np
import glob
from flow import FlowProject
from pathlib import Path
import re
import random

def run_fragmentation_analysis(old_sim_polymer_path,new_folder,eps,cutoff, include_chevron_bonds, which_simulation,epsC,epsT):
    
    path = Path(old_sim_polymer_path)
    name = path.name #gets seed folder name
    
    # get the seed number from the folder name
    seed = int(re.search(r"\d+$", name).group())
    #seed = random.randint(0, 100000)
    
    sim_path_new = new_folder + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/tail_blocking_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/capper_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    
    #old_sim_polymer_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/tail_blocking_monomer_deletion2/sd10878' #5880000 for 5 monomers tail
    #old_sim_polymer_path = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/volume_exclusion_monomer_deletion2/sd10878'#6720000 for 5 monomers capper
    #old_sim_polymer_path = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/LG_withextrabond_kineticpolarity/sd10878" #6950000 for 50 monomers, 3870000 for 25 monomers, 330000 for 4 monomers, 830000 for 5 monomers
    #old_sim_polymer_path ="/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/postanalysis_rerun_constant_concentration_final/littleguys_chevron_fullinterface_nomiddlemonomer_2/sd10878" #proper chevron, 6380000 for 50 monomers, 3920000 for 25
    trajectory_output_file = old_sim_polymer_path + "/output_com.xyz"
    bond_output_file = old_sim_polymer_path + "/bonds.xyz"
    new_output_file_config = sim_path_new + "/restart_config.dat"
    os.makedirs(sim_path_new, exist_ok=True)

    xlo, xhi, ylo, yhi = restart_fragmentation_analysis.create_restart_excluding_molecules(
        atom_dump_file=trajectory_output_file,
        bond_dump_file=bond_output_file,
        which_simulation=which_simulation,
        output_file=new_output_file_config,
        include_chevron_bonds = include_chevron_bonds,
        timestep_index = -2 #int(6720000/10000)
    )
    #dumo_steps was 10000
    fragmentation_lammps.run_simulation(sim_path_new, Lbox=60,  nMonsSol=1, seed=seed, cutoff=cutoff, run_interval=10, run_steps=2000000, dump_steps=100, debug_mode=True,eps=eps, print_mode=False, Nseeds=1, extra_chevron_bond=include_chevron_bonds, which_simulation=which_simulation, epsC=epsC,epsT=epsT)
    
    

