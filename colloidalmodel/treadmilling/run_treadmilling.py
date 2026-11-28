import restart_fragmentation_analysis
import growing_rectbox_treadmilling_mol_hydrolysis_more_monomers_fixbondreact_bubbles_sameconc
import os
import numpy as np
import glob
from flow import FlowProject
from pathlib import Path
import re
import shutil

def run_treadmilling(old_sim_polymer_path,new_folder,eps,epsilon_top_final, hydrolysis_probability,which_simulation):
    """
    #old setup with starting with large polymers
    path = Path(old_sim_polymer_path)
    name = path.name #gets seed folder name
    
    # get the seed number from the folder name
    seed = int(re.search(r"\d+$", name).group())

    sim_path_new = new_folder + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/tail_blocking_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/capper_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    trajectory_output_file = old_sim_polymer_path + "/output_com.xyz"
    bond_output_file = old_sim_polymer_path + "/bonds.xyz"
    new_output_file_config = sim_path_new + "/restart_config.dat"
    os.makedirs(sim_path_new, exist_ok=True)
    """
    #SETUP STARTING FROM PENTAMER TREADMILLING
    path = Path(old_sim_polymer_path)
    name = path.name #gets seed folder name
    
    # get the seed number from the folder name
    seed = int(re.search(r"\d+$", name).group())
    
    sim_path_new = new_folder + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/tail_blocking_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    #sim_path_new = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/capper_fragmentation/eps_" + str(eps) + "/seed_" + str(seed)
    #SETUP FOR TREADMILLING FROM A PENTAMER
    old_path_sim_polymer_trimer = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/little_guys_full_longer_deletion2/sd13293'
    trajectory_output_file = old_path_sim_polymer_trimer + "/output_com.xyz"
    bond_output_file = old_path_sim_polymer_trimer + "/bonds.xyz"
    new_output_file_config = sim_path_new + "/restart_config.dat"
    os.makedirs(sim_path_new, exist_ok=True)
    
    if which_simulation == 'littleguys':

        xlo, xhi, ylo, yhi = restart_fragmentation_analysis.create_restart_excluding_molecules(
            atom_dump_file=trajectory_output_file,
            bond_dump_file=bond_output_file,
            which_simulation=which_simulation,
            output_file=new_output_file_config,
            timestep_index=185 #so that its a trimer - step =80, step = 185 for pentamer, 194 for hexamer, 465 for n=20 monomers
        )
        
        total_volume = (xhi - xlo) * (yhi - ylo)
        concentration = 0.002
        n_monomers = int(concentration * total_volume)
        
        shutil.copy(old_sim_polymer_path + "/monomer.dat", sim_path_new + "/monomer.dat")
        
        growing_rectbox_treadmilling_mol_hydrolysis_more_monomers_fixbondreact_bubbles_sameconc.run_simulation(sim_path_new, seed= seed,nMonsSol=1, run_interval=200, run_steps=2000000000, dump_steps=10000, eps = eps, cutoff =1.1, debug_mode=True, print_mode=False, Nseeds=1, hydrolysis_probability=hydrolysis_probability, epsilon_top_final=epsilon_top_final, which_simulation=which_simulation)
        #fragmentation_lammps.run_simulation(sim_path_new, Lbox=60,  nMonsSol=1, seed=seed, cutoff=1.4, run_interval=10, run_steps=2000000, dump_steps=10000, debug_mode=True,eps=eps, print_mode=False, Nseeds=1, which_simulation=which_simulation)
    
    

