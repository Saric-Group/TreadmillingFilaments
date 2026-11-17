import restart_fragmentation_analysis
import run_fragmentation_analysis
import os
import numpy as np
import glob
from flow import FlowProject
import signac
import signac_fragmentation
import sys

eps = int(sys.argv[1])  # get eps from command line


#eps = 1
which_simulation = 'GA'  #options: 'littleguys', 'GA' due to different interactions
cutoff = 1.2
include_chevron_bonds = False  #True or False
#which_setup = 'LG_1state'
epsC=3
epsT=7
#which_setup = 'capper'
#which_simulation = 'GA' 

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/little_guys_full_longer_deletion2"
#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/LG_withextrabond_kineticpolarity"

old_sim_folder = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/tail_blocking_monomer_deletion2'
#old_sim_folder = '/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/volume_exclusion_monomer_deletion2'

new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_largebox/fragmentation_analysis_GA_tail_frequent_dumping_for_movies/eps_" + str(eps) + "_cutoff_" + str(cutoff) #epsC_"+str(epsC)+"epsT_"+str(epsT)+"/eps_" + str(eps) + "_cutoff_" + str(cutoff)

signac_folder = signac_fragmentation.make_signac_folder(old_sim_folder = old_sim_folder, new_sim_folder = new_folder, eps = eps, cutoff = cutoff, include_chevron_bonds = include_chevron_bonds,which_simulation=which_simulation, epsC=epsC,epsT=epsT)
signac_fragmentation.run_signac_lammps(signac_folder)