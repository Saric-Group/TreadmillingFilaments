import restart_fragmentation_analysis
import run_treadmilling
import os
import numpy as np
import glob
from flow import FlowProject
import signac

class Project(FlowProject):
    pass

@Project.operation
def run(job):
    path = job.sp.path
    new_folder = job.sp.new_folder
    eps = job.sp.eps
    epsilon_top_final = job.sp.epsilon_top_final
    hydrolysis_probability = job.sp.hydrolysis_probability
    which_simulation = job.sp.which_simulation
    run_treadmilling.run_treadmilling(path,new_folder,eps,epsilon_top_final, hydrolysis_probability,which_simulation)
    
    
def run_signac_lammps(folder):
    Project(folder).run(np=300)#200 processes in parallel

def make_signac_folder(old_sim_folder, new_sim_folder, eps, epsilon_top_final,hydrolysis_probability,which_simulation):

    paths = glob.glob(old_sim_folder + "/sd*")
    submit_folder = os.path.join(new_sim_folder,'submission_files')
    signac_folder = os.path.join(submit_folder,'all_seeds')
    project = signac.init_project(signac_folder)
    for p in paths:
        sp = {"path": p, "new_folder": new_sim_folder, "eps": eps, "epsilon_top_final": epsilon_top_final,"hydrolysis_probability":hydrolysis_probability,"which_simulation": which_simulation}
        job = project.open_job(sp).init()

    return signac_folder


eps = 100
epsilon_top_final = 6
radius_region_pool = 16
hydrolysis_probability = 0.00001 #Nevery = 10000
which_simulation = 'littleguys'  #options: 'littleguys', 'GA' due to different interactions
"""

eps = 30
epsilon_top_final = 13
hydrolysis_probability = 0.0001 #Nevery = 10000
which_simulation = 'GA'  
"""
old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/little_guys_full_longer_deletion2" #for treadmilling from scratch, now want to get longet trajectories so restarting from old ones
new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling_LG/new_runs_12_11_circle_centered_on_filament_nofullinterfaceanalyse/radius"+str(radius_region_pool)+"/hydrolysis_prob_" + str(hydrolysis_probability) + '_eps_final_' + str(epsilon_top_final) + "_eps_nonhydrolyzed_" + str(eps) 

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling_LG/little_guys_constant_conc_monomer_pool_bubbles_pentamer_more_dumping_getmolidfromatomid_lowerepshydrolysis6_DILUTEDr12/hydrolysis_prob_1e-07_eps_final_8_eps_nonhydrolyzed_100bubbleradius_8_same_conc_as_growth" #for treadmilling from scratch, now want to get longet trajectories so restarting from old ones
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/detailed_balance/" + "_eps_" + str(eps)
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling_LG/little_guys_constant_conc_monomer_pool_bubbles_pentamer_more_dumping_getmolidfromatomid_lowerepshydrolysis6_DILUTEDr12/hydrolysis_prob_" + str(hydrolysis_probability) + '_eps_final_' + str(epsilon_top_final) + "_eps_nonhydrolyzed_" + str(eps) + "_restart" #+ 'bubbleradius_8_same_conc_as_growth'


#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/tail_blocking_monomer_deletion2"
#new_folder =  "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling/tail_blocking/hydrolysis_prob_" + str(hydrolysis_probability) + '_eps_final_' + str(epsilon_top_final) + "_eps_nonhydrolyzed_" + str(eps)

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/volume_exclusion_monomer_deletion2"
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/treadmilling/capper/hydrolysis_prob_" + str(hydrolysis_probability) + '_eps_final_' + str(epsilon_top_final) + "_eps_nonhydrolyzed_" + str(eps)

signac_folder = make_signac_folder(old_sim_folder = old_sim_folder, new_sim_folder = new_folder, eps = eps, epsilon_top_final = epsilon_top_final, hydrolysis_probability = hydrolysis_probability, which_simulation=which_simulation)
run_signac_lammps(signac_folder)