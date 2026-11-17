import restart_fragmentation_analysis
import run_fragmentation_analysis
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
    cutoff = job.sp.cutoff
    include_chevron_bonds = job.sp.inclde_chevron_bonds
    which_simulation = job.sp.which_simulation
    epsC = job.sp.epsC
    epsT = job.sp.epsT
    run_fragmentation_analysis.run_fragmentation_analysis(path,new_folder,eps,cutoff,include_chevron_bonds,which_simulation,epsC,epsT)
    
    
def run_signac_lammps(folder):
    Project(folder).run(np=300)#10 processes in parallel

def make_signac_folder(old_sim_folder, new_sim_folder, eps, cutoff,include_chevron_bonds,which_simulation,epsC,epsT):

    paths = glob.glob(old_sim_folder + "/sd*")
    submit_folder = os.path.join(new_sim_folder,'submission_files')
    signac_folder = os.path.join(submit_folder,'all_seeds')
    project = signac.init_project(signac_folder)
    for p in paths:
        sp = {"path": p, "new_folder": new_sim_folder, "eps": eps, "which_simulation": which_simulation, "cutoff":cutoff, "inclde_chevron_bonds": include_chevron_bonds, "epsC":epsC, "epsT":epsT}
        job = project.open_job(sp).init()

    return signac_folder
"""
eps = 14
which_simulation = 'littleguys'  #options: 'littleguys', 'GA' due to different interactions
cutoff = 1.2
include_chevron_bonds = False  #True or False
which_setup = 'LG_2state'
epsC=3
epsT=4
#which_setup = 'capper'
#which_simulation = 'GA' 

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/little_guys_full_longer_deletion2"
old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/LG_withextrabond_kineticpolarity"

#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_final/little_guys_fragmentation_" + str(which_setup) + "/eps_" + str(eps) + "_cutoff_" + str(cutoff)
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_with_solid_box/little_guys_fragmentation_" + str(which_setup) + "/eps_" + str(eps) + "_cutoff_" + str(cutoff)
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_with_solid_box/little_guys_fragmentation_" + str(which_setup) + "/eps_" + str(eps) + "_cutoff_" + str(cutoff)

#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_epsC_3_epsT_6/little_guys_fragmentation_" + str(which_setup) + "/eps_" + str(eps) + "_cutoff_" + str(cutoff)
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_epsC_3_epsT_6_testing_box_size/little_guys_fragmentation_" + str(which_setup) + "_size_30/eps_" + str(eps) + "_cutoff_" + str(cutoff)

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/tail_blocking_monomer_deletion2"
#new_folder =  "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/tail_blocking_fragmentation/eps_" + str(eps)

#old_sim_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/good_ind_scan_for_asymmetry/volume_exclusion_monomer_deletion2"
#new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis/capper_fragmentation/eps_" + str(eps)

new_folder = "/nfs/scistore26/saricgrp/mkrstic/ga_code/postanalysis/fragmentation_analysis_largebox/fragmentation_analysis_littleguys_extrabond_50monomers/eps_" + str(eps) + "_cutoff_" + str(cutoff)

signac_folder = make_signac_folder(old_sim_folder = old_sim_folder, new_sim_folder = new_folder, eps = eps, cutoff = cutoff, include_chevron_bonds = include_chevron_bonds,which_simulation=which_simulation,epsC=epsC,epsT=epsT)
run_signac_lammps(signac_folder)
"""