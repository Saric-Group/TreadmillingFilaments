import glob
import sys
import os

def write_slurm_array(gen_id,deme_id,RUN_DIR):

    paths = glob.glob(RUN_DIR +"/deme_"+ str(deme_id)+"/gen_"+ str(gen_id+1)+"/ind_*/seed_*") #all paths in  one generation
    
    gname = 'gen_' + str(gen_id+1) + '_deme_'+ str(deme_id)#name of array run
    submit_folder = os.path.join(RUN_DIR,'submission_files')
    os.chdir(os.path.join(RUN_DIR,'submission_files'))
    spath = os.getcwd()

    path_name = "paths_gen_" + str(gen_id+1) + "_deme_" + str(deme_id) + ".txt"
    names_name = "names_gen_" + str(gen_id+1) + "_deme_" + str(deme_id) + ".txt"
    f = open(os.path.join(submit_folder,path_name), "w")
    g = open(os.path.join(submit_folder,names_name), "w")
    for p in paths:
    	f.write("%s\n"%p)
    	ps = p.split('/')
    	n = ps[0]
    	for i in range(len(ps)-1):
    		n = '%s_%s'%(n,ps[i+1])
    	g.write("%s\n"%(n))
    f.close()
    g.close()

    num_of_ind = len(paths)

    script_name = "job_array_gen_" + str(gen_id+1) + "_deme_"+ str(deme_id) +".sh"
    f = open(os.path.join(submit_folder,script_name), "w")
    f.write('''#!/bin/bash\n#\n''')
    f.write("#SBATCH --job-name=%s\n"%(gname))
    f.write("#SBATCH --output=%s"%(gname))
    f.write("_%j.out\n")
    f.write("#SBATCH --error=job_%j.err\n")
    f.write("#SBATCH --array=1-%d%%10\n"%(len(paths)))
    f.write("#SBATCH --exclude=zeta[243-263],beta234,beta231,gamma156\n")
    f.write('''#Maximum runtime is limited to 10 days, ie. 240 hours\n''')
    f.write("#SBATCH --time=00:05:00\n#\n#SBATCH --mem=10G\n")
    f.write('''#
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#
#for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1
#
#run the respective binary through SLURM's srun\n''')

    f.write('path=$(sed "${SLURM_ARRAY_TASK_ID}q;d" %s)\n'%(path_name))
    f.write('job_name=$(sed "${SLURM_ARRAY_TASK_ID}q;d" %s)\n'%(names_name))
    f.write('''
cd ${path}
echo init
echo task job id:       ${SLURM_JOB_ID}
echo array job id:      ${SLURM_ARRAY_JOB_ID}
echo array task index:  ${SLURM_ARRAY_TASK_ID}
echo simulation folder: ${path}
echo name:              ${job_name}
echo hostname:          ${SLURM_JOB_NODELIST}
echo submit host:       ${SLURM_SUBMIT_HOST}\n''')
    f.write('ln %s/%s_${SLURM_JOB_ID}.out %s/${job_name}.out\n'%(spath,gname,spath))
    f.write('srun --exclusive --cpu_bind=verbose /nfs/scistore15/saricgrp/mkrstic/little_guys/LAMMPS/lammps/src/lmp_serial -in in6.local')
    #f.write("srun --cpu_bind=verbose %s\n"%(command))
    f.close()

    print('SLURM SCRIPT SUBMITTED FOR GENERATION ' + str(gen_id+1) + ' DEME ' + str(deme_id))

    return script_name, num_of_ind