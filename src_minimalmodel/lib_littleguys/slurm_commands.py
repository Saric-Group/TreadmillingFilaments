import os, glob

def prepare_bash_launch(experiment_codename):

    """
    Considers how many simulations have been launched already and tells you
    what directory to pass to slurm script in order to launch new simulations.
    """
    
    # create director_* files for launching directory
    bash_path = experiment_codename+"/bashdirs"
    isExist = os.path.exists(bash_path)
    if not isExist:
       os.makedirs(bash_path)

    # create outputs directory
    output_path = experiment_codename+"/output"
    isExist = os.path.exists(output_path)
    if not isExist:
       os.makedirs(output_path)

    # check how many directory files there are
    counter_directories = 1
    for files in glob.glob(bash_path+'/directories*'):
        counter_directories += 1

    print("LAUNCH ---> bashdirs/directories_"+str(counter_directories))
    launch_directories = open(bash_path+'/directories_'+str(counter_directories)+'.dat', 'w')
    return launch_directories, counter_directories


def write_slurm_launch(job_params):

   # function that writes the slurm launcher for a job array
   #
   # to launch the simulations in the cluster, you just have to type
   #        sbatch --array=1-X slurm_submission.sh
   # where X=the total number of jobs you want to submit
   # This code is useful to launch simulations that rely on running LAMMPS

   to_write =f"""
#SBATCH --job-name={job_params["job_name"]}
#SBATCH --output="output/output-%A_%a_%x.out"
#SBATCH --nodes {job_params["nodes"]}
#SBATCH -c 1
#SBATCH --time={job_params["hours"]}:00:00
#SBATCH --mem={job_params["memory"]}
#SBATCH --mail-user={job_params["usermail"]}
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=None

env > 'env_file.dat'
unset SLURM_EXPORT_ENV

#module load anaconda3/2023.04
#source /mnt/nfs/clustersw/Debian/bullseye/anaconda3/2023.04/activate_anaconda3_2023.04.txt

module purge
module load conda
module load openmpi
conda activate {job_params["environment_name"]}
"""

   to_write+="""
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
"""

   to_write+='path=$(sed "${'
   to_write+='SLURM_ARRAY_TASK_ID}q;d"'
   to_write+=f""" bashdirs/directories_{job_params["directory_number"]}.dat)
cd $path

#run the respective binary through SLURM's srun
srun --cpu_bind=verbose lmp -in inputfile.dat
"""   

   with open(job_params['simulation_folder']+"/slurm_submission.sh", 'w') as f:
      f.write("#!/bin/bash -l\n")
      f.write(to_write)

def write_slurm_launch_directionalgrowth(job_params):

   # function that writes the slurm launcher for a job array
   #
   # to launch the simulations in the cluster, you just have to type
   #        sbatch --array=1-X slurm_submission.sh
   # where X=the total number of jobs you want to submit
   # 
   # This code is useful to launch simulations with also python script

   to_write =f"""
#SBATCH --job-name={job_params["job_name"]}
#SBATCH --output="output/output-%A_%a_%x.out"
#SBATCH --nodes {job_params["nodes"]}
#SBATCH -c 1
#SBATCH --time={job_params["hours"]}:00:00
#SBATCH --mem={job_params["memory"]}
#SBATCH --mail-user={job_params["usermail"]}
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=None

env > 'env_file.dat'
unset SLURM_EXPORT_ENV

#module load anaconda3/2023.04
#source /mnt/nfs/clustersw/Debian/bullseye/anaconda3/2023.04/activate_anaconda3_2023.04.txt

module purge
module load conda
module load openmpi
conda activate {job_params["environment_name"]}
"""

   to_write+="""
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
"""

   to_write+='path=$(sed "${'
   to_write+='SLURM_ARRAY_TASK_ID}q;d"'
   to_write+=f""" bashdirs/directories_{job_params["directory_number"]}.dat)
cd $path

#run the respective binary through SLURM's srun
srun --cpu_bind=verbose python3 directional_growth.py
"""   

   with open(job_params['simulation_folder']+"/slurm_submission.sh", 'w') as f:
      f.write("#!/bin/bash -l\n")
      f.write(to_write)

def write_slurm_launch_treadmilling(job_params):

   # function that writes the slurm launcher for a job array
   #
   # to launch the simulations in the cluster, you just have to type
   #        sbatch --array=1-X slurm_submission.sh
   # where X=the total number of jobs you want to submit
   # 
   # This code is useful to launch simulations with also python script

   to_write =f"""
#SBATCH --job-name={job_params["job_name"]}
#SBATCH --output="output/output-%A_%a_%x.out"
#SBATCH --nodes {job_params["nodes"]}
#SBATCH -c 1
#SBATCH --time={job_params["hours"]}:00:00
#SBATCH --mem={job_params["memory"]}
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=None
#SBATCH --exclude=eta[331-332]

module purge
module load conda
module load openmpi

env > 'env_file.dat'
unset SLURM_EXPORT_ENV
source ~/conda/etc/profile.d/conda.sh

#module load anaconda3/2023.04
#source /mnt/nfs/clustersw/Debian/bullseye/anaconda3/2023.04/activate_anaconda3_2023.04.txt

conda activate {job_params["environment_name"]}
"""

   to_write+="""
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
"""

   to_write+='path=$(sed "${'
   to_write+='SLURM_ARRAY_TASK_ID}q;d"'
   to_write+=f""" bashdirs/directories_{job_params["directory_number"]}.dat)
cd $path

#run the respective binary through SLURM's srun
srun --cpu_bind=verbose python3 treadmilling.py
"""   

   with open(job_params['simulation_folder']+"/slurm_submission.sh", 'w') as f:
      f.write("#!/bin/bash -l\n")
      f.write(to_write)



