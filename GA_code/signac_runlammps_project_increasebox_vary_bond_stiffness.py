from flow import FlowProject
import os
import growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox

class Project(FlowProject):
    pass

@Project.operation
def run_growing_polymer(job):
    """Run LAMMPS for one job."""
    path = job.sp.path
    print(f"Running LAMMPS for job {job.id} at {path}", flush=True)
    growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox.run_simulation(path, path)
    print(f"Finished LAMMPS for job {job.id}", flush=True)

def run_signac_lammps(folder):
    """Run jobs locally with 50 parallel workers."""
    Project(folder).run(np=30)  # 50 parallel jobs, 20min timeout each


"""
# project.py
from flow import FlowProject
import growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox

class Project(FlowProject):
    pass

#@Project.post(is_done)
@Project.operation
def run_growing_polymer(job):
    path = job.sp.path
    growing_polymer_increasebox_changeboxsize_vary_bond_stiffness_rectbox.run_simulation(path,path)


def run_signac_lammps(folder):
    Project(folder).run(np=10)#50 processes in parallel
"""