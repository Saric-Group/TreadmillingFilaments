from flow import FlowProject
import os
import growing_rectbox_const_conc

class Project(FlowProject):
    pass

@Project.operation
def run_growing_polymer(job):
    """Run LAMMPS for one job."""
    path = job.sp.path
    print(f"Running LAMMPS for job {job.id} at {path}", flush=True)
    growing_rectbox_const_conc.run_simulation(path, path)
    print(f"Finished LAMMPS for job {job.id}", flush=True)

def run_signac_lammps(folder):
    """Run jobs locally with 50 parallel workers."""
    Project(folder).run(np=300)  # 50 parallel jobs, 20min timeout each
