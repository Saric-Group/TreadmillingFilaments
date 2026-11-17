# row_jobs.py
import os
import glob
from flow import FlowProject

class Project(FlowProject):
    """Row project for managing LAMMPS jobs."""
    pass

def write_text_file_deme(gen_id, deme_id, RUN_DIR):
    """
    Write a text file containing all seed paths for one deme and generation.
    Returns the file path and number of jobs.
    """
    paths = glob.glob(os.path.join(RUN_DIR, f"deme_{deme_id}/gen_{gen_id+1}/ind_*/seed_*"))
    
    submit_folder = os.path.join(RUN_DIR, 'submission_files')
    os.makedirs(submit_folder, exist_ok=True)
    
    file_name = f"paths_gen_{gen_id+1}_deme_{deme_id}.txt"
    file_path = os.path.join(submit_folder, file_name)
    
    with open(file_path, "w") as f:
        for p in paths:
            f.write(f"{p}\n")
    
    return file_path, len(paths)

def make_signac_folder(gen_id, deme_id, RUN_DIR):
    """
    Creates a Row project folder and initializes a job for every seed path.
    Returns the path to the Row project folder.
    """
    paths = glob.glob(os.path.join(RUN_DIR, f"deme_{deme_id}/gen_{gen_id+1}/ind_*/seed_*"))
    
    submit_folder = os.path.join(RUN_DIR, 'submission_files')
    row_project_folder = os.path.join(submit_folder, f"deme_{deme_id}_gen_{gen_id+1}")
    os.makedirs(row_project_folder, exist_ok=True)
    
    # Initialize the Row project
    project = Project.init_project(row_project_folder)
    
    for p in paths:
        sp = {"path": p}
        job = project.open_job(sp).init()
    
    return row_project_folder
