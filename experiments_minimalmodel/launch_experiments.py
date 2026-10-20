import json, os, itertools, random
from lib_littleguys import *
from dataclasses import asdict 

if __name__=="__main__":

    # ......... COMMON FOR EVERY EXPERIMENT
    # | | | | | | | | | | | |
    # V V V V V V V V V V V V 

    # [CHANGE HERE] set the name of your experiment
    # choose from: 'sim_single_monomer', 'sim_directional_growth', 'sim_treadmilling', 'sim_stabilitypolymer'
    experiment_codename = 'sim_treadmilling_pnasrevision' 
    # [CHANGE HERE] set the index so that you can choose from the parameters
    index = "3"
    # [CHANGE HERE] how many seeds do you want 
    num_seeds = 100

    # factor increase determines 
    # how much we decide to dilute the system
    factor_increase = 2

    # set where your parameters and slurm parameters are
    name_json_file = f'{experiment_codename}_{index}.json'
    slurm_json_file = f'{experiment_codename}_slurm_params_{index}.json'

    # reading parameters
    with open(f"json_files/{name_json_file}", "r") as f:
        data = json.load(f)
    data['simulation_folder'] = experiment_codename

    # get the keys and values
    keys = data["variables"].keys()
    values = data["variables"].values()

    # create the bash directory so that you can launch jobs
    launch_directories, counter_directories = slurm_commands.prepare_bash_launch(experiment_codename)
    simulations_to_launch = 0

    # for each variable that we want to change
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))

        # generate all seed directories
        for seeds in range(num_seeds):

            # introduce the data for the corresponding combination
            dirname = "_"
            for ky in combo_dict:
                data[ky] = combo_dict[ky]
                dirname+=f"{ky.replace('_','')}_{combo_dict[ky]}_"

            # introduce desired seed
            current_seed = random.randint(1, 100000)
            test_dirname=f'{dirname}seed_{current_seed}_'

            # check that the directory does not exist (i.e. not repeating the seed)
            # if it exists, keep on generating a seed until it doesnt anymore
            while os.path.isdir(f'{experiment_codename}/runs/{test_dirname}'):
                current_seed = random.randint(1, 100000)
                test_dirname=f'{dirname}seed_{current_seed}_'

            # commit to current seed because it is not repeated
            dirname +=f'seed_{current_seed}_'
            data["seed"] = current_seed

            # read the parameters for the simulation
            Simulation = parameters.SimulationParameters(**data)

            # save the directory name to launch later with slurm
            simulations_to_launch += 1
            launch_directories.write(f'./runs/{dirname}')
            launch_directories.write('\n')
            dirname = f'{Simulation.simulation_folder}/runs/{dirname}'

            # create directory for the simulation
            os.makedirs(dirname, exist_ok = True)
            Simulation.update_directories(dirname = dirname)        

            # ...... [DEVELOPMENT HERE] SPECIFIC TO THE SIMULATION YOU ARE INTERESTED IN
            # | | | | | | | | | | | |
            # V V V V V V V V V V V V 

            if experiment_codename in ['sim_single_monomer', 'dev_prevent_inversion']:

                # initialize the system
                initializations.write_initconfig(Simulation)
                # generate the LAMMPS input file
                lammps_inputs.write_input_singlemonomer(Simulation)

            elif experiment_codename in ['sim_directional_growth']:

                # initialize the simulation and write input files
                initializations.write_initconfig(Simulation, is_seed = True)
                molecule_templates.write_monomer_template(Simulation)
                lammps_inputs.write_input_polymerizingfilament(Simulation)

                # write the parameters file in the directory
                parameters_simulation = asdict(Simulation)

                # write the json polymerization file
                parameters_simulation["configfile"] = "configuration.dat"
                parameters_simulation["inputfile"] = "inputfile.dat"
                parameters_simulation["monomerfile"] = "monomer.dat"
                with open(f'{dirname}/parameters.json', "w") as f:
                    json.dump(parameters_simulation, f, indent = 4)

                # write the polymerization program within the directory
                polymerization.write_polymerization_program(dirname)
            
            elif experiment_codename in ['sim_treadmilling', 'sim_treadmilling_pnasrevision', 'sim_treadmilling_pnasrevision_diluted', 'sim_treadmilling_pnasrevision_evenmorediluted']:
                
                # initialize the simulation and write input files
                initializations.write_initconfig(Simulation, is_seed = True, treadmilling = True) # start from longer filament, seed moves
                molecule_templates.write_monomer_template(Simulation) # the same
                lammps_inputs.write_input_treadmillingfilament(Simulation) # make sure seed moves

                # write the parameters file in the directory
                parameters_simulation = asdict(Simulation)

                # write the json polymerization file
                parameters_simulation["configfile"] = "configuration.dat"
                parameters_simulation["inputfile"] = "inputfile.dat"
                parameters_simulation["monomerfile"] = "monomer.dat"
                with open(f'{dirname}/parameters.json', "w") as f:
                    json.dump(parameters_simulation, f, indent = 4)

                # write the treadmilling program within the directory
                treadmilling.write_treadmilling_program(dirname, factor_increase)

            elif experiment_codename in ['sim_stabilitypolymer', 'sim_stabilitypolymer_uncoupled', 'sim_stabilitypolymer_coupled']:
                
                # write a filament of a specific length
                initializations.write_initconfig(Simulation, is_seed = True)
                # write the python file that will be used to detect the halting
                initializations.write_python_function(Simulation)
                # write the monomer template -- unnecessary for this type of sim?
                molecule_templates.write_monomer_template(Simulation)
                # write the LAMMPS input file
                lammps_inputs.write_input_stabilityexperiment(Simulation, add_run = True)

    # ...... SLURM FILES
    # | | | | | | | | | | | |
    # V V V V V V V V V V V V 
    
    with open(f"json_files/{slurm_json_file}", "r") as f:
        data_slurm = json.load(f)
    data_slurm['simulation_folder'] = experiment_codename
    data_slurm['directory_number']  = counter_directories

    if experiment_codename in ['sim_single_monomer', 'dev_prevent_inversion', 'sim_stabilitypolymer', 'sim_stabilitypolymer_uncoupled', 'sim_stabilitypolymer_coupled']:
        slurm_commands.write_slurm_launch(data_slurm)
    elif experiment_codename in ['sim_directional_growth']:
        slurm_commands.write_slurm_launch_directionalgrowth(data_slurm)
    elif experiment_codename in ['sim_treadmilling', 'sim_treadmilling_pnasrevision', 'sim_treadmilling_pnasrevision_diluted', 'sim_treadmilling_pnasrevision_evenmorediluted']:
        slurm_commands.write_slurm_launch_treadmilling(data_slurm)

    print(f"Go to {experiment_codename}/ to launch {simulations_to_launch} simulations")
    print(f"You can launch ---> {simulations_to_launch} simulations by running sbatch --array=1-{simulations_to_launch} slurm_submission.sh")
    


