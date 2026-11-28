# Cytomotive (treadmilling) filaments
This repository contains the source codes for the results reported in the paper 'Design and self-assembly of cytomotive filaments' (M. Krstić*, C. Vahnille-Campos*, B. Baum, M. Muñoz-Basagoiti and A. Šarić, 2026).

| Directory | Contents |
| ---------- | ---------|
| `src_minimalmodel` | Contains the `lib_minimalmodel` Python library for the minimal model (model 0) simulations |
| `experiments_minimalmodel` | Contains the `json` parameter files for the data explored in the paper, together with the simulation launcher|
| `colloidalmodel` | Contains simulations that pertain to the colloidal model |
| `GA_code` | Contains the evolutionary algorithm used to obtain designs |

## Installation of the `src_minimalmodel`

Download the repository, and install the source code by running the following command where the `*.toml` file is:

```pip install -e .```

### How to run the codes

Simulations require the `lib_minimalmodel` library. 

- Simulation data is produced in the `experiments_minimalmodel` folder (also launching point)
- Fill in `.json` files to launch the parameters for the simulations
- Use the script `launch_experiments.py` to launch simulations.
    - It produces both the directory tree as well as the slurm scripts to submit simulations to the cluster (if applicable)
    - The job arrays can be launched by running `sbatch --array=1-X slurm_submissions.sh`
 
The GA code can be run by running MAIN_GA_code, while the colloidal simulations are run through run_treadmilling/fragmentation/growth. All of these LAMMPS simulation are run in python, and parallelised using https://signac.readthedocs.io/en/latest/#. The other files contain helper function which mostly handle input file writing. The GA_code and growth simulations require no prior inputs, however the code for fragmentation and treadmilling simulations requires an input path of a polymer grown to a certain length as a starting configuration. When reproducing the data therefore, one should first run growth simulations for the colloidal model, which should be used as inputs for the fragmentation and treadmilling simulations.

## LAMMPS Compilation

Running both the minimal model simulations as well as the colloidal simulations requires LAMMPS. The LAMMPS packages needed are:
- MOLECULE -> ```atom_style molecular```, ```bond_style harmonic```, ```angle_style harmonic```
- EXTRA-PAIR -> ```pair_style cosine/square```

Additionally, we also use a custom-made LAMMPS angle potential called `harmswitch` (within the `MOLECULE` directory), that allows us to introduce the switch in MD versions of model 0.

To compile, you can use the `lammpscompilation_treadmilling.sh` script.

To run the code in GA_code and colloidalmodel, LAMMPS is run through python scripts. To enable this build LAMMPS following instructions from https://docs.lammps.org/Python_install.htmlhttps://docs.lammps.org/Python_install.html

### Updates in LAMMPS needed for the code
To be able to run this script and use fix/adapt with cosine squared potential add the following to `src/EXTRA-PAIR/pair_cosine_squared.cpp`:
    
```
void *PairCosineSquared::extract(const char *str, int &dim){
    dim = 2;
    if (strcmp(str, "eps") == 0) return (void *) epsilon;
    if (strcmp(str, "sigma") == 0) return (void *) sigma;
    return nullptr;
}

```

and the following to `src/EXTRA-PAIR/pair_cosine_squared.h` of your LAMMPS folder:

```
void *extract(const char *, int &) override;
```

Additionally, comment out the rotation in fix_deposit.cpp in the following way(excerpt from fix_deposit.cpp):

```
// coords = coords of all atoms
// for molecule, perform random rotation around center pt
// apply PBC so final coords are inside box
// also modify image flags due to PBC

if (mode == ATOM) { 
    natom = 1;
    coords[0][0] = coord[0];
    coords[0][1] = coord[1];
    coords[0][2] = coord[2];
    imageflags[0] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
} else {
    double rng = random->uniform();
    imol = 0;
    while (rng > molfrac[imol]) imol++;
    natom = onemols[imol]->natoms;
    /*
    if (dimension == 3) {
    if (orientflag) {
        r[0] = rx;
        r[1] = ry;
        r[2] = rz;
    } else {
        r[0] = random->uniform() - 0.5;
        r[1] = random->uniform() - 0.5;
        r[2] = random->uniform() - 0.5;
    }
    } else {
    r[0] = r[1] = 0.0;
    r[2] = 1.0;
    }
    double theta = random->uniform() * MY_2PI;
    MathExtra::norm3(r);
    MathExtra::axisangle_to_quat(r,theta,quat);
    MathExtra::quat_to_mat(quat,rotmat);
    for (i = 0; i < natom; i++) {
    MathExtra::matvec(rotmat,onemols[imol]->dx[i],coords[i]);
    coords[i][0] += coord[0];
    coords[i][1] += coord[1];
    coords[i][2] += coord[2];
    imageflags[i] = ((imageint) IMGMAX << IMG2BITS) |
        ((imageint) IMGMAX << IMGBITS) | IMGMAX;
    domain->remap(coords[i],imageflags[i]);
    }
}
*/
```

and add this instead:

```
// No rotation here, just apply translation directly
for (i = 0; i < natom; i++) {
coords[i][0] = onemols[imol]->dx[i][0] + coord[0];
coords[i][1] = onemols[imol]->dx[i][1] + coord[1]; 
coords[i][2] = onemols[imol]->dx[i][2] + coord[2];

imageflags[i] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
domain->remap(coords[i],imageflags[i]);
}}
```

