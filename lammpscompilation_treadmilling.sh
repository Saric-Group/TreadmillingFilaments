#!/bin/bash 
set -euo pipefail

# CAREFUL: MAKE SURE YOU DON'T HAVE OPENMPI ALREADY LOADED AS A MODULE!

# ensure that the environment is correctly set
env_prefix="${CONDA_PREFIX:-}"
if [[ -z "${env_prefix}" ]]; then
  echo "No active conda environment. Activate env_littleguys first."
  exit 1
fi
echo "Current environment --> ${env_prefix}"

read -p "Proceed with compilation in this env? (y/n): " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "Aborting."
  exit 1
fi

# access directory
cd lammps

# create fresh build directory
rm -rf build
mkdir build
cd build || exit

# Ensure we use the MPI tooling from this env (OpenMPI recommended).
# In your env, `conda install -c conda-forge openmpi` if you don’t have it.
export CC="$(which mpicc)"
export CXX="$(which mpicxx)"

# Python from the env
PYEXE="$(which python)"

# One variable, never reset it
LAMMPS_OPTS=()
LAMMPS_OPTS+=(-D CMAKE_C_COMPILER="${CC}")
LAMMPS_OPTS+=(-D CMAKE_CXX_COMPILER="${CXX}")
LAMMPS_OPTS+=(-D CMAKE_INSTALL_PREFIX="${env_prefix}")

# Build a shared library + Python
LAMMPS_OPTS+=(-D BUILD_SHARED_LIBS=on)
LAMMPS_OPTS+=(-D BUILD_LIB=on)
LAMMPS_OPTS+=(-D PKG_PYTHON=on)
LAMMPS_OPTS+=(-D PYTHON_EXECUTABLE="${PYEXE}")

# MPI + OpenMP
LAMMPS_OPTS+=(-D BUILD_MPI=on)
LAMMPS_OPTS+=(-D BUILD_OMP=on)
LAMMPS_OPTS+=(-D PKG_OPENMP=on)

# Your extra packages
LAMMPS_OPTS+=(-D PKG_MOLECULE=on)
LAMMPS_OPTS+=(-D PKG_REACTION=on)
LAMMPS_OPTS+=(-D PKG_EXTRA-PAIR=on)
LAMMPS_OPTS+=(-D PKG_EXTRA-FIX=on)
LAMMPS_OPTS+=(-D PKG_EXTRA-COMPUTE=on)

# Configure & build
cmake "${LAMMPS_OPTS[@]}" ../cmake
cmake --build . -j