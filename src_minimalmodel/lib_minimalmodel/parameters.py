from dataclasses import dataclass, field
import numpy as np 
import os 

@dataclass
class MonomerParameters:

    # non-default arguments
    x: float
    angle_4_k: float 
    prefactor: float

    # decide whether you want coupled interfaces
    coupled_interfaces: bool 

    # overall scale of the monomer
    sigma: float = 1.0
    bbscale: int = 10
    particles_per_monomer: int = 7 # 6 + the new particle that prevents the inversion
    bonds_per_monomer: int = 6 # 5 + the one new bond that fixes the particle that prevents inversion - only use one for connectivity issues

    # bond information
    bond_1_k: float = 100000.0
    bond_2_k: float = 100000.0
    bond_3_k: float = 100000.0

    # angle information
    angle_1_k: float = 100000.0
    angle_2_k: float = 100000.0 
    angle_open: float = 45.0
    angle_3_eq: float = 90.0

    def __post_init__(self):
        self.bond_1_rc = self.x*self.bbscale
        self.bond_2_rc = (1-self.x)*self.bbscale
        self.bond_3_rc = 1.0*self.bbscale
        self.angle_3_k = self.prefactor*self.angle_4_k
        self.angle_open_rad = self.angle_open*np.pi/180.0
        self.angle_4_eq = 90-self.angle_open
        self.angle_4_eq_rad = self.angle_4_eq*np.pi/180.0
        self.angle_3_rc = 90 + 0.5*self.angle_open
        self.angle_4_rc = 90 - 0.5*self.angle_open
        self.delta_L = np.sqrt(2*(1-np.cos(self.angle_open_rad))*(self.x*self.bbscale)**2)
        self.delta_R = np.sqrt(2*(1-np.cos(self.angle_open_rad))*((1-self.x)*self.bbscale)**2)
        self.general_cutoff = min(self.delta_L, self.delta_R)
        self.sigma_preventinversion = 0.5*self.x*self.bbscale*np.sqrt(2*(1-np.cos(self.angle_4_eq_rad)))
        if self.coupled_interfaces:
            self.angles_per_monomer = 8
        else:
            self.angles_per_monomer = 7
@dataclass
class FilamentParameters(MonomerParameters):

    epsilon_bottom : float = 1
    epsilon_top : float = 1
    numblocks: int = 1
    buffer: float = 1.0
    cutoff_bottom: float = 1.0
    cutoff_top: float = 1.0
    conservative_cutoff_top: float = 1.0
    hydrolysis_probability: float = 0.1
    epsilon_top_final: float = 7
    hydrolysisfreq: float = 5000000

    def __post_init__(self):
        super().__post_init__()

@dataclass
class SimulationParameters(FilamentParameters):

    variables : str = None
    configfile: str = 'configuration.dat'
    inputfile: str = 'inputfile.dat'
    monomerfile: str = 'monomer.dat'
    comfile: str = 'comtreadmilling.dat'
    pythonfile: str = 'max_inter_mol_46.py'

    simulation_folder: str = './'

    xlo: float = -6000
    xhi: float = 6000
    ylo: float = -400
    yhi: float = 400
    zlo: float = -400
    zhi: float = 400

    temp: float = 1.0
    damping: float = 0.01
    dt: float = 0.001
    totaltime : float = 1000000
    dumptime: float = 5
    seed: int = 123

    run_interval: int = 100
    loops: int = 100000000
    nMonsSol: int = 1
    round_frequency: int = 100
    faci: float = 2
    facp: float = 1.3
    
    def __post_init__(self):
        self.steps = int(self.totaltime/self.dt) 
        self.dumpfreq = int(self.dumptime/self.dt)

        super().__post_init__()

    def update_directories(self, dirname):
        self.configfile = f'{dirname}/{self.configfile}'
        self.inputfile = f'{dirname}/{self.inputfile}'
        self.monomerfile = f'{dirname}/{self.monomerfile}'
        self.pythonfile = f'{dirname}/{self.pythonfile}'
