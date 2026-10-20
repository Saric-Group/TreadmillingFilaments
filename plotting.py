from lib_littleguys.molecule_templates import generate_buildingblock_structure
import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_pretty_figure_setup():
    # if mpl seems to work with the wrong latex executable, uncomment and adapt path according to your installation
    os.environ["PATH"] = os.path.expanduser("~/texlive/2025/bin/x86_64-linux") + ":" + os.environ["PATH"]
    
    _preamble_shared = R"""
        \usepackage{graphicx}
        \DeclareMathOperator{\arcsinh}{arcsinh}
        \DeclareMathOperator{\km}{k_\mathrm{m}}
        \DeclareMathOperator{\fbi}{f_\mathrm{bi}}
        \DeclareMathOperator{\eps}{\epsilon_{\mathrm{mc}}}
        \DeclareMathOperator{\epscrit}{\epsilon_{\mathrm{mc}}^*}
        \DeclareMathOperator{\uf}{u_{\mathrm{f}}}
        \DeclareMathOperator{\kBT}{k_\mathrm{B}T}
        """[
        1:
    ]

    def mpl_rcParams_avenir():
        rcParams = {}
        rcParams["font.family"] = "sans-serif"
        rcParams["font.cursive"] = ["Optima"]
        rcParams["text.usetex"] = True
        # rcParams['text.latex.unicode']= True
        rcParams["pgf.texsystem"] = "lualatex"
        rcParams["pgf.rcfonts"] = False
        rcParams["pgf.preamble"] = (
            R"""
        \usepackage[utf8x]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{fontspec}
        \usepackage{amsmath}
        \setmainfont{Avenir}[Scale=.9]
        \renewcommand{\setmainfont}{}
        \renewcommand{\sffamily}{}
        """[
                1:
            ]
            + "\n"
            + _preamble_shared
        )
        return rcParams
    
    def rc_params_setup():
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["figure.constrained_layout.use"] = True
        mpl.rcParams.update(mpl_rcParams_avenir())
        # mpl.rcParams["pgf.texsystem"] = "lualatex"
        # mpl.rcParams["text.latex.preamble"] = mpl.rcParams['pgf.preamble'] #R"\usepackage{amsmath}\usepackage{lmodern}"
        mpl.rcParams["text.latex.preamble"] = (
            R"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        """
            + "\n"
            + _preamble_shared
        )

    rc_params_setup()
    print("Pretty figure set-up loaded.")     

def generate_open_and_closed_conformation(SimulationParameters):
    open_coords = np.zeros((SimulationParameters.particles_per_monomer, 3))
    generate_buildingblock_structure(0, open_coords, SimulationParameters, open = True)
    closed_coords = np.zeros((SimulationParameters.particles_per_monomer, 3))
    generate_buildingblock_structure(0, closed_coords, SimulationParameters, open = False)
    intermediate_coords = np.zeros((SimulationParameters.particles_per_monomer, 3))
    generate_buildingblock_structure(0, intermediate_coords, SimulationParameters, intermediate = True)
    return open_coords*SimulationParameters.bbscale, closed_coords*SimulationParameters.bbscale, intermediate_coords*SimulationParameters.bbscale

def visualize_building_blocks_ovito(SimulationParameters):

    open_coords, closed_coords = generate_open_and_closed_conformation(SimulationParameters)

def visualize_building_blocks_python(SimulationParameters, dirname, identifier):

    def generate_figure(coords, pdfname, add_circles = True, add_MD_beads = False, add_vertical = True, add_angle_open = True):
        fig, ax = plt.subplots(tight_layout = True)
        
        ax.plot([coords[0, 0], coords[2, 0]], [coords[0, 1], coords[2, 1]], color = 'k', zorder = 0, lw = 2)
        ax.plot([coords[3, 0], coords[5, 0]], [coords[3, 1], coords[5, 1]], color = 'k', zorder = 0, lw = 2)
        ax.vlines(coords[1, 0], coords[1, 1], coords[4, 1], color = 'k', zorder = 0, lw = 2)

        # add beads in the building block
        if add_circles:
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i in [1, 4, 6]:
                    if add_MD_beads:
                        circle = Circle((coords[i, 0], coords[i, 1]), radius = 1, facecolor = 'C0', edgecolor = 'k', lw = 0.5)
                        ax.add_patch(circle)
                else:
                    circle = Circle((coords[i, 0], coords[i, 1]), radius = 1, facecolor = 'C0', edgecolor = 'k', lw = 0.5)
                    ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 20)
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    # obtain the coordinates
    open_coords, closed_coords, intermediate_coords = generate_open_and_closed_conformation(SimulationParameters)
    
    # obtain the figures
    generate_figure(open_coords, f"{dirname}/open_{identifier}.pdf")
    generate_figure(closed_coords, f"{dirname}/closed_{identifier}.pdf")
    generate_figure(intermediate_coords, f"{dirname}/intermediate_{identifier}.pdf")

def visualize_binding_building_blocks_python(SimulationParameters, dirname, identifier, left_distance, right_distance):

    def generate_figure(open_coords, closed_coords, pdfname, binding_range, moved = False):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, displ_x, draw_range, int_range, colors):
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 3)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 3)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 3)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.5)
                    ax.add_patch(circle)
                if i in draw_range:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = colors, edgecolor = 'k', lw = 0.5, alpha = 0.2, ls = '--')
                    ax.add_patch(circle)

            ax.set_xlim(-24, 24)
            ax.set_ylim(-24, 24)
        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)

        factor = 1
        if moved:
            factor = 1.5
        # central seed
        draw_building_block(closed_coords, xcom_closed, [3, 5], binding_range, colors = 'k')
        # right seed
        draw_building_block(open_coords, xcom_closed-factor*(SimulationParameters.bbscale+SimulationParameters.sigma), [3], binding_range, colors = 'C0')
        # left seed
        draw_building_block(open_coords, xcom_closed+factor*(SimulationParameters.bbscale+SimulationParameters.sigma), [5], binding_range, colors = 'C1')

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    # obtain the coordinates of open and closed building blocks
    open_coords, closed_coords, _ = generate_open_and_closed_conformation(SimulationParameters)
    
    # obtain the figures
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_left.pdf", left_distance)
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_right.pdf", right_distance)
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_sigma.pdf", 1.0)

    # obtain the figures
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_left_moved.pdf", left_distance, moved = True)

def visualize_binding_building_blocks_python_trueseed(SimulationParameters, dirname, identifier, left_distance, right_distance):

    def generate_figure(open_coords, closed_coords, pdfname, binding_range, moved = False):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, displ_x, draw_range, int_range, colors):
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 3)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 3)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 3)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.5)
                    ax.add_patch(circle)
                if i in draw_range:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = colors, edgecolor = 'k', lw = 0.5, alpha = 0.2, ls = '--')
                    ax.add_patch(circle)

            ax.set_xlim(-24, 50)
            ax.set_ylim(-24, 50)
        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)

        factor = 1
        if moved:
            factor = 1.5

        # central seed
        draw_building_block(closed_coords, xcom_closed, [3, 5], binding_range, colors = 'k')
        draw_building_block(closed_coords, xcom_closed-(SimulationParameters.bbscale+SimulationParameters.sigma), [3, 5], binding_range, colors = 'k')
        
        # right seed
        draw_building_block(open_coords, xcom_closed-1.7*factor*(SimulationParameters.bbscale+SimulationParameters.sigma), [3], binding_range, colors = 'C0')
        # left seed
        draw_building_block(open_coords, xcom_closed+factor*(SimulationParameters.bbscale+SimulationParameters.sigma), [5], binding_range, colors = 'C1')

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    # obtain the coordinates of open and closed building blocks
    open_coords, closed_coords, _ = generate_open_and_closed_conformation(SimulationParameters)
    
    # obtain the figures
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_left.pdf", left_distance)
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_right.pdf", right_distance)
    generate_figure(open_coords, closed_coords, f"{dirname}/trio_{identifier}_sigma.pdf", 1.0)

    # obtain the figures
    generate_figure(open_coords, closed_coords, f"{dirname}/quartet_{identifier}_left_moved.pdf", left_distance, moved = True)

def visualize_catalytic_mechanism(SimulationParameters, dirname, identifier, left_distance, right_distance):

    def generate_figure(open_coords, closed_coords, intermediate_coords, pdfname, binding_range, moved = False, left_interface = False):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, inter_coords, displ_x, draw_range, int_range, colors):
            
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 3)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 3)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 2)

            if inter_coords is not None:
                ax.plot([inter_coords[0, 0]-displ_x, inter_coords[2, 0]-displ_x], [inter_coords[0, 1], inter_coords[2, 1]], color = colors, zorder = 0, lw = 1, ls = '--', alpha = 0.5)
                ax.plot([inter_coords[3, 0]-displ_x, inter_coords[5, 0]-displ_x], [inter_coords[3, 1], inter_coords[5, 1]], color = 'k', zorder = 0, lw = 1, ls = '--', alpha = 0.5)

                vector_of_interest_x = inter_coords[5, 0] - inter_coords[4, 0]
                vector_of_interest_y = inter_coords[5, 1] - inter_coords[4, 1]

                if colors == 'C1':
                    newpos_x = inter_coords[3, 0] + 2*vector_of_interest_x
                    newpos_y = inter_coords[3, 1] + 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'k', zorder = 0, lw = 1, ls = '--', alpha = 0.5)
                else:
                    newpos_x = inter_coords[5, 0] - 2*vector_of_interest_x
                    newpos_y = inter_coords[5, 1] - 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'k', zorder = 0, lw = 1, ls = '--', alpha = 0.5)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [1, 4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1)
                    ax.add_patch(circle)
                    #if inter_coords is not None:
                    #    circle2 = Circle((inter_coords[i, 0]-displ_x, inter_coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1, alpha = 0.2)
                    #    ax.add_patch(circle2)
                if i in draw_range:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = colors, edgecolor = 'k', lw = 0.5, alpha = 0.1, ls = '--')
                    ax.add_patch(circle)
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = 'none', edgecolor = 'k', lw = 1.0, alpha = 1, ls = ':')
                    ax.add_patch(circle)

            ax.set_xlim(-24, 24)
            ax.set_ylim(-24, 24)
        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)
    
        factor = 1
        if moved:
            factor = 1.5

        if left_interface:
            # right seed
            draw_building_block(open_coords, intermediate_coords, xcom_closed-factor*(SimulationParameters.bbscale), [], 0, colors = 'C0')
            # central seed
            draw_building_block(closed_coords, None, xcom_closed, [5], binding_range, colors = 'k')

        else:
            # left seed
            draw_building_block(open_coords, intermediate_coords, xcom_closed+factor*(SimulationParameters.bbscale), [], 0, colors = 'C1')
            # central seed
            draw_building_block(closed_coords, None, xcom_closed, [3], binding_range, colors = 'k')

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)


    def generate_figure_minimal(open_coords, pdfname):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, colors, displ_x = 0):
            
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 3)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 3)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 3)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [1, 4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1)
                    ax.add_patch(circle)

            ax.set_xlim(-24, 24)
            ax.set_ylim(-24, 24)

        fig, ax = plt.subplots(tight_layout = True)
        xcom_open, ycom_open, zcom_open = compute_com(open_coords)
        draw_building_block(open_coords, colors = 'k')
        
        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    # obtain the coordinates of open and closed building blocks
    open_coords, closed_coords, intermediate_coords = generate_open_and_closed_conformation(SimulationParameters)
    
    # obtain the figures
    generate_figure(open_coords, closed_coords, intermediate_coords, f"{dirname}/catmech_{identifier}_leftinterface.pdf", left_distance, left_interface = True)
    generate_figure(open_coords, closed_coords, intermediate_coords, f"{dirname}/catmech_{identifier}_rightinterface.pdf", left_distance)
    generate_figure_minimal(closed_coords, f"{dirname}/catmech_{identifier}_middle.pdf") 

def visualize_catalytic_mechanism_clearer(SimulationParameters, dirname, identifier, left_distance, right_distance):

    """
    Make the schematic of the catalytic mechanism clearer. For that,
    - We will compute the theta_star angles for each side
    - Place the monomers in these configurations
    - Mark where the theta_c lays
    """

    def generate_figure(open_coords, closed_coords, intermediate_coords, pdfname, binding_range, moved = False, left_interface = False):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, inter_coords, displ_x, draw_range, int_range, colors):
            
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 2)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 2)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 2)

            if inter_coords is not None:
                ax.plot([inter_coords[0, 0]-displ_x, inter_coords[2, 0]-displ_x], [inter_coords[0, 1], inter_coords[2, 1]], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)
                ax.plot([inter_coords[3, 0]-displ_x, inter_coords[5, 0]-displ_x], [inter_coords[3, 1], inter_coords[5, 1]], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)

                vector_of_interest_x = inter_coords[5, 0] - inter_coords[4, 0]
                vector_of_interest_y = inter_coords[5, 1] - inter_coords[4, 1]

                if colors == 'C1':
                    newpos_x = inter_coords[3, 0] + 2*vector_of_interest_x
                    newpos_y = inter_coords[3, 1] + 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)
                else:
                    newpos_x = inter_coords[5, 0] - 2*vector_of_interest_x
                    newpos_y = inter_coords[5, 1] - 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha =1)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1)
                    ax.add_patch(circle)
                    #if inter_coords is not None:
                    #    circle2 = Circle((inter_coords[i, 0]-displ_x, inter_coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1, alpha = 0.2)
                    #    ax.add_patch(circle2)
                if i in draw_range:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = colors, edgecolor = 'k', lw = 0.5, alpha = 0.1, ls = '--')
                    ax.add_patch(circle)
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = 'none', edgecolor = 'k', lw = 1.0, alpha = 1, ls = ':')
                    ax.add_patch(circle)

            ax.set_xlim(-24, 24)
            ax.set_ylim(-24, 24)
        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)
    
        factor = 1
        if moved:
            factor = 1.5

        if left_interface:
            # right seed
            draw_building_block(open_coords, intermediate_coords, xcom_closed-factor*(SimulationParameters.bbscale), [], 0, colors = 'C0')
            # central seed
            draw_building_block(closed_coords, None, xcom_closed, [5], binding_range, colors = 'k')

        else:
            # left seed
            draw_building_block(open_coords, intermediate_coords, xcom_closed+factor*(SimulationParameters.bbscale), [], 0, colors = 'C1')
            # central seed
            draw_building_block(closed_coords, None, xcom_closed, [3], binding_range, colors = 'k')

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    
    def distance_left_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*((1-x)*sigma)**2
    def distance_right_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*(x*sigma)**2

    # start calculation
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    # I am setting some cutoff range
    cutoff_range = 0.85*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))

    # calculation of the theta stars
    theta_star_R = np.arccos(1-(cutoff_range**2/(2*((1-SimulationParameters.x)*SimulationParameters.bbscale)**2)))
    theta_star_L = np.arccos(1-(cutoff_range**2/(2*(SimulationParameters.x*SimulationParameters.bbscale)**2)))
    print("This is theta_star_R ", theta_star_R)
    print("This is theta_star_L ", theta_star_L)

    # obtain the coordinates of open and closed building blocks
    open_coords_base, closed_coords_base, intermediate_coords_base = generate_open_and_closed_conformation(SimulationParameters)
    # obtain the building block on the right
    SimulationParameters.angle_open = theta_star_R*180/np.pi
    open_coords_R, closed_coords_R, intermediate_coords_R = generate_open_and_closed_conformation(SimulationParameters)
    print("These are the R open coordinates: ", open_coords_R)
    
    """
    fig, ax = plt.subplots()
    for i in range(len(open_coords_R)):
        ax.scatter(open_coords_R[i, 0], open_coords_R[i, 1])
    plt.show()
    plt.close()
    """
    # obtain the building block on the left
    SimulationParameters.angle_open = theta_star_L*180/np.pi
    open_coords_L, closed_coords_L, intermediate_coords_L = generate_open_and_closed_conformation(SimulationParameters)
    print("These are the L open coordinates", open_coords_L)
    
    """
    fig, ax = plt.subplots()
    for i in range(len(open_coords_L)):
        ax.scatter(open_coords_L[i, 0], open_coords_L[i, 1])
    plt.show()
    plt.close()
    """
    
    # obtain the figures
    generate_figure(open_coords_L, closed_coords_L, intermediate_coords_base, f"{dirname}/clearer_catmech_{identifier}_leftinterface.pdf", cutoff_range, left_interface = True)
    generate_figure(open_coords_R, closed_coords_R, intermediate_coords_base, f"{dirname}/clearer_catmech_{identifier}_rightinterface.pdf", cutoff_range)

def visualize_opening_angle(SimulationParameters, dirname, identifier):

    """
    Explicitly show what a building block in its open and closed conformations look like
    """

    def generate_figure(open_coords, closed_coords, intermediate_coords, pdfname):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, inter_coords, displ_x, colors):
            
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 2)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 2)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 2)

            if inter_coords is not None:
                ax.plot([inter_coords[0, 0]-displ_x, inter_coords[2, 0]-displ_x], [inter_coords[0, 1], inter_coords[2, 1]], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)
                ax.plot([inter_coords[3, 0]-displ_x, inter_coords[5, 0]-displ_x], [inter_coords[3, 1], inter_coords[5, 1]], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)

                vector_of_interest_x = inter_coords[5, 0] - inter_coords[4, 0]
                vector_of_interest_y = inter_coords[5, 1] - inter_coords[4, 1]

                if colors == 'C1':
                    newpos_x = inter_coords[3, 0] + 2*vector_of_interest_x
                    newpos_y = inter_coords[3, 1] + 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha = 1)
                else:
                    newpos_x = inter_coords[5, 0] - 2*vector_of_interest_x
                    newpos_y = inter_coords[5, 1] - 2*vector_of_interest_y
                    ax.plot([inter_coords[3, 0]-displ_x, newpos_x-displ_x], [inter_coords[3, 1], newpos_y], color = 'gray', zorder = 0, lw = 1, ls = '--', alpha =1)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [1, 4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1)
                    ax.add_patch(circle)
                    #if inter_coords is not None:
                    #    circle2 = Circle((inter_coords[i, 0]-displ_x, inter_coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1, alpha = 0.2)
                    #    ax.add_patch(circle2)

            ax.set_xlim(-24, 24)
            ax.set_ylim(-24, 24)

        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)
    
        # left seed
        draw_building_block(open_coords, intermediate_coords, 0, colors = 'k')
        # central seed
        draw_building_block(closed_coords, None, 0, colors = 'k')

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    
    def distance_left_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*((1-x)*sigma)**2
    def distance_right_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*(x*sigma)**2

    # obtain the coordinates of open and closed building blocks
    open_coords_base, closed_coords_base, intermediate_coords_base = generate_open_and_closed_conformation(SimulationParameters)

    # obtain the figures
    generate_figure(open_coords_base, closed_coords_base, intermediate_coords_base, f"{dirname}/thetac_{identifier}.pdf")



def plot_model0_equivalent_area(SimulationParameters, dirname, identifier, left_distance, right_distance):

    def generate_figure(open_coords, closed_coords, intermediate_coords, pdfname, left_distance, right_distance, moved = False):
        
        def compute_com(coords):
            return np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])

        def draw_building_block(coords, displ_x, draw_range, int_range, colors, color_range):
            
            ax.plot([coords[0, 0]-displ_x, coords[2, 0]-displ_x], [coords[0, 1], coords[2, 1]], color = colors, zorder = 0, lw = 2)
            ax.plot([coords[3, 0]-displ_x, coords[5, 0]-displ_x], [coords[3, 1], coords[5, 1]], color = colors, zorder = 0, lw = 2)
            ax.vlines(coords[1, 0]-displ_x, coords[1, 1], coords[4, 1], color = colors, zorder = 0, lw = 2)

            # add beads in the building block
            num_beads = coords.shape[0]
            for i in range(num_beads):
                if i not in [4, 6]:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1)
                    ax.add_patch(circle)
                    #if inter_coords is not None:
                    #    circle2 = Circle((inter_coords[i, 0]-displ_x, inter_coords[i, 1]), radius = 0.5, facecolor = colors, edgecolor = 'k', lw = 0.1, alpha = 0.2)
                    #    ax.add_patch(circle2)
                if i in draw_range:
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = color_range, edgecolor = 'k', lw = 0.35, alpha = 0.5, ls = '--')
                    ax.add_patch(circle)
                    circle = Circle((coords[i, 0]-displ_x, coords[i, 1]), radius = int_range, facecolor = 'none', edgecolor = 'k', lw = 1.0, alpha = 1, ls = ':')
                    ax.add_patch(circle)

            #ax.set_xlim(-24, 24)
            #ax.set_ylim(-24, 24)
        
        fig, ax = plt.subplots(tight_layout = True)
        
        xcom_closed, ycom_closed, zcom_closed = compute_com(closed_coords)
        xcom_open, ycom_open, zcom_open       = compute_com(open_coords)
    
        factor = 1.4

        # right seed
        draw_building_block(open_coords, xcom_closed-factor*(SimulationParameters.bbscale), [], 0, colors = 'C0', color_range = None)
        # central polymer (1)
        draw_building_block(closed_coords, xcom_closed, [5], right_distance, colors = 'k', color_range = 'C0')
        # central polymer (2)
        draw_building_block(closed_coords, xcom_closed+(SimulationParameters.bbscale), [], 0, colors = 'k', color_range = None)
        # central polymer (3)
        draw_building_block(closed_coords, xcom_closed+2*(SimulationParameters.bbscale), [3], left_distance, colors = 'k', color_range = 'C1')
        # left seed
        draw_building_block(open_coords, xcom_closed+SimulationParameters.bbscale*(2+factor), [], 0, colors = 'C1', color_range = None)

        ax.set_aspect('equal')
        ax.axis('off')

        fig.savefig(f"{pdfname}", transparent = True)

    # obtain the coordinates of open and closed building blocks
    open_coords, closed_coords, intermediate_coords = generate_open_and_closed_conformation(SimulationParameters)
    
    # obtain the figures
    generate_figure(open_coords, closed_coords, intermediate_coords, f"{dirname}/model0_areaequivalents_{identifier}_.pdf", left_distance, right_distance)


def plot_internal_energy_landscape(SimulationParameters, savepdf):

    # mapping from simulation parameters to theory parameters
    # k ---> elastic constant of the open state
    # kappa*k --> elastic constant of the closed state
    # k3 --> elastic constant closed
    # k4 --> elastic constant open
    # therefore, from simulations to theory
    # prefactor --> kappa
    # k4 --> k
    kappa = SimulationParameters.prefactor
    k     = SimulationParameters.angle_4_k
    
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    theta = np.linspace(-0.25, angle_open_rad*1.5, 1000)
    scale = 1/8 * k * angle_open_rad**2

    # add energy from each torsion angle
    cutoff_angle = 0.5*angle_open_rad
    torsion_open = 0.5*k*((theta[theta>=cutoff_angle]-angle_open_rad)**2-cutoff_angle**2)    
    torsion_closed = 0.5*k*kappa*(theta[theta<=cutoff_angle]**2 - cutoff_angle**2)
    
    # generate figure
    fig, ax = plt.subplots(tight_layout = True, figsize = (3.2, 2))
    ax.plot(theta[theta<=cutoff_angle], torsion_closed/scale, color = 'k', label = 'Free')
    ax.plot(theta[theta>=cutoff_angle], torsion_open/scale, color = 'k')

    ax.plot(theta[theta>=cutoff_angle], torsion_open/scale, color = '#f562f9', ls = '--', lw = 1)
    ax.plot(theta[theta<=cutoff_angle], 3*torsion_closed/scale, color = '#f562f9', ls = '--', lw = 1)

    #ax.plot(theta[theta<=cutoff_angle]+2*cutoff_angle, torsion_closed/scale, color = 'purple', ls = '--', lw = 1)
    #ax.plot(theta[theta>=cutoff_angle]-2*cutoff_angle, torsion_open/scale, color = 'purple', ls = '--', lw = 1, label = 'Assembled')

    #index1 = np.where(np.abs(theta)< 1e-3)[0][0]
    #index2 = np.where(np.abs(theta[theta>=cutoff_angle]-angle_open_rad)< 1e-3)[0][0]
    #ax.scatter(theta[index1], torsion_closed[index1], color = 'C2', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = '^')
    #ax.scatter(theta[theta>=cutoff_angle][index2], torsion_open[index2], color = 'C0', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'v')
    #ax.scatter(cutoff_angle, 0, color = 'r', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'h')

    ax.axvline(x = 0, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = cutoff_angle, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = angle_open_rad, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.set_xticks([0, cutoff_angle, angle_open_rad])
    ax.set_xticklabels([r'$\theta_P$', r'$\theta_C$', r'$\theta_S$'])
    ax.set_xlabel('Opening angle '+r'$\theta$')
    ax.set_ylabel(r'$U_I(\theta)/u_0$')
    #ax.legend(loc = 'lower left')

    # ADD A CURVE WHERE KAPPA IS LARGER THAN ONE JUST TO CONTRAST
    kappa = 2#SimulationParameters.prefactor
    k     = 50#.25*SimulationParameters.angle_4_k
    scale = 1/8 * k * angle_open_rad**2

    # add energy from each torsion angle
    cutoff_angle = 0.5*angle_open_rad
    torsion_open = 0.5*k*((theta[theta>=cutoff_angle]-angle_open_rad)**2-cutoff_angle**2)    
    torsion_closed = 0.5*k*kappa*(theta[theta<=cutoff_angle]**2 - cutoff_angle**2)

    # generate figure
    #ax.plot(theta[theta<=cutoff_angle], torsion_closed/scale, color = 'gray', lw = 1, ls = '--')
    #ax.plot(theta[theta>=cutoff_angle], torsion_open/scale, color = 'gray', lw = 1, ls = '--')

    #ax.xaxis.set_label_position('top') 
    fig.savefig(savepdf, transparent = True)

def plot_internal_energy_landscape_with_hydrolysis(SimulationParameters, savepdf):

    def distance_left_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*((1-x)*sigma)**2
    def distance_right_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*(x*sigma)**2

    # mapping from simulation parameters to theory parameters
    # k ---> elastic constant of the open state
    # kappa*k --> elastic constant of the closed state
    # k3 --> elastic constant closed
    # k4 --> elastic constant open
    # therefore, from simulations to theory
    # prefactor --> kappa
    # k4 --> k

    # .....................................
    # PLOT DATA WITH THE FIRST EPSILON
    kappa = SimulationParameters.prefactor
    k     = SimulationParameters.angle_4_k
    epsilon = 0.05*k
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    theta = np.linspace(-0.1, angle_open_rad*1.2, 1000)
    cutoff_range = 0.9*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
    
    print(f"Open angle rad: {angle_open_rad}")
    print(f"CUTOFF RANGE: {cutoff_range}")

    # add energy from each torsion angle
    cutoff_angle = 0.5*angle_open_rad
    torsion_open = 0.5*k*((theta[theta>=cutoff_angle]-angle_open_rad)**2-cutoff_angle**2)    
    torsion_closed = 0.5*k*kappa*(theta[theta<=cutoff_angle]**2 - cutoff_angle**2)

    left_binding = np.sqrt(distance_left_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    right_binding = np.sqrt(distance_right_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    left_interaction = 0.5*epsilon*np.heaviside(cutoff_range-left_binding, 0)*(left_binding**2-cutoff_range**2)
    right_interaction = 0.5*epsilon*np.heaviside(cutoff_range-right_binding, 0)*(right_binding**2-cutoff_range**2)

    # generate figure
    fig, ax = plt.subplots(tight_layout = True, figsize = (3, 2))
    ax.plot(theta[theta<=cutoff_angle], torsion_closed, color = 'k', label = r'$U_I$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open, color = 'k')

    ax.plot(theta[theta<=cutoff_angle], torsion_closed + left_interaction[theta<=cutoff_angle], color = 'C1', label = r'$U_B|_{L}$', ls = '--', alpha = 0.8)
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + left_interaction[theta>=cutoff_angle], color = 'C1', ls = '--', alpha = 0.8)

    ax.plot(theta[theta<=cutoff_angle], torsion_closed + right_interaction[theta<=cutoff_angle], color = 'C0', label = r'$U_B|_{R}$', ls = '--', alpha = 0.8)
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + right_interaction[theta>=cutoff_angle], color = 'C0', ls = '--', alpha = 0.8)

    #index1 = np.where(np.abs(theta)< 1e-3)[0][0]
    #index2 = np.where(np.abs(theta[theta>=cutoff_angle]-angle_open_rad)< 1e-3)[0][0]
    #ax.scatter(theta[index1], torsion_closed[index1], color = 'C2', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = '^')
    #ax.scatter(theta[theta>=cutoff_angle][index2], torsion_open[index2], color = 'C0', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'v')
    #ax.scatter(cutoff_angle, 0, color = 'r', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'h')
    
    theta_star_R = np.arccos(1-(cutoff_range**2/(2*((1-SimulationParameters.x)*SimulationParameters.bbscale)**2)))
    theta_star_L = np.arccos(1-(cutoff_range**2/(2*(SimulationParameters.x*SimulationParameters.bbscale)**2)))
    #ax.axvline(x = theta_star_L, ls = '--', lw = 0.5, zorder = 0, color = 'C1')
    #ax.axvline(x = theta_star_R, ls = '--', lw = 0.5, zorder = 0, color = 'C0')
    
    ax.axvline(x = 0, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = cutoff_angle, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = angle_open_rad, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.set_xticks([0, cutoff_angle, angle_open_rad])
    ax.set_xticklabels([r'$\theta_P$',r'$\theta_C$', r'$\theta_S$'])
    ax.set_xlabel('Reaction coordinate '+r'$\theta$')
    ax.set_ylabel(r'$U(\theta)/k_B T$')
    ax.legend(fontsize = 6, loc = 'lower right')

    print("Theta star R: ", theta_star_R)
    print("Theta star L: ", theta_star_L) 

    # .....................................
    # DECREASE THE EPSILON
    epsilon = 0.01*k
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    theta = np.linspace(-0.1, angle_open_rad*1.2, 1000)
    cutoff_range = 0.9*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
    
    # add energy from each torsion angle
    cutoff_angle = 0.5*angle_open_rad
    torsion_open = 0.5*k*((theta[theta>=cutoff_angle]-angle_open_rad)**2-cutoff_angle**2)    
    torsion_closed = 0.5*k*kappa*(theta[theta<=cutoff_angle]**2 - cutoff_angle**2)

    left_binding = np.sqrt(distance_left_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    right_binding = np.sqrt(distance_right_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    left_interaction = 0.5*epsilon*np.heaviside(cutoff_range-left_binding, 0)*(left_binding**2-cutoff_range**2)
    right_interaction = 0.5*epsilon*np.heaviside(cutoff_range-right_binding, 0)*(right_binding**2-cutoff_range**2)

    # generate figure
    ax.plot(theta[theta<=cutoff_angle], torsion_closed + left_interaction[theta<=cutoff_angle], color = 'C1', label = r'$U_H|_{L}$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + left_interaction[theta>=cutoff_angle], color = 'C1')

    ax.plot(theta[theta<=cutoff_angle], torsion_closed + right_interaction[theta<=cutoff_angle], color = 'C0', label = r'$U_H|_{R}$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + right_interaction[theta>=cutoff_angle], color = 'C0')

    #index1 = np.where(np.abs(theta)< 1e-3)[0][0]
    #index2 = np.where(np.abs(theta[theta>=cutoff_angle]-angle_open_rad)< 1e-3)[0][0]
    #ax.scatter(theta[index1], torsion_closed[index1], color = 'C2', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = '^')
    #ax.scatter(theta[theta>=cutoff_angle][index2], torsion_open[index2], color = 'C0', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'v')
    #ax.scatter(cutoff_angle, 0, color = 'r', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'h')
    
    theta_star_R = np.arccos(1-(cutoff_range**2/(2*((1-SimulationParameters.x)*SimulationParameters.bbscale)**2)))
    theta_star_L = np.arccos(1-(cutoff_range**2/(2*(SimulationParameters.x*SimulationParameters.bbscale)**2)))
    #ax.axvline(x = theta_star_L, ls = '--', lw = 0.5, zorder = 0, color = 'C1')
    #ax.axvline(x = theta_star_R, ls = '--', lw = 0.5, zorder = 0, color = 'C0')
    
    ax.axvline(x = 0, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = cutoff_angle, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = angle_open_rad, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.set_xticks([0, theta_star_R, cutoff_angle, theta_star_L, angle_open_rad])
    ax.set_xticklabels([r'$\theta_P$', r'$\theta_L^*$', r'$\theta_C$', r'$\theta_R^*$', r'$\theta_S$'])
    ax.set_xlabel('Reaction coordinate '+r'$\theta$')
    ax.set_ylabel(r'$U(\theta)/k_B T$')
    ax.legend(fontsize = 6, loc = 'lower right')

    print("Theta star R: ", theta_star_R)
    print("Theta star L: ", theta_star_L) 

    #ax.xaxis.set_label_position('top') 
    fig.savefig(savepdf, transparent = True)

def plot_switch_amplitudes_function_x(savepdf):

    fig, ax = plt.subplots(tight_layout = True, figsize = (3.2, 2))

    x = np.linspace(0, 1.0, 1000)
    alpha = lambda x: (1-x)/x
    ax.plot(x, alpha(x), color = 'k')
    ax.plot(x, np.ones(1000), ls = '--', color = 'k', lw = 0.2)
    ax.axvline(x = 0.5, ls = '--', color = 'k', lw = 0.2)

    ax.set_xlabel(r'Support position '+r'$x/x_0$')
    ax.set_ylabel(r'Asymmetry '+r'$\alpha = \delta_R/\delta_L$')
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.savefig(savepdf, transparent = True)

def plot_building_block_binding(SimulationParameters, savepdf):

    def cosine_square_potential(r, epsilon = 1, sigma = 1, rc = 1.2):

        f = -epsilon*np.cos(0.5*np.pi*(r-sigma)/(rc-sigma))**2
        f[r<sigma] = epsilon*((sigma/r[r<sigma])**12 - 2*(sigma/r[r<sigma])**6)
        f[r>rc] = 0
        return f

    rc = 1.2
    r = np.linspace(0, 2, 1000)
    f1 = cosine_square_potential(r, rc = rc)
    f2 = cosine_square_potential(r, epsilon = 0.5, rc = rc)

    fig, ax = plt.subplots(tight_layout = True, figsize = (3.2, 2))

    ax.plot(r, f1, color = 'k', label = 'Non-hydrolyzed')
    # [COMMENT HERE TO REMOVE DASHED LINE]
    ax.plot(r, f2, color = 'k', ls = '--', label = 'Hydrolyzed')

    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0.85, 1.3)
    ax.set_xticks([1.0, rc])
    ax.set_xticklabels([r'$\sigma$', r'$r_c$'])
    ax.set_xlabel('Binding site distance '+r'$r/\sigma$')
    ax.set_ylabel(r'$U(r)/\epsilon$')
    ax.set_yticks([-1, 0, 1])
    ax.plot(r, np.zeros(1000), color = 'k', lw = 1)
    ax.legend()
    
    fig.savefig(savepdf, transparent = True)

def left_distance(theta_open, x, sigma):
    return np.sqrt(2*(1-np.cos(theta_open*np.pi/180.0))*(x*sigma)**2)

def right_distance(theta_open, x, sigma):
    return np.sqrt(2*(1-np.cos(theta_open*np.pi/180.0))*((1-x)*sigma)**2)

def add_colorbar(fig, ax, cs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(cs, cax = cax, orientation = 'vertical')
    
def produce_heatmap_binding(sigma, theta_desired, x_desired, pdfname_general, pdfname_panel):
    theta_open = np.linspace(0, 80, 100)
    x = np.linspace(0.1, 0.5, 100)

    theta_mesh, x_mesh = np.meshgrid(theta_open, x)
    left_distance_mesh = left_distance(theta_mesh, x_mesh, sigma)
    right_distance_mesh = right_distance(theta_mesh, x_mesh, sigma)

    fig, ax = plt.subplots(tight_layout = True, ncols = 3, figsize = (10, 3))
    fig_panel, ax_panel = plt.subplots(tight_layout = True, figsize = (3.2, 2))

    CSL = ax[0].contourf(theta_mesh, x_mesh, left_distance_mesh, levels = 50)
    CSL_lines = ax[0].contour(CSL, levels = (1, 3, 5), colors = 'w')
    CSR = ax[1].contourf(theta_mesh, x_mesh, right_distance_mesh, levels = 50)
    CSR_lines = ax[1].contour(CSR, levels = (1, 3, 5), colors = 'w')

    add_colorbar(fig, ax[0], CSL)
    add_colorbar(fig, ax[1], CSR)

    ax[0].clabel(CSL_lines, fmt='%2.1f', colors='w', fontsize=14)
    ax[1].clabel(CSR_lines, fmt='%2.1f', colors='w', fontsize=14)

    CSL_lines = ax[2].contour(CSL, levels = (1, 2, 3, 4, 5), colors = 'r')
    CSR_lines = ax[2].contour(CSR, levels = (1, 2, 3, 4, 5), colors = 'g')
    ax[2].clabel(CSL_lines, fmt='%2.1f', colors='r', fontsize=14)
    ax[2].clabel(CSR_lines, fmt='%2.1f', colors='g', fontsize=14)

    CSL_lines = ax_panel.contour(CSL, levels = (1, 2, 3, 4, 5), colors = 'C0', lw = 1)
    CSR_lines = ax_panel.contour(CSR, levels = (1, 2, 3, 4, 5), colors = 'C1', lw = 1)
    ax_panel.clabel(CSL_lines, fmt='%2.0f', colors='C0', fontsize=14)
    ax_panel.clabel(CSR_lines, fmt='%2.0f', colors='C1', fontsize=14)
    
    # Create legend handles
    handles = [Line2D([0], [0], color='C0', label=r'$\delta_L$'), Line2D([0], [0], color='C1', label=r'$\delta_R$') ]
    ax_panel.legend(handles=handles )

    ax[0].set_title("Left BS to seed")
    ax[1].set_title("Right BS to seed")
    ax[2].set_title("L and R combined")

    for i in range(3):
        ax[i].set_xlabel("Substrate angle " + r"$\theta_O$")
        ax[i].set_ylabel("Hinge position "+ r"$x$")
        ax[i].scatter(theta_desired, x_desired, marker = "*", color = "k")

    ax_panel.set_xlabel("Substrate angle " + r"$\theta_S$")
    ax_panel.set_ylabel("Hinge position "+ r"$x$")
    #ax_panel.scatter(30, 0.2, marker = "h", color = "green", zorder = 1000, s = 100)
    #ax_panel.scatter(30, 0.5, marker = "s", color = "red", zorder = 1000, s = 100)

    fig.savefig(pdfname_general, transparent = True)
    fig_panel.savefig(pdfname_panel, transparent = True)

    print(f"FOR BB {sigma}")
    print(" - LEFT DESIRED: ", left_distance(theta_desired, x_desired, sigma))
    print(" - RIGHT DESIRED: ", right_distance(theta_desired, x_desired, sigma))
    print(f"In between: {(right_distance(theta_desired, x_desired, sigma) + left_distance(theta_desired, x_desired, sigma))*0.5}")

def plot_binding_energy_landscape(SimulationParameters, savepdf, factor_range):

    def distance_left_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*((1-x)*sigma)**2
    def distance_right_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*(x*sigma)**2

    # mapping from simulation parameters to theory parameters
    # k ---> elastic constant of the open state
    # kappa*k --> elastic constant of the closed state
    # k3 --> elastic constant closed
    # k4 --> elastic constant open
    # therefore, from simulations to theory
    # prefactor --> kappa
    # k4 --> k

    kappa = SimulationParameters.prefactor
    k     = SimulationParameters.angle_4_k
    epsilon = 0.05*k
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    theta = np.linspace(-0.1, angle_open_rad*1.2, 1000)
    cutoff_range = 0.9*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
    
    print(f"Open angle rad: {angle_open_rad}")
    print(f"CUTOFF RANGE: {cutoff_range}")

    # add energy from each torsion angle
    cutoff_angle = 0.5*angle_open_rad
    torsion_open = 0.5*k*((theta[theta>=cutoff_angle]-angle_open_rad)**2-cutoff_angle**2)    
    torsion_closed = 0.5*k*kappa*(theta[theta<=cutoff_angle]**2 - cutoff_angle**2)

    left_binding = np.sqrt(distance_left_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    right_binding = np.sqrt(distance_right_interface_binding(theta, SimulationParameters.x, SimulationParameters.bbscale))
    left_interaction = 0.5*epsilon*np.heaviside(cutoff_range-left_binding, 0)*(left_binding**2-cutoff_range**2)
    right_interaction = 0.5*epsilon*np.heaviside(cutoff_range-right_binding, 0)*(right_binding**2-cutoff_range**2)

    # generate figure
    fig, ax = plt.subplots(tight_layout = True, figsize = (3, 2))
    ax.plot(theta[theta<=cutoff_angle], torsion_closed, color = 'k', label = r'$U_I$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open, color = 'k')

    ax.plot(theta[theta<=cutoff_angle], torsion_closed + left_interaction[theta<=cutoff_angle], color = 'C1', label = r'$U_B|_{L}$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + left_interaction[theta>=cutoff_angle], color = 'C1')

    ax.plot(theta[theta<=cutoff_angle], torsion_closed + right_interaction[theta<=cutoff_angle], color = 'C0', label = r'$U_B|_{R}$')
    ax.plot(theta[theta>=cutoff_angle], torsion_open   + right_interaction[theta>=cutoff_angle], color = 'C0')

    #index1 = np.where(np.abs(theta)< 1e-3)[0][0]
    #index2 = np.where(np.abs(theta[theta>=cutoff_angle]-angle_open_rad)< 1e-3)[0][0]
    #ax.scatter(theta[index1], torsion_closed[index1], color = 'C2', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = '^')
    #ax.scatter(theta[theta>=cutoff_angle][index2], torsion_open[index2], color = 'C0', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'v')
    #ax.scatter(cutoff_angle, 0, color = 'r', edgecolor = 'k', lw = 0.5, zorder = 1000, marker = 'h')
    
    theta_star_R = np.arccos(1-(cutoff_range**2/(2*((1-SimulationParameters.x)*SimulationParameters.bbscale)**2)))
    theta_star_L = np.arccos(1-(cutoff_range**2/(2*(SimulationParameters.x*SimulationParameters.bbscale)**2)))
    ax.axvline(x = theta_star_L, ls = '--', lw = 0.5, zorder = 0, color = 'C1')
    ax.axvline(x = theta_star_R, ls = '--', lw = 0.5, zorder = 0, color = 'C0')
    
    ax.axvline(x = 0, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = cutoff_angle, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.axvline(x = angle_open_rad, ls = '--', lw = 0.5, color = 'k', zorder = 0)
    ax.set_xticks([0, theta_star_R, cutoff_angle, theta_star_L, angle_open_rad])
    ax.set_xticklabels([r'$\theta_P$', r'$\theta_L^*$', r'$\theta_C$', r'$\theta_R^*$', r'$\theta_S$'])
    ax.set_xlabel('Opening angle '+r'$\theta$')
    ax.set_ylabel(r'$U(\theta)/k_B T$')
    ax.legend(fontsize = 6, loc = 'lower right')

    print("Theta star R: ", theta_star_R)
    print("Theta star L: ", theta_star_L) 

    #ax.xaxis.set_label_position('top') 
    fig.savefig(savepdf, transparent = True)

def plot_test(SimulationParameters, savepdf):

    def distance_left_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*((1-x)*sigma)**2
    def distance_right_interface_binding(theta, x, sigma):
        return 2*(1-np.cos(theta))*(x*sigma)**2

    def angle_left(q, x, scale):
        return np.arccos(1-(q**2 / (2*(x*scale)**2)  ))
    def angle_right(q, x, scale):
        return np.arccos(1-(q**2/ (2*((1-x)*scale)**2) ))

    kappa = SimulationParameters.prefactor
    k     = SimulationParameters.angle_4_k
    epsilon = 0.05*k
    angle_open_rad = SimulationParameters.angle_open*np.pi/180.0
    theta = np.linspace(-0.1, angle_open_rad*1.2, 1000)
    lower_cutoff = 0*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
    upper_cutoff = 0.8*np.sqrt(distance_left_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
        
    # generate figure
    fig, ax = plt.subplots(tight_layout = True, figsize = (2.5, 2))
    cutoff_range = 0.9*np.sqrt(distance_right_interface_binding(angle_open_rad, SimulationParameters.x, SimulationParameters.bbscale))
    theta_star_L = np.arccos(1-(cutoff_range**2/(2*((1-SimulationParameters.x)*SimulationParameters.bbscale)**2)))
    theta_star_R = np.arccos(1-(cutoff_range**2/(2*(SimulationParameters.x*SimulationParameters.bbscale)**2)))
    ax.scatter(cutoff_range,theta_star_R, zorder = 1000, marker = "d", edgecolor = 'k', lw = 0.5, color = 'C0')
    ax.scatter(cutoff_range,theta_star_L, zorder = 1000, marker = "d", edgecolor = 'k', lw = 0.5, color = 'C1')

    qc = np.linspace(lower_cutoff, upper_cutoff, 100)
    aleft = angle_left(qc, SimulationParameters.x, SimulationParameters.bbscale)
    aright = angle_right(qc, SimulationParameters.x, SimulationParameters.bbscale)

    ax.plot(qc, aleft, label = r'$\theta^*_R$')
    ax.plot(qc, aright, label = r'$\theta^*_L$')
    ax.plot(qc, np.ones(len(qc))*angle_open_rad/2, label = r'$\theta_c$', color = 'k', lw = 1, ls = '--')

    ax.set_xlabel('Interaction range '+r'$q_c/x_0$')
    ax.set_ylabel(r'Cutoff angles '+r'$\theta^*$')
    #ax.legend(fontsize = 8, loc = 'lower right')
    ax.set_xlim(0, upper_cutoff)
    ax.set_ylim(0, angle_open_rad)
    ax.set_yticks([0, angle_open_rad/2, angle_open_rad])
    ax.set_yticklabels([0, r'$\theta_c$', r'$\theta_s$'])

    region_1 = np.sqrt(distance_right_interface_binding(angle_open_rad/2, SimulationParameters.x, SimulationParameters.bbscale))
    region_2 = np.sqrt(distance_left_interface_binding(angle_open_rad/2, SimulationParameters.x, SimulationParameters.bbscale))
    ax.axvline(x = region_1, ls = '--', color = 'k', lw = 0.5)
    ax.axvline(x = region_2, ls = '--', color = 'k', lw = 0.5)
    ax.set_xticks([0, region_1, region_2])
    ax.set_xticklabels([0, r'$q_R$', r'$q_L$'])
    ax.fill_between([0, region_1], 0, angle_open_rad, alpha = 0.1, color = 'k')
    ax.fill_between([region_1, region_2], 0, angle_open_rad, alpha = 0.5, color = 'C0')
    ax.fill_between([region_2, upper_cutoff], 0, angle_open_rad, alpha = 0.5, color = 'C2')

    #ax.xaxis.set_label_position('top') 
    fig.savefig(savepdf, transparent = True)

def plot_diagram_hinge_chevron_triangle(savepdf):

    fig, ax = plt.subplots(tight_layout = False)
    
    # triangle binding from the left (left interface)
    coordinates_triangle_left = np.array([[0, 0], [-1.0, 0], [-2.0, 0], [-0.5, np.sqrt(3)/2], [-1.5, np.sqrt(3)/2], [-1, np.sqrt(3)] , [-2, np.sqrt(3)], [-0.5, -np.sqrt(3)/2]])
    ax.plot([coordinates_triangle_left[0, 0], coordinates_triangle_left[1, 0]], [coordinates_triangle_left[0, 1], coordinates_triangle_left[1, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[1, 0], coordinates_triangle_left[2, 0]], [coordinates_triangle_left[1, 1], coordinates_triangle_left[2, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[0, 0], coordinates_triangle_left[3, 0]], [coordinates_triangle_left[0, 1], coordinates_triangle_left[3, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[1, 0], coordinates_triangle_left[3, 0]], [coordinates_triangle_left[1, 1], coordinates_triangle_left[3, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[1, 0], coordinates_triangle_left[4, 0]], [coordinates_triangle_left[1, 1], coordinates_triangle_left[4, 1]], color = 'C1', lw = 2, ls = ":")
    ax.plot([coordinates_triangle_left[2, 0], coordinates_triangle_left[4, 0]], [coordinates_triangle_left[2, 1], coordinates_triangle_left[4, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[3, 0], coordinates_triangle_left[4, 0]], [coordinates_triangle_left[3, 1], coordinates_triangle_left[4, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[3, 0], coordinates_triangle_left[5, 0]], [coordinates_triangle_left[3, 1], coordinates_triangle_left[5, 1]], color = 'C1', lw = 2)
    ax.plot([coordinates_triangle_left[4, 0], coordinates_triangle_left[5, 0]], [coordinates_triangle_left[4, 1], coordinates_triangle_left[5, 1]], color = 'C1', lw = 2)

    #ax.plot([coordinates_triangle_left[5, 0], coordinates_triangle_left[6, 0]], [coordinates_triangle_left[5, 1], coordinates_triangle_left[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_left[4, 0], coordinates_triangle_left[6, 0]], [coordinates_triangle_left[4, 1], coordinates_triangle_left[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_left[0, 0], coordinates_triangle_left[7, 0]], [coordinates_triangle_left[0, 1], coordinates_triangle_left[7, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_left[7, 0], coordinates_triangle_left[1, 0]], [coordinates_triangle_left[7, 1], coordinates_triangle_left[1, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    for i in range(6):
        ax.scatter(coordinates_triangle_left[i, 0], coordinates_triangle_left[i, 1], color = 'C1', s= 100, zorder = 100)
    

    # triangle binding from the right (right interface)
    coordinates_triangle_right = coordinates_triangle_left.copy() + np.array([2.5, np.sqrt(3)/2])
    ax.plot([coordinates_triangle_right[0, 0], coordinates_triangle_right[1, 0]], [coordinates_triangle_right[0, 1], coordinates_triangle_right[1, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[1, 0], coordinates_triangle_right[2, 0]], [coordinates_triangle_right[1, 1], coordinates_triangle_right[2, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[0, 0], coordinates_triangle_right[3, 0]], [coordinates_triangle_right[0, 1], coordinates_triangle_right[3, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[1, 0], coordinates_triangle_right[3, 0]], [coordinates_triangle_right[1, 1], coordinates_triangle_right[3, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[1, 0], coordinates_triangle_right[4, 0]], [coordinates_triangle_right[1, 1], coordinates_triangle_right[4, 1]], color = 'C0', lw = 2, ls = ':')
    ax.plot([coordinates_triangle_right[2, 0], coordinates_triangle_right[4, 0]], [coordinates_triangle_right[2, 1], coordinates_triangle_right[4, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[3, 0], coordinates_triangle_right[4, 0]], [coordinates_triangle_right[3, 1], coordinates_triangle_right[4, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[3, 0], coordinates_triangle_right[5, 0]], [coordinates_triangle_right[3, 1], coordinates_triangle_right[5, 1]], color = 'C0', lw = 2)
    ax.plot([coordinates_triangle_right[4, 0], coordinates_triangle_right[5, 0]], [coordinates_triangle_right[4, 1], coordinates_triangle_right[5, 1]], color = 'C0', lw = 2)
    for i in range(6):
        ax.scatter(coordinates_triangle_right[i, 0], coordinates_triangle_right[i, 1], color = 'C0', s= 100, zorder = 100)
    
    #ax.plot([coordinates_triangle_right[5, 0], coordinates_triangle_right[6, 0]], [coordinates_triangle_right[5, 1], coordinates_triangle_right[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_right[4, 0], coordinates_triangle_right[6, 0]], [coordinates_triangle_right[4, 1], coordinates_triangle_right[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_right[0, 0], coordinates_triangle_right[7, 0]], [coordinates_triangle_right[0, 1], coordinates_triangle_right[7, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_triangle_right[7, 0], coordinates_triangle_right[1, 0]], [coordinates_triangle_right[7, 1], coordinates_triangle_right[1, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')

    # structure chevron
    coordinates_chevron = np.array([[0, 0], [1.0, 0], [-0.5, np.sqrt(3)/2], [0.5, np.sqrt(3)/2], [0, np.sqrt(3)], [1, np.sqrt(3)], [0.5, -np.sqrt(3)/2], [0.5, 3*np.sqrt(3)/2]])
    ax.plot([coordinates_chevron[0, 0], coordinates_chevron[1, 0]], [coordinates_chevron[0, 1], coordinates_chevron[1, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[0, 0], coordinates_chevron[2, 0]], [coordinates_chevron[0, 1], coordinates_chevron[2, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[0, 0], coordinates_chevron[3, 0]], [coordinates_chevron[0, 1], coordinates_chevron[3, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[1, 0], coordinates_chevron[3, 0]], [coordinates_chevron[1, 1], coordinates_chevron[3, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[2, 0], coordinates_chevron[3, 0]], [coordinates_chevron[2, 1], coordinates_chevron[3, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[2, 0], coordinates_chevron[4, 0]], [coordinates_chevron[2, 1], coordinates_chevron[4, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[3, 0], coordinates_chevron[4, 0]], [coordinates_chevron[3, 1], coordinates_chevron[4, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[3, 0], coordinates_chevron[5, 0]], [coordinates_chevron[3, 1], coordinates_chevron[5, 1]], color = 'k', lw = 2)
    ax.plot([coordinates_chevron[4, 0], coordinates_chevron[5, 0]], [coordinates_chevron[4, 1], coordinates_chevron[5, 1]], color = 'k', lw = 2)
    
    for i in range(6):
        ax.scatter(coordinates_chevron[i, 0], coordinates_chevron[i, 1], color = 'k', s= 100)
    
    #ax.plot([coordinates_chevron[0, 0], coordinates_chevron[6, 0]], [coordinates_chevron[0, 1], coordinates_chevron[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_chevron[1, 0], coordinates_chevron[6, 0]], [coordinates_chevron[1, 1], coordinates_chevron[6, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_chevron[4, 0], coordinates_chevron[7, 0]], [coordinates_chevron[4, 1], coordinates_chevron[7, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')
    #ax.plot([coordinates_chevron[5, 0], coordinates_chevron[7, 0]], [coordinates_chevron[5, 1], coordinates_chevron[7, 1]], color = 'k', lw = 1, alpha = 0.2, ls = '--')

    circle_left = Circle((coordinates_triangle_left[3, 0], coordinates_triangle_left[3, 1]), radius = 1, facecolor = 'C1', edgecolor = 'C1', lw = 0.5, alpha = 0.3)
    circle_right = Circle((coordinates_triangle_right[2, 0], coordinates_triangle_right[2, 1]), radius = 1, facecolor = 'C0', edgecolor = 'C0', lw = 0.5, alpha = 0.3)

    ax.add_patch(circle_left)
    ax.add_patch(circle_right)
    
    ax.scatter(coordinates_triangle_left[5, 0], coordinates_triangle_left[5, 1], s = 200, color = 'C1', edgecolor = 'k', zorder = 100)
    ax.scatter(coordinates_triangle_right[1, 0], coordinates_triangle_right[1, 1], s = 200, color = 'C0', edgecolor = 'k', zorder = 100)

    ax.set_aspect('equal')
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(savepdf, transparent = True)


def produce_internal_landscape_colloids(savepdf):

    def cosine_square_potential(x, epsilon, rc, sigma = 1):
        potential = -epsilon*np.cos(0.5*np.pi*(x-sigma)/(rc-sigma))**2
        potential[x<sigma] = -epsilon + epsilon*((sigma/x[x<sigma])**12 - 2*(sigma/x[x<sigma])**6+1)
        potential[x>rc] = 0
        return potential

    def potential_switch(x, epsilon1, epsilon2, rc):
        pot_form_1 = cosine_square_potential(x, epsilon1, rc)
        x2 = x/np.tan(np.arcsin(x*0.5))
        print(x2)
        pot_form_2 = cosine_square_potential(x2, epsilon2, rc)
        return pot_form_1 + pot_form_2

    # generate figure
    fig, ax = plt.subplots(tight_layout = True, figsize = (3, 2.2))
    
    r13 = np.linspace(1.0, np.sqrt(3), 100)

    rc = 1.2
    epsilon1 = 1
    epsilon2 = 2
    potential = potential_switch(r13, epsilon1, epsilon2, rc)
    
    ax.set_xlim(1, np.sqrt(3))
    ax.set_yticks([0, -epsilon1, -epsilon2])
    ax.set_yticklabels(['0', r'$\varepsilon_1$', r'$\varepsilon_2$'])

    ax.plot(r13, potential, lw = 1, color = 'k')
    ax.plot(r13, np.zeros(len(r13)), lw = 0.5, color = 'k', ls = '--')
    ax.set_ylabel('Potential energy '+r'$U/k_B T$')
    ax.set_xlabel('Distance 1-3 '+r'$r_{1,3}/\sigma$')

    ax.axvline(x = rc, ls = ':', lw = 0.5, color = 'k')
    ax.axvline(x = np.sqrt(4-rc**2), color = 'k', ls = ':', lw = 0.5)
    #ax.fill_between([1, rc], -1.1*epsilon2, 1.05, color = 'pink', alpha = 0.4)
    #ax.fill_between([np.sqrt(4-rc**2), np.sqrt(3)], -1.1*epsilon2, 1.05, color = 'blue', alpha = 0.2)
    ax.set_ylim(-1.1*epsilon2, 1.05)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.25)

    fig.savefig(savepdf, transparent = True)
