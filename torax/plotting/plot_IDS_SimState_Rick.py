import imas
import imaspy
import imaspy.util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torax.torax_imastools.util import requires_module, face_to_cell



matplotlib.use("TkAgg")

# This is a function that is able to plot data from IDS and ToraxSimState


def plot_IDS_ToraxSimState(IDS_1,
                           SimState_1,
                           IDS_2: None=None,
                           SimState_2: None=None,
                           ):


    # load data from IDS_1 (final) and IDS_2 (initial)
    psi_IDS_1 = IDS_1.time_slice[0].profiles_1d.psi 
    rho_tor_norm_IDS_1 = IDS_1.time_slice[0].profiles_1d.rho_tor_norm
    pprime_IDS_1 = IDS_1.time_slice[0].profiles_1d.dpressure_dpsi
    ffprime_IDS_1 = IDS_1.time_slice[0].profiles_1d.f_df_dpsi


    psi_IDS_2 = IDS_2.time_slice[0].profiles_1d.psi 
    rho_tor_norm_IDS_2 = IDS_2.time_slice[0].profiles_1d.rho_tor_norm
    pprime_IDS_2 = IDS_2.time_slice[0].profiles_1d.dpressure_dpsi
    ffprime_IDS_2 = IDS_2.time_slice[0].profiles_1d.f_df_dpsi



    # load data from SimState_1 (final) and SimState_2 (initial)
    psi_SimState_1 = SimState_1.core_profiles.psi.value
    rho_tor_norm_SimState_1 = SimState_1.geometry.torax_mesh.cell_centers
    pprime_SimState_1 = face_to_cell(SimState_1.post_processed_outputs.pprime_face)
    ffprime_SimState_1 = face_to_cell(SimState_1.post_processed_outputs.FFprime_face)

    psi_SimState_2 = SimState_2.core_profiles.psi.value
    rho_tor_norm_SimState_2 = SimState_2.geometry.torax_mesh.cell_centers
    pprime_SimState_2 = face_to_cell(SimState_2.post_processed_outputs.pprime_face)
    ffprime_SimState_2 = face_to_cell(SimState_2.post_processed_outputs.FFprime_face)



    print("length of IDS initial:", len(rho_tor_norm_IDS_2))
    print("length of Simstate initial:", len(rho_tor_norm_SimState_2))
    print("length of IDS final:", len(rho_tor_norm_IDS_1))
    print("length of Simstate final:", len(rho_tor_norm_SimState_1))



    '''
    # plot of final psi against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_1, psi_IDS_1, color='yellow', label=r"final $\psi$ from IDS output TORAX ")
    plt.plot(rho_tor_norm_SimState_1, psi_SimState_1, color='blue', linestyle=':', label=r"final $\psi$ from ToraxSimState ")

    # plot of initial psi against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_2, psi_IDS_2 - psi_IDS_2[0], color='green', alpha=0.7, label=r"initial $\psi$ from IDS input TORAX (unnormalised) ")
    plt.plot(rho_tor_norm_SimState_2, psi_SimState_2, color='magenta', linestyle=':', label=r"initial $\psi$ from ToraxSimState ")
    
    plt.title(r"Poloidal flux $\psi$ against normalised toroidal magnetic flux $\rho_{tor,norm}$ from IDS and ToraxSimState")
    plt.xlabel(r"$\rho_{tor,norm}$")
    plt.ylabel(r"$\psi$ (Wb)")
    plt.xlim(0,1)
    '''

    '''
    # plot of final pprime against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_1, pprime_IDS_1, color='yellow', label=r"final p' from IDS output TORAX ")
    plt.plot(rho_tor_norm_SimState_1, pprime_SimState_1, color='blue', linestyle=':', label=r"final p' from ToraxSimState ")

    # plot of initial pprime against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_2, pprime_IDS_2, color='green', alpha=0.7, label=r"initial p' from IDS input TORAX")
    plt.plot(rho_tor_norm_SimState_2, pprime_SimState_2, color='magenta', linestyle=':', label=r"initial p' from ToraxSimState ")
    
    plt.title(r"p' against normalised toroidal magnetic flux $\rho_{tor,norm}$ from IDS and ToraxSimState")
    plt.xlabel(r"$\rho_{tor,norm}$")
    plt.ylabel(r"p' (Pa/Wb)")
    plt.xlim(0,1)
    '''


    # plot of final ffprime against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_1, ffprime_IDS_1, color='yellow', label=r"final ff' from IDS output TORAX ")
    plt.plot(rho_tor_norm_SimState_1, ffprime_SimState_1, color='blue', linestyle=':', label=r"final ff' from ToraxSimState ")

    # plot of initial ffprime against rho_tor_norm from IDS and ToraxSimState
    plt.plot(rho_tor_norm_IDS_2, ffprime_IDS_2, color='green', alpha=0.7, label=r"initial ff' from IDS input TORAX")
    plt.plot(rho_tor_norm_SimState_2, ffprime_SimState_2, color='magenta', linestyle=':', label=r"initial ff' from ToraxSimState ")
    
    plt.title(r"ff' against normalised toroidal magnetic flux $\rho_{tor,norm}$ from IDS and ToraxSimState")
    plt.xlabel(r"$\rho_{tor,norm}$")
    plt.ylabel(r"ff' ")
    plt.xlim(0,1)



    plt.legend()
    plt.show()