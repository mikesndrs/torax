# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful functions for handling of IMAS core_profiles or plasma_profiles IDSs and converts them from/into TORAX
objects"""
from typing import Any
import numpy as np
import datetime
import jax
try:
    import imas
    from imas.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any
from torax._src.orchestration.sim_state import ToraxSimState
from torax._src.output_tools import post_processing
from torax._src.geometry.geometry import face_to_cell

def update_dict(old_dict:dict, updates:dict) -> dict:
  """Recursively modify the fields from the original dict old_dict using the values contained in updates dict.
  Used to update config dict fields more easily. Use case is to update config dict with output from core_profiles.core_profiles_from_IMAS().
  Args:
    old_dict: The current dict that needs to be updated.
    updates: Dict containing the values of the keys that need to be updated in old_dict.

  Returns:
    New updated copy of the dict.
  """
  new_dict = old_dict.copy()
  for key, value in updates.items():
      if isinstance(value, dict) and key in new_dict and isinstance(new_dict[key], dict):
          if all(isinstance(k, float) for k in value.keys()):
            new_dict[key] = value #Needed to replace completely the time slices profiles, instead of keeping the initial ones.
          else:
            new_dict[key] = update_dict(new_dict[key], value)
      else:
          new_dict[key] = value
  return new_dict

def core_profiles_from_IMAS(
    ids: IDSToplevel,
    read_psi_from_geo: bool = True,
    ) -> dict:
    """Converts core_profiles IDS to a dict with the input profiles for the config.
    Args:
    ids: IDS object. Can be either core_profiles or plasma_profiles.
    read_psi_from_geo: Decides either to read psi from the geometry or from the input core/plasma_profiles IDS. Default value is True meaning that psi is taken from the geometry.

    Returns:
    Dict containing the updated fields read from the IDS that need to be replaced in the input config."""
    profiles_1d = ids.profiles_1d
    time_array = [float(profiles_1d[i].time) for i in range(len(profiles_1d))]
    rhon_array = [profiles_1d[i].grid.rho_tor_norm for i in range(len(profiles_1d))]
    # numerics
    t_initial = float(profiles_1d[0].time)

    #plasma_composition (should be set in the config as user defined free parameter)
    #Zeff taken from here or set into config ?
    if len(ids.global_quantities.z_eff_resistive>0):
      Z_eff = {time_array[ti]: ids.global_quantities.z_eff_resistive[ti] for ti in range(len(time_array))}
    else:
      Z_eff = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].zeff[rj] for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
      # Zi_override = 1.0
      # Ai_override = 2.5
      # Zimp_override = 1.0
      # Aimp_override = 1.0

    #profile_conditions
    # Should we shift it to get psi=0 at the center ?
    if not read_psi_from_geo:
      psi = {t_initial: {rhon_array[0][rj]: 1 * (profiles_1d[0].grid.psi[rj]) for rj in range(len(rhon_array[0]))}} #To discuss either we provide it here or init it from geo
    else:
       psi = None
    #Will be overwritten anyway if Ip_from_parameters = False, when Ip is given from the equilibrium (in most cases probably).
    Ip = {time_array[ti]: -1 * ids.global_quantities.ip[ti]for ti in range(len(time_array))} #Should come from geometry. Need to be mapped or not ?

    T_e = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].electrons.temperature[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
    #bound_right conditions commented: Will raise an error if rhon[-1]!= 1.
    # Te_bound_right = {time_array[ti]: profiles_1d[ti].electrons.temperature[-1]/1e3 for ti in range(len(time_array))}
    if len(profiles_1d[0].t_i_average>0):
      T_i = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].t_i_average[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
      # Ti_bound_right = {time_array[ti]: profiles_1d[ti].t_i_average[-1]/1e3 for ti in range(len(time_array))}
    else:
      t_i_average = [np.mean([profiles_1d[ti].ion[iion].temperature for iion in range(len(profiles_1d[ti].ion))], axis = 0) for ti in range(len(time_array))]
      T_i = {time_array[ti]: {rhon_array[ti][rj]: t_i_average[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
      # Ti_bound_right = {time_array[ti]: t_i_average[-1]/1e3 for ti in range(len(time_array))}

    n_e = {time_array[ti]: {rhon_array[ti][ri]: profiles_1d[ti].electrons.density[ri] for ri in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
    # ne_bound_right = {time_array[ti]: profiles_1d[ti].electrons.density[-1]for ti in range(len(time_array))}
    # nbar =

    return {"plasma_composition" :{
            "Z_eff": Z_eff,
        },
        "profile_conditions": {
            "Ip": Ip,
            "psi": psi,
            "T_i": T_i,
            "T_i_right_bc": None,
            "T_e": T_e,
            "T_e_right_bc": None,
            "n_e_right_bc_is_fGW": False,
            "n_e_right_bc": None,
            "n_e_nbar_is_fGW": False,
            "n_e" : n_e,
            "normalize_n_e_to_nbar": False,
            "set_pedestal": True, #Should it be set ? Probably not
        },
        "numerics": {
            "t_initial": t_initial,
            "t_final": t_initial+80., #How to define it ? Somewhere else ?
        }
    }

#TODO: Add option to save entire state history in one core_profiles output. 
def core_profiles_to_IMAS(
    post_processed_outputs: post_processing.PostProcessedOutputs,
    state: ToraxSimState,
    ids: IDSToplevel = imas.IDSFactory().core_profiles, 
) -> IDSToplevel:
    """Converts torax core_profiles to IMAS IDS.
    Takes the cell grid as a basis and converts values on face grid to cell.
    Args:
    post_processed_outputs: TORAX post_processed_outputs with many useful data to output to the IDS.
    state: A ToraxSimState object.
    ids: Optional IDS object to be filled. Can be either core_profiles or plasma_profiles. Default is an empty core_profiles IDS. Note that both exists currently from Data Dictionary version 4, with plasma_profiles being the union of core_profiles and edge_profiles. 

    Returns:
      Filled core_profiles or plas_profiles IDS object."""
    t = state.t
    cp_state = state.core_profiles
    cs_state = state.core_sources
    geometry = state.geometry
    ids.ids_properties.comment = "IDS built from TORAX sim output. Grid based on torax cell grid, used cell grid values and interpolated face grid values"
    ids.ids_properties.homogeneous_time = 1
    ids.ids_properties.creation_date = datetime.date.today().isoformat()
    ids.time = [t]
    ids.code.name = "TORAX"
    ids.code.description = "TORAX is a differentiable tokamak core transport simulator aimed for fast and accurate forward modelling, pulse-design, trajectory optimization, and controller design workflows."
    ids.code.repository = "https://github.com/google-deepmind/torax"
    ids.vacuum_toroidal_field.b0.resize(1)
    ids.global_quantities.current_bootstrap.resize(1)
    ids.global_quantities.ip.resize(1)
    ids.global_quantities.v_loop.resize(1)
    ids.global_quantities.li_3.resize(1)
    ids.global_quantities.beta_pol.resize(1)
    ids.global_quantities.beta_tor.resize(1)
    ids.global_quantities.beta_tor_norm.resize(1)
    ids.global_quantities.t_e_volume_average.resize(1)
    ids.global_quantities.n_e_volume_average.resize(1)
    ids.global_quantities.ion.resize(1) #Volume average Ti and ni only available for main ion (could be modified to define it for each of the main ions at least, and t_i_average for all ions, impurities included).
    ids.global_quantities.ion[0].t_i_volume_average.resize(1)
    ids.global_quantities.ion[0].n_i_volume_average.resize(1)

    ids.profiles_1d.resize(1)
    ids.profiles_1d[0].time = t
    ids.profiles_1d[0].ion.resize(2)
    ids.profiles_1d[0].ion[0].element.resize(1)
    ids.profiles_1d[0].ion[1].element.resize(1)

    ids.vacuum_toroidal_field.r0 = geometry.R_major
    ids.vacuum_toroidal_field.b0[0] = geometry.B_0 # +1 or -1 ?

    ids.global_quantities.ip[0] = -1 * cp_state.Ip_profile_face[-1]
    ids.global_quantities.current_bootstrap[0] = -1 * post_processed_outputs.I_bootstrap
    ids.global_quantities.v_loop[0] = cp_state.v_loop_lcfs
    ids.global_quantities.li_3[0] = post_processed_outputs.li3
    ids.global_quantities.beta_pol[0] = post_processed_outputs.beta_pol
    ids.global_quantities.beta_tor[0] = post_processed_outputs.beta_tor
    ids.global_quantities.beta_tor_norm[0] = post_processed_outputs.beta_N
    ids.global_quantities.t_e_volume_average[0] = post_processed_outputs.T_e_volume_avg * 1e3
    ids.global_quantities.n_e_volume_average[0] = post_processed_outputs.n_e_volume_avg 
    ids.global_quantities.ion[0].t_i_volume_average[0] = post_processed_outputs.T_i_volume_avg * 1e3
    ids.global_quantities.ion[0].n_i_volume_average[0] = post_processed_outputs.n_i_volume_avg
  
    ids.profiles_1d[0].grid.rho_tor_norm = np.concatenate([[0.0], geometry.torax_mesh.cell_centers, [1.0]]) 
    Phi = np.concatenate([[geometry.Phi_face[0]],geometry.Phi, [geometry.Phi_face[-1]]])
    ids.profiles_1d[0].grid.rho_tor =  np.sqrt(Phi / (np.pi * geometry.B_0))
    ids.profiles_1d[0].grid.psi = cp_state.psi.cell_plus_boundaries()
    ids.profiles_1d[0].grid.psi_magnetic_axis = cp_state.psi._left_face_value()[0]
    ids.profiles_1d[0].grid.psi_boundary = cp_state.psi._right_face_value()[0]
    volume = np.concatenate([[geometry.volume_face[0]], geometry.volume, [geometry.volume_face[-1]]])
    area = np.concatenate([[geometry.area_face[0]], geometry.area, [geometry.area_face[-1]]])
    ids.profiles_1d[0].grid.volume = volume
    ids.profiles_1d[0].grid.area = area 

    ids.profiles_1d[0].electrons.temperature = cp_state.T_e.cell_plus_boundaries() * 1e3
    ids.profiles_1d[0].electrons.density = cp_state.n_e.cell_plus_boundaries() 
    ids.profiles_1d[0].electrons.density_thermal = cp_state.n_e.cell_plus_boundaries() 
    ids.profiles_1d[0].electrons.density_fast = np.zeros(len(ids.profiles_1d[0].grid.rho_tor_norm)) #TODO: check if we can deduce the density thermal and fast somehow
    ids.profiles_1d[0].electrons.pressure_thermal = post_processed_outputs.pressure_thermal_e.cell_plus_boundaries() 
    ids.profiles_1d[0].pressure_ion_total = post_processed_outputs.pressure_thermal_i.cell_plus_boundaries() 
    ids.profiles_1d[0].pressure_thermal = post_processed_outputs.pressure_thermal_total.cell_plus_boundaries() 
    ids.profiles_1d[0].t_i_average = cp_state.T_i.cell_plus_boundaries() * 1e3
    ids.profiles_1d[0].n_i_total_over_n_e = (cp_state.n_i.cell_plus_boundaries() + cp_state.n_impurity.cell_plus_boundaries()) / cp_state.n_e.cell_plus_boundaries()
    Z_i = np.concatenate([[cp_state.Z_i_face[0]], cp_state.Z_i, [cp_state.Z_i_face[-1]]])
    Z_impurity = np.concatenate([[cp_state.Z_impurity_face[0]], cp_state.Z_impurity, [cp_state.Z_impurity_face[-1]]])
    ids.profiles_1d[0].zeff = (Z_i**2 * cp_state.n_i.cell_plus_boundaries() + Z_impurity**2 * cp_state.n_impurity.cell_plus_boundaries()) / cp_state.n_e.cell_plus_boundaries() #Formula correct ?

    #TODO:add ion mixture details. Currently, only fill values for main ion and impurity averaged, do not take into account the mixture from config
    ids.profiles_1d[0].ion[0].z_ion = np.mean(cp_state.Z_i)  # Change to make it correspond to volume average over plasma radius
    ids.profiles_1d[0].ion[0].z_ion_1d = Z_i 
    ids.profiles_1d[0].ion[0].temperature = cp_state.T_i.cell_plus_boundaries() * 1e3
    ids.profiles_1d[0].ion[0].density = cp_state.n_i.cell_plus_boundaries() 
    ids.profiles_1d[0].ion[0].density_thermal = cp_state.n_i.cell_plus_boundaries()
    ids.profiles_1d[0].ion[0].density_fast = np.zeros(len(ids.profiles_1d[0].grid.rho_tor_norm))
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[0].element[0].a = cp_state.A_i
    ids.profiles_1d[0].ion[0].element[0].z_n = np.round(np.mean(cp_state.Z_i)) # This or read the data from IonMixture ?

    ids.profiles_1d[0].ion[1].z_ion = np.mean(cp_state.Z_impurity_face) # Change to make it correspond to volume average over plasma radius
    ids.profiles_1d[0].ion[1].z_ion_1d = Z_impurity
    ids.profiles_1d[0].ion[1].temperature = cp_state.T_i.cell_plus_boundaries()
    ids.profiles_1d[0].ion[1].density = cp_state.n_impurity.cell_plus_boundaries()
    ids.profiles_1d[0].ion[1].density_thermal = cp_state.n_impurity.cell_plus_boundaries()
    ids.profiles_1d[0].ion[1].density_fast = np.zeros(len(ids.profiles_1d[0].grid.rho_tor_norm))
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[1].element[0].a = cp_state.A_impurity
    ids.profiles_1d[0].ion[1].element[0].z_n = np.round(np.mean(cp_state.Z_impurity_face)) # This or read the data from IonMixture ?
    q_cell = face_to_cell(cp_state.q_face)
    s_cell = face_to_cell(cp_state.s_face)
    ids.profiles_1d[0].q = np.concatenate([[cp_state.q_face[0]], q_cell, [cp_state.q_face[-1]]]) 
    ids.profiles_1d[0].magnetic_shear = np.concatenate([[cp_state.s_face[0]], s_cell, [cp_state.s_face[-1]]]) 
    j_total = np.concatenate([[cp_state.j_total_face[0]], cp_state.j_total, [cp_state.j_total_face[-1]]]) 
    j_bootstrap = np.concatenate([[cs_state.bootstrap_current.j_bootstrap_face[0]], cs_state.bootstrap_current.j_bootstrap, [cs_state.bootstrap_current.j_bootstrap_face[-1]]]) 
    ids.profiles_1d[0].j_total = -1 * j_total 
    ids.profiles_1d[0].j_ohmic = -1 * post_processed_outputs.j_ohmic #TODO: Extend grid with boundaries : Need to find a way for these 2 as there is only values on cell grid for external current sources
    ids.profiles_1d[0].j_non_inductive = -1 *(sum(cs_state.psi.values()) + cs_state.bootstrap_current.j_bootstrap) # Extend grid with boundaries 
    ids.profiles_1d[0].j_bootstrap = j_bootstrap
    sigma = np.concatenate([[cp_state.sigma_face[0]], cp_state.sigma, [cp_state.sigma_face[-1]]]) 
    ids.profiles_1d[0].conductivity_parallel = sigma
    return ids
