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

"""Useful functions for handling of IMAS IDSs and converts them into TORAX
objects"""
from typing import Any
import numpy as np
import datetime
import jax
try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any
from torax import state
from torax import constants
from torax.state import ToraxSimState
from torax.torax_imastools.util import face_to_cell, requires_module

_trapz = jax.scipy.integrate.trapezoid

@requires_module("imaspy")
def core_profiles_from_IMAS(
    ids: IDSToplevel,
    read_psi_from_geo: bool = True,
    ) -> dict:
    """Converts torax core_profiles to IMAS IDS.
    Takes the cell grid as a basis and converts values on face grid to cell.
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
      Zeff = {time_array[ti]: ids.global_quantities.z_eff_resistive[ti] for ti in range(len(time_array))}
    else:
      Zeff = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].zeff[rj] for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
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
    Ip_tot = {time_array[ti]: -1 * ids.global_quantities.ip[ti]/1e6 for ti in range(len(time_array))} #Should come from geometry. Need to be mapped or not ?

    Te = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].electrons.temperature[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
    #bound_right conditions commented: Will raise an error if rhon[-1]!= 1.
    # Te_bound_right = {time_array[ti]: profiles_1d[ti].electrons.temperature[-1]/1e3 for ti in range(len(time_array))}
    if len(profiles_1d[0].t_i_average>0):
      Ti = {time_array[ti]: {rhon_array[ti][rj]: profiles_1d[ti].t_i_average[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
      # Ti_bound_right = {time_array[ti]: profiles_1d[ti].t_i_average[-1]/1e3 for ti in range(len(time_array))}
    else:
      t_i_average = [np.mean([profiles_1d[ti].ion[iion].temperature for iion in range(len(profiles_1d[ti].ion))], axis = 0) for ti in range(len(time_array))]
      Ti = {time_array[ti]: {rhon_array[ti][rj]: t_i_average[rj]/1e3 for rj in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
      # Ti_bound_right = {time_array[ti]: t_i_average[-1]/1e3 for ti in range(len(time_array))}

    ne = {time_array[ti]: {rhon_array[ti][ri]: profiles_1d[ti].electrons.density[ri]/1e20 for ri in range(len(rhon_array[ti]))} for ti in range(len(time_array))}
    # ne_bound_right = {time_array[ti]: profiles_1d[ti].electrons.density[-1]/1e20 for ti in range(len(time_array))}
    # nbar =

    # pedestal (not inside runtime_params, separate dict), should be set in the input config as well, not provided in ids
      # neped = 0.7
      # Tiped = 5.0
      # Teped = 5.0
      # rho_norm_ped_top = 0.91

    return {"runtime_params":{
        "plasma_composition" :{
            "Zeff": Zeff,
        },
        "profile_conditions": {
            "Ip_tot": Ip_tot,
            "psi": psi,
            "Ti": Ti,
            "Ti_bound_right": None,
            "Te": Te,
            "Te_bound_right": None,
            "ne_bound_right_is_fGW": False,
            "ne_bound_right": None,
            "ne_is_fGW": False,
            "ne" : ne,
            "normalize_to_nbar": False,
            "set_pedestal": True, #Should it be set ? Probably not
        },
        "numerics": {
            "t_initial": t_initial,
            "t_final": t_initial+5., #How to define it ? Somewhere else ?
            "nref": 1e20, #Ensure nref is consistent with the normalisation done to density
        }
      }
    }

@requires_module("imaspy")
def core_profiles_to_IMAS(
    ids: IDSToplevel, state: ToraxSimState
) -> IDSToplevel:
    """Converts torax core_profiles to IMAS IDS.
    Takes the cell grid as a basis and converts values on face grid to cell.
    Args:
    ids: IDS object. Can be either core_profiles or plasma_profiles.
    state: torax state object

    Returns:
    filled IDS object"""
    t = state.t
    cp_state = state.core_profiles
    geometry = state.geometry
    post_processed_outputs = state.post_processed_outputs
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

    ids.vacuum_toroidal_field.r0 = geometry.Rmaj
    ids.vacuum_toroidal_field.b0[0] = geometry.B0 # +1 or -1 ?

    ids.global_quantities.ip[0] = -1 * cp_state.currents.Ip_total
    ids.global_quantities.current_bootstrap[0] = -1 * cp_state.currents.I_bootstrap
    ids.global_quantities.v_loop[0] = cp_state.vloop_lcfs
    ids.global_quantities.li_3[0] = post_processed_outputs.li3
    ids.global_quantities.beta_pol[0] = _trapz(post_processed_outputs.pressure_thermal_tot_face * geometry.vpr_face, geometry.rho_face_norm) \
      / (constants.CONSTANTS.mu0 * cp_state.currents.Ip_profile_face[-1]**2 * geometry.Rmaj) #Not sure of the formula. To be checked. Is the thermal pressure = to total perpendicular pressure ?
    # Also Could be interesting to add beta_pol in post_processed_outputs ?
    ids.global_quantities.t_e_volume_average[0] = post_processed_outputs.te_volume_avg * 1e3
    ids.global_quantities.n_e_volume_average[0] = post_processed_outputs.ne_volume_avg * cp_state.nref
    ids.global_quantities.ion[0].t_i_volume_average[0] = post_processed_outputs.ti_volume_avg * 1e3
    ids.global_quantities.ion[0].n_i_volume_average[0] = post_processed_outputs.ni_volume_avg * cp_state.nref

    ids.profiles_1d[0].grid.rho_tor_norm = geometry.rho_norm
    ids.profiles_1d[0].grid.rho_tor = geometry.rho
    ids.profiles_1d[0].grid.psi = cp_state.psi.value
    ids.profiles_1d[0].grid.psi_magnetic_axis = cp_state.psi.face_value()[0]
    ids.profiles_1d[0].grid.psi_boundary = cp_state.psi.face_value()[-1]
    ids.profiles_1d[0].grid.volume = geometry.volume
    ids.profiles_1d[0].grid.area = geometry.area

    ids.profiles_1d[0].electrons.temperature = cp_state.temp_el.value * 1e3
    ids.profiles_1d[0].electrons.density = cp_state.ne.value * cp_state.nref
    ids.profiles_1d[0].electrons.density_thermal = cp_state.ne.value * cp_state.nref
    ids.profiles_1d[0].electrons.density_fast = np.zeros(len(geometry.rho_norm))
    ids.profiles_1d[0].electrons.pressure_thermal = face_to_cell(post_processed_outputs.pressure_thermal_el_face)
    ids.profiles_1d[0].pressure_ion_total = face_to_cell(post_processed_outputs.pressure_thermal_ion_face)
    ids.profiles_1d[0].pressure_thermal = face_to_cell(post_processed_outputs.pressure_thermal_tot_face)
    ids.profiles_1d[0].t_i_average = cp_state.temp_ion.value * 1e3
    ids.profiles_1d[0].n_i_total_over_n_e = (cp_state.ni.value + cp_state.nimp.value) / cp_state.ne.value
    ids.profiles_1d[0].zeff = (cp_state.Zi**2 * cp_state.ni.value + cp_state.Zimp**2 * cp_state.nimp.value) / cp_state.ne.value #Formula correct ?

    #TODO:add ion mixture details. Currently, only fill values for main ion and impurity averaged, do not take into account the mixture from config
    ids.profiles_1d[0].ion[0].z_ion = np.mean(cp_state.Zi)  # Change to make it correspond to volume average over plasma radius
    ids.profiles_1d[0].ion[0].z_ion_1d = cp_state.Zi
    ids.profiles_1d[0].ion[0].temperature = cp_state.temp_ion.value * 1e3
    ids.profiles_1d[0].ion[0].density = cp_state.ni.value * cp_state.nref
    ids.profiles_1d[0].ion[0].density_thermal = cp_state.ni.value * cp_state.nref
    ids.profiles_1d[0].ion[0].density_fast = np.zeros(len(geometry.rho_norm))
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[0].element[0].a = cp_state.Ai
    ids.profiles_1d[0].ion[0].element[0].z_n = np.mean(cp_state.Zi) # This or read the data from IonMixture ?

    ids.profiles_1d[0].ion[1].z_ion = np.mean(cp_state.Zimp) # Change to make it correspond to volume average over plasma radius
    ids.profiles_1d[0].ion[1].z_ion_1d = cp_state.Zimp
    ids.profiles_1d[0].ion[1].temperature = cp_state.temp_ion.value
    ids.profiles_1d[0].ion[1].density = cp_state.nimp.value * cp_state.nref
    ids.profiles_1d[0].ion[1].density_thermal = cp_state.nimp.value * cp_state.nref
    ids.profiles_1d[0].ion[1].density_fast = np.zeros(len(geometry.rho_norm))
    # assume no molecules, revisit later
    ids.profiles_1d[0].ion[1].element[0].a = cp_state.Aimp
    ids.profiles_1d[0].ion[1].element[0].z_n = np.mean(cp_state.Zimp) # This or read the data from IonMixture ?

    ids.profiles_1d[0].q = face_to_cell(cp_state.q_face)
    ids.profiles_1d[0].magnetic_shear = face_to_cell(cp_state.s_face)
    ids.profiles_1d[0].j_total = -1 * cp_state.currents.jtot #Clarify jtot jphi jparallel
    ids.profiles_1d[0].j_ohmic = -1 * cp_state.currents.johm
    ids.profiles_1d[0].j_non_inductive = -1 *(cp_state.currents.external_current_source + cp_state.currents.j_bootstrap)
    ids.profiles_1d[0].j_bootstrap = -1 * cp_state.currents.j_bootstrap
    ids.profiles_1d[0].conductivity_parallel = cp_state.currents.sigma
    return ids
