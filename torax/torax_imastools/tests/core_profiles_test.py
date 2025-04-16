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

"""Unit tests for torax.torax_imastools.equilibrium.py"""

import importlib
import os
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
import pytest
from absl.testing import absltest, parameterized

from torax.geometry import pydantic_model as geometry_pydantic_model

try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any
import torax
from torax import post_processing
from torax.config import build_runtime_params
from torax.orchestration.run_simulation import prep_simulation
from torax.tests.test_lib import sim_test_case
from torax.torax_imastools.equilibrium import geometry_to_IMAS
from torax.torax_imastools.util import load_IMAS_data, load_ids_from_Data_entry, update_dict
from torax.torax_imastools.core_profiles import core_profiles_from_IMAS, core_profiles_to_IMAS
from torax.torax_pydantic import model_config

pytest.mark.skipif(
    importlib.util.find_spec("imaspy") is None,
    reason="IMASPy optional dependency"
)

class Core_profilesTest(sim_test_case.SimTestCase):
    """Integration Run with core_profiles from a reference run. To be integrated in sim_test_case probably."""

    @parameterized.parameters(
        [
            dict(config_name="test_iterhybrid_rampup_short.py",),
        ]
    )
    def test_run_with_core_profiles_to_IMAS(
        self,
        config_name,
    ):
        """Test that TORAX simulation example can be made with input core_profiles ids profiles, without raising error."""
        if importlib.util.find_spec("imaspy") is None:
            self.skipTest("IMASPy optional dependency")

        # Input core_profiles reading and config loading
        config = self._get_config_dict(config_name)

        #Has to be replaced to load open access data
        path = '/home/ITER/belloum/git/torax_dir/torax/torax/data/third_party/geo/scenario.yaml' #Specify path to load core_profiles -> should we generate example core_profiles ?
        # core_profiles_in = load_IMAS_data(path, "core_profiles")
        core_profiles_in = load_ids_from_Data_entry(path, "core_profiles")

        # Modifying the input config profiles_conditions class
        core_profiles_conditions = core_profiles_from_IMAS(core_profiles_in)
        # config.update_fields(core_profiles_conditions)
        config_with_IMAS_profiles = update_dict(config, core_profiles_conditions) #Is it better to do like this, or first convert to ToraxConfig and use config.config_args.recursive_replace or maybe another function that does the same instead ?
        # Or use ToraxConfig.update_fields ?
        torax_config = model_config.ToraxConfig.from_dict(config_with_IMAS_profiles)

        #Run Sim
        results = torax.run_simulation(torax_config)
        # Check that the simulation completed successfully.
        if results.sim_error != torax.SimError.NO_ERROR:
          raise ValueError(
              f'TORAX failed to run the simulation with error: {results.sim_error}.'
          )

    @parameterized.parameters(
        [
            dict(config_name="test_iterhybrid_rampup_short.py", rtol=0.02, atol=1e-8),
        ]
    )
    def test_init_profiles_from_IMAS(
        self,
        config_name,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
      """Test to compare initialized profiles in TORAX with the initial core_profiles used to check consistency."""
      if importlib.util.find_spec("imaspy") is None:
          self.skipTest("IMASPy optional dependency")

      if rtol is None:
          rtol = self.rtol
      if atol is None:
          atol = self.atol
      # Input core_profiles reading and config loading
      config = self._get_config_dict(config_name)
      #Has to be replaced to load open access data
      path = '/home/ITER/belloum/git/torax_dir/torax/torax/data/third_party/geo/scenario.yaml' #Specify path to load core_profiles -> should we generate example core_profiles ?
      # core_profiles_in = load_IMAS_data(path, "core_profiles")
      core_profiles_in = load_ids_from_Data_entry(path, "core_profiles")
      rhon_in = core_profiles_in.profiles_1d[0].grid.rho_tor_norm

      # Modifying the input config profiles_conditions class
      core_profiles_conditions = core_profiles_from_IMAS(core_profiles_in, read_psi_from_geo= False)
      # config.update_fields(core_profiles_conditions)
      config_with_IMAS_profiles = update_dict(config, core_profiles_conditions) #Is it better to do like this, or first convert to ToraxConfig and use config.config_args.recursive_replace or maybe another function that does the same instead ?
      config_with_IMAS_profiles['geometry']['n_rho']=200 #len(rhon_in): With less resolution we loose some accuracy doing two interpolations
      # Or use ToraxConfig.update_fields ?
      torax_config = model_config.ToraxConfig.from_dict(config_with_IMAS_profiles)

      #Init sim from config
      state_history = prep_simulation(torax_config)

      #Read output values
      torax_mesh=torax_config.geometry.build_provider.torax_mesh
      cell_centers = torax_mesh.cell_centers
      face_centers = torax_mesh.face_centers
      #Compare the initial core_profiles with the ids profiles
      init_core_profiles = state_history[3].core_profiles
      np.testing.assert_allclose(
            np.interp(rhon_in, face_centers, init_core_profiles["temp_el"].face_value())*1e3,
            core_profiles_in.profiles_1d[0].electrons.temperature,
            rtol=rtol,
            atol=atol,
            err_msg="Te profile failed",
        )
      np.testing.assert_allclose(
            np.interp(rhon_in, face_centers, init_core_profiles['temp_ion'].face_value()),
            core_profiles_in.profiles_1d[0].t_i_average/1e3,
            rtol=rtol,
            atol=atol,
            err_msg="Ti profile failed",
        )
      np.testing.assert_allclose(
            np.interp(rhon_in, face_centers, init_core_profiles["ne"].face_value()),
            core_profiles_in.profiles_1d[0].electrons.density/1e20,
            rtol=rtol,
            atol=atol,
            err_msg="ne profile failed",
        )
      np.testing.assert_allclose(
            np.interp(rhon_in, face_centers, init_core_profiles["psi"].face_value()),
            -1 * core_profiles_in.profiles_1d[0].grid.psi,
            rtol=rtol,
            atol=atol,
            err_msg="psi profile failed",
        )

    @parameterized.parameters(
      [
          dict(config_name="test_iterhybrid_rampup_short.py", ids_out = imaspy.IDSFactory().core_profiles()),
          dict(config_name="test_iterhybrid_rampup_short.py", ids_out = imaspy.IDSFactory().plasma_profiles()),
      ]
    )
    def test_save_profiles_to_IMAS(
        self,
        config_name,
        ids_out,
    ):
      """Test to check that data can be written in output to the IDS, either core_profiles or plasma_profiles."""
      if importlib.util.find_spec("imaspy") is None:
          self.skipTest("IMASPy optional dependency")

      # Input core_profiles reading and config loading
      config = self._get_config_dict(config_name)
      #Has to be replaced to load open access data
      path = '/home/ITER/belloum/git/torax_dir/torax/torax/data/third_party/geo/scenario.yaml' #Specify path to load core_profiles -> should we generate example core_profiles ?
      # core_profiles_in = load_IMAS_data(path, "core_profiles")
      core_profiles_in = load_ids_from_Data_entry(path, "core_profiles")

      # Modifying the input config profiles_conditions class
      core_profiles_conditions = core_profiles_from_IMAS(core_profiles_in, read_psi_from_geo = False)
      config_with_IMAS_profiles = update_dict(config, core_profiles_conditions)
      config_with_IMAS_profiles['geometry']['n_rho']=20
      torax_config = model_config.ToraxConfig.from_dict(config_with_IMAS_profiles)

      #Init sim from config
      (
        static_runtime_params_slice,
        dynamic_runtime_params_slice_provider,
        geometry_provider,
        initial_state,
        restart_case,
        step_fn,
        source_models,
      ) = prep_simulation(torax_config)

      sim_state, sim_error = step_fn(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
          initial_state,
      )

      if sim_error != torax.SimError.NO_ERROR:
          raise ValueError(
              f'TORAX failed to run the simulation with error: {sim_error}.'
          )

      core_profiles_to_IMAS(ids_out, sim_state)



if __name__ == "__main__":
    absltest.main()
