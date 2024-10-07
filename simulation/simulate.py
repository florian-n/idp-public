import os

from helpers.debye_helper import (
    calculate_distance_matrix,
    calculate_scattering_length_matrix,
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"

import jax
from jax import random, lax
import jax.numpy as jnp
import numpy as np

# import numpy as np
from jax_md import space, simulate, rigid_body, util, energy, units


from helpers.converters import shrink_scale_box_size
from helpers.grid import get_points_on_grid

jax.config.update("jax_enable_x64", True)


def run_simulation(LJ_SIGMA_OO, N_STEPS, N_MOLECULES_PER_AXIS, N_SLICES, N_Q, init_key):
    # N_STEPS = 3000
    # N_MOLECULES_PER_AXIS = 5
    # init_key = 345346

    unit = units.real_unit_system()
    kT = 296 * unit["temperature"]

    N_MOLECULES_X = N_MOLECULES_PER_AXIS
    N_MOLECULES_Y = N_MOLECULES_PER_AXIS
    N_MOLECULES_Z = N_MOLECULES_PER_AXIS
    N_MOLECULES = N_MOLECULES_X * N_MOLECULES_Y * N_MOLECULES_Z
    N_PARTICLES = N_MOLECULES * 3
    NUM_EM_STEPS = 500

    # Source: https://docs.lammps.org/Howto_tip3p.html
    O_MASS = 15.9994
    H_MASS = 1.008

    O_CHARGE = -0.834 / 0.05487686461
    H_CHARGE = 0.417 / 0.05487686461

    # LJ_SIGMA_OO = 3.188
    LJ_EPSILON_OO = 0.102

    O_POS = jnp.array([0.00000, -0.06556, 0.00000])
    H1_POS = jnp.array([0.75695, 0.52032, 0.00000])
    H2_POS = jnp.array([-0.75695, 0.52032, 0.00000])

    masses = jnp.array([O_MASS, H_MASS, H_MASS])
    positions = jnp.array([O_POS, H1_POS, H2_POS])
    charges = jnp.array([O_CHARGE, H_CHARGE, H_CHARGE])

    H_SCATTERING_LENGTH = -3.7390
    O_SCATTERING_LENGTH = 5.803
    D_SCATTERING_LENGTH = 6.671

    BOX_SIZE = shrink_scale_box_size(62.086, 8000, N_MOLECULES)  # 46.610

    # density = get_density(N_MOLECULES * 3, BOX_SIZE)

    displacement, shift = space.periodic(BOX_SIZE)

    key = random.PRNGKey(init_key)
    key, mol_ori_key = random.split(key, 2)
    quat_key = random.split(mol_ori_key, N_MOLECULES)

    initial_molecule_positions = get_points_on_grid(
        BOX_SIZE, (N_MOLECULES_X, N_MOLECULES_Y, N_MOLECULES_Z)
    )

    initial_molecule_orientations = rigid_body.random_quaternion(
        quat_key, dtype=util.f64
    )

    full_configuration = rigid_body.RigidBody(
        initial_molecule_positions, initial_molecule_orientations
    )

    shape = rigid_body.point_union_shape(positions, masses)

    # mask: [1 0 0 1 0 0]
    lj_species = np.array([1, 0, 0] * N_MOLECULES)

    lj_sigmas = jnp.array([[1.0, 1.0], [1.0, LJ_SIGMA_OO]])  # 00, 01  # 10, 11

    lj_epsilon = jnp.array([[0.001, 0.001], [0.001, LJ_EPSILON_OO]])

    all_charges = jnp.tile(charges, (N_MOLECULES,))

    def my_energy_function(arr):
        en_lennard_jones = energy.lennard_jones_pair(
            displacement,
            species=lj_species,
            sigma=lj_sigmas,
            epsilon=lj_epsilon,
            r_onset=8,
            r_cutoff=10,
        )(arr)
        en_coulomb = energy.coulomb(
            displacement, box=BOX_SIZE, charge=all_charges, grid_points=96
        )(arr)

        return en_coulomb + en_lennard_jones

    energy_fn = rigid_body.point_energy(my_energy_function, shape)

    def run_simulation(key, configuration, num_steps, timestep, tau):
        init_fn, step_fn = simulate.nvt_nose_hoover(
            energy_fn, shift, timestep, kT, tau=tau
        )

        state = init_fn(key, configuration, mass=shape.mass())

        @jax.remat
        @jax.jit
        def step(state, t):
            real_positions, _ = rigid_body.union_to_points(state.position, shape)
            temperature = rigid_body.temperature(
                state.position, state.momentum, state.mass
            )
            velocities = state.momentum.center / state.mass.center

            return (
                step_fn(state),
                (real_positions, temperature, velocities),
            )

        steps = jnp.arange(num_steps)
        state, result = lax.scan(step, state, steps)

        return state, result

    TIMESTEP = 2 * unit["time"]
    NUM_STEPS = N_STEPS
    _, simulation_result = run_simulation(
        key, full_configuration, NUM_STEPS, TIMESTEP, TIMESTEP * 100
    )

    atom_positions, _, _ = simulation_result

    def get_slices(all_positions: jnp.array, start, n_samples, timestep):
        indices = np.array(
            np.floor(np.linspace(start, len(all_positions) - 1, n_samples)),
            dtype=int,
        )
        selected_states = jnp.array(all_positions[indices])
        timesteps = indices * (timestep / unit["time"])
        return selected_states, timesteps

    frame_r, frame_times = get_slices(atom_positions, 3000, N_SLICES, TIMESTEP)

    return frame_r, frame_times
