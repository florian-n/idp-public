import os
from typing import TypeVar, Callable

from helpers.slicing import get_slices

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

from helpers.converters import get_box_length_from_density
import jax
from jax import random, lax
import jax.numpy as jnp
import numpy as np

from jax_md import space, simulate, rigid_body, util, energy, units


from helpers.grid import get_points_on_grid

jax.config.update("jax_enable_x64", True)


def run_simulation(LJ_SIGMA_OO, N_STEPS, N_MOLECULES_PER_AXIS, N_SLICES, init_key):
    unit = units.real_unit_system()
    kT = 296 * unit["temperature"]

    N_MOLECULES_X = N_MOLECULES_PER_AXIS
    N_MOLECULES_Y = N_MOLECULES_PER_AXIS
    N_MOLECULES_Z = N_MOLECULES_PER_AXIS
    N_MOLECULES = N_MOLECULES_X * N_MOLECULES_Y * N_MOLECULES_Z

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

    BOX_SIZE = get_box_length_from_density(N_MOLECULES * 3, 0.1)

    displacement, shift = space.periodic(BOX_SIZE)
    _, unwrapped_shift = space.periodic(BOX_SIZE)

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

            return (
                step_fn(state, unwrapped_shift_fn=unwrapped_shift),
                (real_positions),
            )

        steps = jnp.arange(num_steps)
        state, result = lax.scan(step, state, steps)

        return state, result

    TIMESTEP = 2 * unit["time"]
    NUM_STEPS = N_STEPS
    _, simulation_result = run_simulation(
        key, full_configuration, NUM_STEPS, TIMESTEP, TIMESTEP * 100
    )

    atom_positions = simulation_result

    frame_r, dt_per_snapshot = get_slices(
        atom_positions, 3000, N_SLICES, TIMESTEP / unit["time"]
    )

    return np.array(frame_r), dt_per_snapshot
