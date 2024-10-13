# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================= #
# Modifed for this project to work with both wrapped and unwrapped trajectories #
# ============================================================================= #

from jax_md.simulate import *
from jax_md import simulate


@dataclasses.dataclass
class MyNVTNoseHooverState:
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    Attributes:
      position: The current position of particles. An ndarray of floats
        with shape `[n, spatial_dimension]`.
      momentum: The momentum of particles. An ndarray of floats
        with shape `[n, spatial_dimension]`.
      force: The current force on the particles. An ndarray of floats with shape
        `[n, spatial_dimension]`.
      mass: The mass of the particles. Can either be a float or an ndarray
        of floats with shape `[n]`.
      chain: The variables describing the Nose-Hoover chain.
    """

    position: Array
    unwrapped_position: Array
    momentum: Array
    force: Array
    mass: Array
    chain: NoseHooverChain

    @property
    def velocity(self):
        return self.momentum / self.mass


def my_nvt_nose_hoover(
    energy_or_force_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    unwrapped_shift_fn: ShiftFn,
    dt: float,
    kT: float,
    chain_length: int = 5,
    chain_steps: int = 2,
    sy_steps: int = 3,
    tau: Optional[float] = None,
    **sim_kwargs
) -> Simulator:
    """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

    Samples from the canonical ensemble in which the number of particles (N),
    the system volume (V), and the temperature (T) are held constant. We use a
    Nose Hoover Chain (NHC) thermostat described in [#martyna92]_ [#martyna98]_
    [#tuckerman]_. We follow the direct translation method outlined in
    Tuckerman et al. [#tuckerman]_ and the interested reader might want to look
    at that paper as a reference.

    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        `[n, spatial_dimension]`.
      shift_fn: A function that displaces positions, `R`, by an amount `dR`.
        Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      kT: Floating point number specifying the temperature in units of Boltzmann
        constant. To update the temperature dynamically during a simulation one
        should pass `kT` as a keyword argument to the step function.
      chain_length: An integer specifying the number of particles in
        the Nose-Hoover chain.
      chain_steps: An integer specifying the number, :math:`n_c`, of outer
        substeps.
      sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
        must be either `1`, `3`, `5`, or `7`.
      tau: A floating point timescale over which temperature equilibration
        occurs. Measured in units of `dt`. The performance of the Nose-Hoover
        chain thermostat can be quite sensitive to this choice.
    Returns:
      See above.

    .. rubric:: References
    .. [#martyna92] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
      "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
      The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
    .. [#martyna98] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
      "Explicit reversible integrators for extended systems dynamics."
      Molecular Physics 87. (1998) 1117-1157.
    .. [#tuckerman] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
      Andrea L. Jochim, and Glenn J. Martyna.
      "A Liouville-operator derived measure-preserving integrator for molecular
      dynamics simulations in the isothermal-isobaric ensemble."
      Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
    """
    force_fn = quantity.canonicalize_force(energy_or_force_fn)
    dt = f32(dt)
    dt_2 = f32(dt / 2)
    if tau is None:
        tau = dt * 100
    tau = f32(tau)

    thermostat = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        dof = quantity.count_dof(R)

        state = MyNVTNoseHooverState(R, R, None, force_fn(R, **kwargs), mass, None)
        state = canonicalize_mass(state)
        state = initialize_momenta(state, key, _kT)
        KE = kinetic_energy(state)
        return state.set(chain=thermostat.initialize(dof, KE, _kT))

    @jit
    def apply_fn(state, **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        chain = state.chain

        chain = thermostat.update_mass(chain, _kT)

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p)

        state = my_velocity_verlet(
            force_fn, shift_fn, unwrapped_shift_fn, dt, state, **kwargs
        )

        chain = chain.set(kinetic_energy=kinetic_energy(state))

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p, chain=chain)

        return state

    return init_fn, apply_fn


# Hook into dispatcher fot position_step and add new default option "all"
def my_dispatcher_caller(self, state, *args, **kwargs):
    if type(state.position) in self._registry:
        return self._registry[type(state.position)](state, *args, **kwargs)
    elif "all" in self._registry:
        return self._registry["all"](state, *args, **kwargs)
    return self._fn(state, *args, **kwargs)


def my_position_step(state: T, shift_fn: Callable, dt: float, **kwargs) -> T:
    """Apply a single step of the time evolution operator for positions."""

    if isinstance(shift_fn, Callable):
        shift_fn = tree_map(lambda r: shift_fn, state.position)
    unwrapped_shift_fn = kwargs.pop("unwrapped_shift_fn", None)

    new_position = tree_map(
        lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs),
        shift_fn,
        state.position,
        state.momentum,
        state.mass,
    )

    if unwrapped_shift_fn is not None:
        new_unwrapped_position = tree_map(
            lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs),
            unwrapped_shift_fn,
            state.unwrapped_position,
            state.momentum,
            state.mass,
        )
        return state.set(
            position=new_position, unwrapped_position=new_unwrapped_position
        )
    else:
        return state.set(position=new_position)


def my_velocity_verlet(
    force_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    unwrapped_shift_fn: ShiftFn,
    dt: float,
    state: T,
    **kwargs
) -> T:
    """Apply a single step of velocity Verlet integration to a state."""
    dt = f32(dt)
    dt_2 = f32(dt / 2)

    state = momentum_step(state, dt_2)
    state = position_step(
        state, shift_fn, dt, unwrapped_shift_fn=unwrapped_shift_fn, **kwargs
    )
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_step(state, dt_2)

    return state


def hook():
    simulate.dispatch_by_state.__call__ = my_dispatcher_caller
    simulate.position_step.register("all")(my_position_step)
    simulate.nvt_nose_hoover = my_nvt_nose_hoover
    simulate.NVTNoseHooverState = MyNVTNoseHooverState
