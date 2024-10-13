import MDAnalysis as mda
import numpy as np


def load_trajectory(dcd_file: str, pdb_file: str) -> np.ndarray:
    universe = load_universe(dcd_file, pdb_file)

    trajectory = []
    for _ in universe.trajectory:
        trajectory.append(universe.atoms.positions)

    return np.array(trajectory)


def load_universe(dcd_file: str, pdb_file: str) -> np.ndarray:
    universe = mda.Universe(dcd_file, pdb_file)
    return universe
