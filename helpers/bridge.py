import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from helpers.better_debye import get_averaged_debye
from helpers.debye_helper import calculate_scattering_length_matrix

H_SCATTERING_LENGTH = -3.7390
O_SCATTERING_LENGTH = 5.803


"""
(Originally intended as) Bridge between the MDAnalysis and JAX-MD libraries.
"""


class MDABridge:
    def __init__(
        self,
        trajectory: np.ndarray,
        dt_per_frame: np.array,
        box_size: int,
        masses: np.array,
    ):
        self.trajectory = trajectory
        self.n_frames, self.n_particles, _ = self.trajectory.shape
        self.n_molecules = int(self.n_particles / 3)
        self.box_size = box_size
        self.masses = masses

        self.dt_per_frame = dt_per_frame

        self.universe = self._create_mda_universe()

    def from_file(filename: str):
        data = np.load(filename, allow_pickle=True)
        return MDABridge(
            trajectory=data["trajectory"],
            dt_per_frame=data["dt_per_frame"],
            box_size=data["box_size"],
            masses=data["masses"],
        )

    def _create_mda_universe(self) -> mda.Universe:
        n_residues = self.n_molecules
        n_atoms = n_residues * 3

        resindices = np.repeat(range(n_residues), 3).tolist()
        segindices = [0] * n_residues

        universe = mda.Universe.empty(
            n_atoms,
            n_residues=n_residues,
            atom_resindex=resindices,
            residue_segindex=segindices,
            trajectory=True,
        )

        universe.add_TopologyAttr("name", ["O", "H1", "H2"] * n_residues)
        universe.add_TopologyAttr("type", ["O", "H", "H"] * n_residues)
        universe.add_TopologyAttr("type", ["1", "2", "2"] * n_residues)
        universe.add_TopologyAttr("resnames", ["WAT"] * n_residues)
        if self.masses:
            universe.add_TopologyAttr("mass", self.masses.tolist() * n_residues)

        bonds = []
        for o in range(0, self.n_particles, 3):
            bonds.extend([(o, o + 1), (o, o + 2)])
        universe.add_bonds(np.array(bonds))

        angles = [(i * 3 + 1, i * 3, i * 3 + 2) for i in range(self.n_molecules)]
        universe.add_TopologyAttr("angles", angles)
        universe.add_TopologyAttr("dihedrals", [])
        universe.add_TopologyAttr("impropers", [])

        universe.load_new(
            self.trajectory,
            format=MemoryReader,
            dimensions=[self.box_size, self.box_size, self.box_size, 90, 90, 90],
            dt=self.dt_per_frame / 1000,
        )

        return universe

    def dump(self, filename: str):
        np.savez(
            file=filename,
            trajectory=self.trajectory,
            dt_per_frame=self.dt_per_frame,
            box_size=self.box_size,
            masses=self.masses,
        )

    def write_lammps(self, filename: str, ix_frame: int):
        self.universe.trajectory[ix_frame]

        with mda.Writer(filename, multiframe=False) as w:
            w.write(self.universe.atoms)

    def write_lammps_dcd(self, filename: str):
        atoms = self.universe.select_atoms("all")
        atoms.write(filename, frames=self.universe.trajectory)

    def write_lammps_pdb(self, filename: str, all_frames=False):
        atoms = self.universe.select_atoms("all")
        with mda.Writer(filename, atoms.n_atoms) as pdb_writer:
            if all_frames:
                for _ in self.universe.trajectory:
                    pdb_writer.write(atoms)
            else:
                self.universe.trajectory[0]
                pdb_writer.write(atoms)

    def get_scattering_patterns(self, max_q, n_q, n_samples=None):
        if n_samples == None:
            n_samples = len(self.trajectory)

        qs = np.linspace(2.0 * np.pi / self.box_size, max_q, n_q)

        scattering_lengths = calculate_scattering_length_matrix(
            self.n_molecules,
            np.array([O_SCATTERING_LENGTH, H_SCATTERING_LENGTH, H_SCATTERING_LENGTH]),
        )

        debye_result = get_averaged_debye(
            self.trajectory, qs, self.box_size, scattering_lengths
        )
        averages = np.mean(debye_result, axis=0) / (self.n_molecules**2)

        return averages
