import MDAnalysis as mda
import numpy as np

universe = mda.Universe("./lammps/output/tip3p.data", "./data/lammps/output.dcd")
atoms = universe.select_atoms("all")

n_residues = len(universe.atoms) // 3
n_atoms = n_residues * 3

resindices = np.repeat(range(n_residues), 3).tolist()
segindices = [0] * n_residues

universe.add_TopologyAttr("name", ["O", "H1", "H2"] * n_residues)
universe.add_TopologyAttr("type", ["O", "H", "H"] * n_residues)
universe.add_TopologyAttr("resnames", ["SOL"] * n_residues)

with mda.Writer("./data/lammps/output.pdb", atoms.n_atoms) as w:
    w.write(universe.atoms)


with mda.Writer("./data/lammps/trajectory.pdb", atoms.n_atoms) as pdb_writer:
    for _ in universe.trajectory:
        pdb_writer.write(atoms)
