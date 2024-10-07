import MDAnalysis as mda
import numpy as np
import sys

data_file = sys.argv[1]
dcd_file = sys.argv[2]
output_directory = sys.argv[3]

universe = mda.Universe(data_file, dcd_file)
atoms = universe.select_atoms("all")

n_residues = len(universe.atoms) // 3
n_atoms = n_residues * 3

resindices = np.repeat(range(n_residues), 3).tolist()
segindices = [0] * n_residues

universe.add_TopologyAttr("name", ["O", "H1", "H2"] * n_residues)
universe.add_TopologyAttr("type", ["O", "H", "H"] * n_residues)
universe.add_TopologyAttr("resnames", ["SOL"] * n_residues)

with mda.Writer(f"{output_directory}/output.pdb", atoms.n_atoms) as w:
    w.write(universe.atoms)


with mda.Writer(f"{output_directory}/trajectory.pdb", atoms.n_atoms) as pdb_writer:
    for _ in universe.trajectory:
        pdb_writer.write(atoms)
