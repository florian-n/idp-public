lmp_serial -in ./lammps/input.lammps
python3 ./lammps/pdbconverter.py ./lammps/output/tip3p.data ./data/lammps/output.dcd ./data/lammps
rm ./sassena/signal.h5
sassena --config ./sassena/sassena_lammps.xml
python3 ./sassena/h5plotter.py lammps_then_sassena.csv
