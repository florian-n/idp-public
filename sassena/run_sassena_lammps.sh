lmp_serial -in ./lammps/input.lammps
python3 ./lammps/pdbconverter.py
rm ./sassena/signal.h5
sassena --config ./sassena/sassena_lammps.xml