# lmp_serial -in ./lammps/input_large.lammps
#python3 ./lammps/pdbconverter.py ./lammps/output/tip3p_large.data ./data/lammps_large/output.dcd ./data/lammps_large
rm ./sassena/signal.h5
sassena --config ./sassena/sassena_lammps_large.xml
python3 ./sassena/h5plotter.py lammps_then_sassena.csv
