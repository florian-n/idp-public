rm ./sassena/signal.h5
sassena --config ./sassena/sassena_lammps.xml
python3 ./sassena/h5plotter.py jaxmd_then_sassena.csv