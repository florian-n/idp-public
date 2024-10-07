LAMMPS_OUT_FOLDER="./data/lammps"
SIGNAL_H5="$LAMMPS_OUT_FOLDER/signal.h5"

lmp_serial -in ./lammps/input.lammps
python3 ./lammps/pdbconverter.py ./lammps/output/tip3p.data "$LAMMPS_OUT_FOLDER/output.dcd" "$LAMMPS_OUT_FOLDER"
rm $SIGNAL_H5
sassena --config ./sassena/sassena_lammps.xml --scattering.signal.file $SIGNAL_H5
python3 ./sassena/h5plotter.py lammps_then_sassena.csv