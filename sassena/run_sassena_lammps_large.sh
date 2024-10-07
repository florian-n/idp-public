LAMMPS_OUT_FOLDER="./data/lammps_large"
SIGNAL_H5="$LAMMPS_OUT_FOLDER/signal.h5"

#lmp_serial -in ./lammps/input_large.lammps
#python3 ./lammps/pdbconverter.py ./lammps/output/tip3p_large.data "$LAMMPS_OUT_FOLDER/output.dcd" "$LAMMPS_OUT_FOLDER"
#rm $SIGNAL_H5
#sassena --config ./sassena/sassena_lammps_large.xml --scattering.signal.file $SIGNAL_H5
python3 ./sassena/h5plotter.py lammps_large_then_sassena.csv
