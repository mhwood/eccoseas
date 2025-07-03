cd ../
rm build_noIB/*
cd build_noIB
export MPI_HOME=/home/mwood/.conda/envs/cs185c
../../../tools/genmake2 -of ../../../tools/build_options/linux_amd64_gfortran -mods ../code_noIB -mpi
make depend
make
