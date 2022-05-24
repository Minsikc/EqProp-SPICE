######### build Xyce directly #############
# installation confirmed only at ubuntu 20.04

apt update && apt upgrade -y
### install prerequisites
sudo apt-get install -y \
g++ \
gfortran \
make \
cmake \
bison \
flex \
libfl-dev \
libfftw3-dev \
libsuitesparse-dev \
libblas-dev \
liblapack-dev \
libtool \
libopenmpi-dev \
openmpi-bin
### build trilinos
cd $HOME
mkdir Trilinos12.12 && cd Trilinos12.12
wget https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-12-12-1.tar.gz
tar -xzf trilinos-release-12-12-1.tar.gz
mkdir trilinos-build && cd trilinos-build

cat > reconfigure <<- EOM
#!/bin/sh
SRCDIR=$HOME/Trilinos12.12/Trilinos-trilinos-release-12-12-1
ARCHDIR=$HOME/XyceLibs/Parallel
FLAGS="-O3 -fPIC"
cmake \
-G "Unix Makefiles" \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_Fortran_COMPILER=mpif77 \
-DCMAKE_CXX_FLAGS="$FLAGS" \
-DCMAKE_C_FLAGS="$FLAGS" \
-DCMAKE_Fortran_FLAGS="$FLAGS" \
-DCMAKE_INSTALL_PREFIX=$ARCHDIR \
-DCMAKE_MAKE_PROGRAM="make" \
-DTrilinos_ENABLE_NOX=ON \
  -DNOX_ENABLE_LOCA=ON \
-DTrilinos_ENABLE_EpetraExt=ON \
  -DEpetraExt_BUILD_BTF=ON \
  -DEpetraExt_BUILD_EXPERIMENTAL=ON \
  -DEpetraExt_BUILD_GRAPH_REORDERINGS=ON \
-DTrilinos_ENABLE_TrilinosCouplings=ON \
-DTrilinos_ENABLE_Ifpack=ON \
-DTrilinos_ENABLE_Isorropia=ON \
-DTrilinos_ENABLE_AztecOO=ON \
-DTrilinos_ENABLE_Belos=ON \
-DTrilinos_ENABLE_Teuchos=ON \
  -DTeuchos_ENABLE_COMPLEX=ON \
-DTrilinos_ENABLE_Amesos=ON \
 -DAmesos_ENABLE_KLU=ON \
-DTrilinos_ENABLE_Amesos2=ON \
 -DAmesos2_ENABLE_KLU2=ON \
 -DAmesos2_ENABLE_Basker=ON \
-DTrilinos_ENABLE_Sacado=ON \
-DTrilinos_ENABLE_Stokhos=ON \
-DTrilinos_ENABLE_Kokkos=ON \
-DTrilinos_ENABLE_Zoltan=ON \
-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-DTrilinos_ENABLE_CXX11=ON \
-DTPL_ENABLE_AMD=ON \
-DAMD_LIBRARY_DIRS="/usr/lib" \
-DTPL_AMD_INCLUDE_DIRS="/usr/include/suitesparse" \
-DTPL_ENABLE_BLAS=ON \
-DTPL_ENABLE_LAPACK=ON \
-DTPL_ENABLE_MPI=ON \
$SRCDIR
EOM

chmod u+x reconfigure
./reconfigure
make -j8
sudo make install -j8

###build Xyce
cd $HOME
read -n 1 -p "get Xyce-7.4 source from sandia.gov -> download it -> press enter"
read -p "Input base folder path where xyce located in" xycesource
cd $xycesource
tar -xzf Xyce-7.4.tar.gz

cd ./Xyce-7.4
mkdir build && cd build
$xycesource/Xyce-7.4/configure \
CXXFLAGS="-O3" \
ARCHDIR=$xycesource/XyceLibs/Parallel \
CPPFLGS="-l/usr/include/suiteparse" \
--enable-mpi \
CXX=mpicxx \
CC=mpicc \
F77=mpif77 \
--enable-stokhos \
--enable-amesos2 \
--prefix=$xycesource/Xyceinstall/Parallel

make -j8
sudo make install -j8

# add to $PATH
Xycepath=$xycesource/Xyceinstall/Parallel/bin
export PATH=$PATH:$Xycepath >> ~/.profile