#!/bin/bash
#
# Check debian packages
#
function check_package {
	dpkg -s $1 1> /dev/null 2> /dev/null
	if [[ "$?" != "0" ]];
	then
		echo "Package $1 needs to be installed first"
		exit 1
	fi
}
#module load oneapi/eng-compiler/2023.12.15.002 python py-numpy
module load python py-numpy fftw pti-gpu
source ~/spack/share/spack/setup-env.sh
#spack load c-lime
spack load openssl
#export CLIME=`spack find --paths c-lime | grep ^c-lime | awk '{print $2}' `
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

#module load python
#module load py-numpy
#module load PrgEnv-gnu craype-accel-amd-gfx90a amd-mixed rocm cray-python cray-mpich craype-x86-trento cray-fftw
#export MPICH_GPU_SUPPORT_ENABLED=1
#check_package gcc
#check_package python3
#check_package python3-pip
#check_package wget
#check_package autoconf
#check_package libssl-dev
#check_package zlib1g-dev
#check_package libfftw3-dev

#
# Install python3 if it is not yet there
#
echo "Checking numpy"
hasNumpy=$(python3 -c "import numpy" 2>&1 | grep -c ModuleNotFound)
if [[ "$hasNumpy" == "1" ]];
then
    echo "Install numpy"
    python3 -m pip install --user numpy
fi

#
# Get root directory
#
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

#
# Precompile python
#
echo "Compile gpt"
python3 -m compileall ${root}/lib/gpt


#
# Create dependencies and download
#
dep=${root}/dependencies
if [ ! -f ${dep}/Grid/build/Grid/libGrid.a ];
then

	if [ -d ${dep} ];
	then
	    echo "$dep already exists ; rm -rf $dep before bootstrapping again"
	    exit 1
	fi

	mkdir -p ${dep}
	cd ${dep}

	#
	# Lime
	#
	wget https://github.com/usqcd-software/c-lime/tarball/master
	tar xzf master
	mv usqcd-software-c-lime* lime
	rm -f master
	cd lime
	./autogen.sh
	./configure
	make
	cd ..

	#
	# Grid
	#
	
	git clone https://github.com/dbollweg/Grid.git
	cd Grid
	git checkout gpt_proton
	./bootstrap.sh
	mkdir build
	cd build
	TOOLS=$HOME/tools
	../configure \
            --enable-simd=GPU \
	    --enable-reduction=grid \
        --enable-gen-simd-width=64 \
        --enable-comms=mpi-auto \
        --enable-accelerator-cshift \
        --disable-gparity \
        --disable-fermion-reps \
        --enable-shm=nvlink \
        --enable-accelerator=sycl \
        --enable-unified=no \
	--enable-accelerator-aware-mpi=no \
        MPICXX=mpicxx \
        CXX=icpx \
        LDFLAGS="-fiopenmp -fsycl -fsycl-device-code-split=per_kernel -fsycl-device-lib=all -lze_loader -L${MKLROOT}/lib -qmkl=parallel -fsycl -lsycl -fPIC" \
        CXXFLAGS="-O3 -fiopenmp -fsycl-unnamed-lambda -fsycl -Wno-tautological-compare -qmkl=parallel -fsycl -fno-exceptions -fPIC"

	cd Grid
	make -j 32
fi

if [ ! -f ${root}/lib/cgpt/build/cgpt.so ];
then
	#
	# cgpt
	#
	cd ${root}/lib/cgpt
	./make ${root}/dependencies/Grid/build 32
fi

#cd ${root}/tests
#source ${root}/lib/cgpt/build/source.sh
#./run "" "--mpi_split 1.1.1.1"

echo "To use:"
echo "source ${root}/lib/cgpt/build/source.sh"

