#!/bin/sh
#	
# Quantum Lattice Boltzmann 
# (c) 2015 Fabian Thüring, ETH Zürich
#
# This script can profile the application with 'gprof' (GNU Profiler) and 
# 'nvprof' (NVIDIA Cuda profiler) aswell as perform some sanatizing tests
# provided by the LLVM sanatizer or Valgrind. In addition it supports
# Profile Guided Optimization (PGO) for LLVM (Linux only).
# 
# Note: To use PGO to it's full potential you might have to allow sampling of 
#       kernel functions during recording
#       > sudo -s
#       > echo 0 > /proc/sys/kernel/kptr_restrict

print_help() 
{
	echo "usage: $0 [options]"
	echo ""
	echo "options:"
	echo "   --gprof           Profile the program with 'gprof'"
	echo "   --nvprof          Profile the program with 'nvprof'"
	echo "   --pgo             Use LLVM's Profile Guided Optimization (PGO) with the"
	echo "                     Linux Perf profiler [Linux only]"
	echo "   --llvm-sanatize   Run LLVM sanatizer tests (address and undefined)"
	echo "   --valgrind        Run the program in valgrind to detect memory management"
	echo "                     and threading bugs"
	echo ""
	echo "   --CUDA=S          Compile against CUDA if S=true [default S=true]"
	echo "   --args=Args       The follwing arguments are passed to the executing"
	echo "                     program while multiple arguments are delimited with ','" 
	echo "                     (e.g '--args=--L=128,--dx=1.5')"
	echo "   --g++=DIR         Command to invoke g++ (GNU Compiler used by '--gprof')"
	echo "                     (e.g '--g++=/usr/bin/g++')"
	echo "   --clang++=DIR     Command to invoke clang++ (used by '--llvm-sanatizer')"
	echo "   --llvm_prof=DIR   Command to invoke create_llvm_prof (used by '--pgo')"
	echo "                     from http://github.com/google/autofdo"
	exit 1
}

exit_after_error()
{
	echo "$1"
	exit 1
}

# ======================== Parse Command-line Arguments ========================
gprof=false
nvprof=false
pgo=false
llvm_sanatizer=false
valgrind=false
CUDA=true

GCC=g++
CLANG=clang++
CREATE_LLVM_PROF=create_llvm_prof
CXX=""

EXE=./QLB
EXEargs=""
exe_args_cmd=""

for param in $*
	do case $param in
		"--help") print_help ;;
		"--CUDA="*)  CUDA="${param#*=}" ;;
		"--args="*) exe_args_cmd="${param#*=}" ;;
		"--valgrind") valgrind=true ;;
		"--g++="*) GCC="${param#*=}" ;;
		"--clang++="*) CLANG="${param#*=}" ;;
		"--llvm_prof="*) CREATE_LLVM_PROF="${param#*=}" ;;
		"--llvm-sanatize") llvm_sanatizer=true ;;
		"--gprof") gprof=true ;;
		"--nvprof") nvprof=true ;;
		"--pgo") pgo=true ;;
	esac
done

# Check if the compiler works
if [ "$llvm_sanatizer" = "false" ]; then
	gcc_ver=$($GCC --version | grep LLVM)
	if [ "$gcc_ver" != "" ]; then
		exit_after_error  "$0 : error : 'g++' is symlinked to 'clang++'"
	fi
	CXX=$GCC
else
	clang_ver=$($CLANG --version | grep LLVM)
	if [ "$clang_ver" = "" ]; then
		exit_after_error  "$0 : error : '$CLANG' is incompatible with '--llvm-sanatize'"
	fi
	CXX=$CLANG
fi

# Separate arguments
for arg in $(echo $exe_args_cmd | tr "," "\n")
do
	EXEargs="$EXEargs $arg"
done

# =============================== Valgrind =====================================
if [ "$valgrind" != "false" ]; then
	valgrind --version > /dev/null 2>&1
	if [ "$?" != "0" ]; then
		exit_after_error "$0 : error : cannot find 'valgrind'"
	fi
	echo " === VALGRIND ==="
	EXEargs="$EXEargs --tmax=1 --no-gui"
	valgrind $EXE $EXEargs
	exit 0
fi

# ============================= LLVM sanatizer =================================
if [ "$llvm_sanatizer" = "true" ]; then
	# Recompile with sanatizer flags
	make clean && make -j 4 \
		 PROFILING='-fsanitize=undefined -fsanitize=address' CXX=$CXX CUDA=false
	
	echo " === LLVM SANATIZER === "
	EXEargs="$EXEargs --tmax=1 --no-gui"
	$EXE $EXEargs
	exit 0
fi

# ================================ gprof =======================================
if [ "$gprof" = "true" ]; then
	
	# Check for gprof
	gprof --version > /dev/null 2>&1
	if [ "$?" != "0" ]; then
		exit_after_error "$0 : error : cannot find 'gprof'"
	fi
	
	# Recompile with '-pg' flag
	printf "Recompiling ... "
	make -s clean && make -s -j 4 PROFILING='-pg' CXX=$CXX CUDA=$CUDA
	printf "Done\n"

	# Execute the program
	printf "Profiling ... \n\n"
	$EXE $EXEargs
	if [ "$?" != "0" ]; then
		exit 1
	fi

	# Create profile information
	printf "Creating profile information 'analysis.txt' ..."
	gprof QLB gmon.out > analysis.txt
	printf "Done\n"
	exit 0
fi

# =============================== nvprof =======================================
if [ "$nvprof" = "true" ]; then
	
	# Check for nvprof
	nvprof --version > /dev/null 2>&1
	if [ "$?" != "0" ]; then
		exit_after_error "$0 : error : cannot find 'nvprof'"
	fi
	
	# Recompile to assure CUDA is enabled
	printf "Recompiling ... "
	make -s clean && make -j -s CUDA=true
	printf "Done\n"

	# Check if '--device=gpu' flag is already present
	device_flag=false
	for cmd in $(echo $EXEargs | tr "," "\n")
	do
		if [ "$cmd" = "--device=gpu" ]; then
			device_flag=true
		fi
	done

	if [ "$device_flag" = "false" ]; then
		EXEargs="$EXEargs --device=gpu"
	fi 
	
	# Execute the program
	printf "Profiling ... \n\n"
	nvprof --log-file analysis.txt $EXE $EXEargs
	if [ "$?" != "0" ]; then
		exit 1
	fi

	# Create profile information
	printf "\nCreating profile information 'analysis.txt' ... Done\n"
	exit 0
fi

# ================================== PGO =======================================
if [ "$pgo" = "true" ]; then
	
	# Check if we are on Linux
	if [ $(uname -s 2>/dev/null) = "Linux" ]; then
		exit_after_error "$0 : error : PGO is only supported on Linux"
	fi
	
	# Check for perf
	perf --version > /dev/null 2>&1
	if [ "$?" != "0" ]; then
		exit_after_error "$0 : error : cannot find 'perf'"
	fi
	
	# Check for create_llvm_prof
	$CREATE_LLVM_PROF --version > /dev/null 2>&1
	if [ "$?" != "0" ]; then
		echo "$0 : error : cannot find 'create_llvm_prof' take a look at"
		exit_after_error " http://github.com/google/autofdo"
	fi

	# Recompile with '-g -gline-tables-only' flag
	printf "Recompiling ... "
	make -s clean && make -s -j 4 PROFILING='-g -gline-tables-only' CXX=$CLANG CUDA=$CUDA
	printf "Done\n"
	
	# Execute the program
	printf "Profiling ... \n\n"
	perf record -b $EXE $EXEargs
	if [ "$?" != "0" ]; then
		exit 1
	fi

	#Convert 	
	printf "\nCreating profile information 'perf.data' ... "
	$CREATE_LLVM_PROF --binary=$EXE --out=code.prof
	printf "Done\n"
	printf "Converting to LLVM's format 'code.prof' ... Done\n"

	printf "Recompiling ... "
	# Recompile with '-g -gline-tables-only -fprofile-sample-use=code.prof' flag
	make -s clean && make -s -j 4 PROFILING='-g -gline-tables-only -fprofile-sample-use=code.prof' \
	CXX=$CLANG CUDA=$CUDA
	printf "Done\n"

	printf "Cleaning up ..."
	rm -f code.prof perf.data
	printf "Done\n"
	printf "Profile Guided Optimization completed successfully.\n"
	exit 0	
fi

print_help
