#!/usr/bin/env sh
#	
# Quantum Lattice Boltzmann 
# (c) 2015 Fabian Thüring, ETH Zürich
#
# This script can profile the application with 'gprof' and perform sanatizing t
# test provided by LLVM sanatizer or Valgrind

print_help() 
{
	echo "usage: $0 [options]"
	echo ""
	echo "options:"
	echo "   --profile         Profile the program with 'gprof'"
	echo "   --llvm-sanatize   Run LLVM sanatizer tests"
	echo "   --valgrind        Run the program in valgrind to detect memory management"
	echo "                     and threading bugs"
	echo ""
	echo "   --no-recompile    Do not recompile the program"
	echo "   --args=Args       The follwing arguments are passed to the executing"
	echo "                     program while multiple arguments are delimit with ','" 
	echo "                     (e.g '--args=--L=128,--dx=1.5')"
	echo "   --g++=S           Command to invoke g++ (GNU Compiler used by '--profile')"
	echo "                     (e.g '--g++=/usr/bin/g++')"
	echo "   --clang++=S       Command to invoke clang++ (used by '--llvm-sanatizer')"
	exit 1
}

exit_after_error()
{
	echo "$1"
	exit 1
}

# ======================== Parse Command-line Arguments ========================
gprof=false
llvm_sanatizer=false
valgrind=false

GCC=g++
CLANG=clang++
CXX=""
no_recompile=false

EXE=./QLB
EXEargs=""
exe_args_cmd=""

for param in $*
	do case $param in
		"--help") print_help ;;
		"--no-recompile")  no_recompile=true ;;
		"--args="*) exe_args_cmd="${param#*=}" ;;
		"--valgrind") valgrind=true ;;
		"--g++="*) GCC="${param#*=}" ;;
		"--clang++="*) CLANG="${param#*=}" ;;
		"--llvm-sanatize") llvm_sanatizer=true ;;
		"--profile") gprof=true ;;
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

# Seprate arguments
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
	EXEargs="$EXEargs --tmax=1"
	valgrind $EXE $EXEargs
	exit 0
fi

# ================================ LLVM sanatizer ==============================
if [ "$llvm_sanatizer" = "true" ]; then
	# Recompile with sanatizer flags
	if [ "$no_recompile" != "true" ]; then
		make clean && make -j 4 \
		 PROFILING='-fsanitize=undefined -fsanitize=address' CXX=$CXX
	fi	
	echo " === LLVM SANATIZER === "
	EXEargs="$EXEargs --tmax=1"
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
	if [ "$no_recompile" != "true" ]; then
		make clean && make -j 4 PROFILING='-pg' CXX=$CXX
	fi

	# Execute the program
	echo " === EXECUTING === "
	$EXE $EXEargs
	if [ "$?" != "0" ]; then
		exit 1
	fi

	# Create profile information
	echo " === PROFILING === "
	echo "Creating profile information ... analysis.txt"
	gprof QLB gmon.out > analysis.txt
	exit 0
fi

print_help
