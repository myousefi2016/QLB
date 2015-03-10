#!/usr/bin/env sh
#	
# Quantum Lattice Boltzmann 
# (c) 2015 Fabian Thüring, ETH Zürich
#
# This script reproduces the figures (although in 2D) from the paper [1] 
# by passing the appropriate command-line arguments to the program QLB.
#
# [1] Isotropy of three-dimensional quantum lattice Boltzmann schemes, 
#     P.J. Dellar, D. Lapitski, S. Palpacelli, S. Succi, 2011

print_help() 
{
	echo "usage: $0 [options]"
	echo ""
	echo "options:"
	echo "   --fig-1     Print the figure 1 (no potential)"
	echo "   --fig-4     Print the figure 4 (harmonic potential)"
	echo "   --help      Print this help statement"
	echo "   --exe=PATH  Set the path of the QLB executable (e.g --exe=./QLB)"
	echo "   --tmax=X    Set the parameter '--tmax' of QLB to 'X'"
	echo "   --L=X       Set the parameter '--L' of QLB to 'X' (multiple possible"
	echo "               delimit with ',' e.g --L=128,256,512)"
	echo "   --plot      Plot the output with python (executes plot/plot_spread.py)"
	echo "   --save      Save the plot as a pdf"
	echo "   --mt        Run the multi-threaded cpu version of QLB"
	exit 1
}

exit_after_error()
{
	echo "$1"
	exit 1
}

QLB_exe=./QLB

print_fig_1=false
print_fig_4=false
tmax_arg="--tmax=256"
mt_arg="";
L_list="128"
plot_arg=""
plot_files=""
python_flags=""

# Handle command-line arguments
if [ "$#" =  "0" ] 
then
   print_help 
fi

for param in $*
	do case $param in
		"--fig-1")  print_fig_1=true ;
		            python_flags="$python_flags --no-potential" ;;
		"--fig-4")  print_fig_4=true ;;
		"--help")   print_help ;;
		"--plot")   plot_arg="--plot=spread" ;;
		"--save")   python_flags="$python_flags --save" ;;
		"--exe="*)  QLB_exe="${param##*=}" ;;
		"--tmax="*) tmax_arg="$param" ;;
		"--L="*)    L_list="${param##*=}" ;;
		"--mt")     mt_arg="--device=cpu-thread";; 
		*) 	print_help ;; 
	esac
done

# Check if the executable exists/is valid
$QLB_exe --version > /dev/null 2>&1
if [ "$?" != "0" ]; then
	exit_after_error "$0 : error : executable '$QLB_exe' is not valid"
fi

# Set the appropriate command-line arguments
if [ "$print_fig_1" = "true" ]; then
	QLB_args="--dx=0.78125 --dt=0.78125 --mass=0.35 --V=free \
              --gui=none $tmax_arg $plot_arg $mt_arg"
elif [ "$print_fig_4" = "true" ]; then
	QLB_args="--dx=1.5625 --dt=1.5625 --mass=0.1 --V=harmonic \
	          --gui=none $tmax_arg $plot_arg $mt_arg"
else
	exit_after_error "$0 : error : no figure specified try '$0 --help'"
fi

# Run the simulation (for each L)
for L in $(echo $L_list | tr "," "\n")
do
	$QLB_exe --L=$L $QLB_args
	if [ "$plot_arg" != "" ]; then
		mv spread.dat "spread$L.dat"
		plot_files="$plot_files,spread$L.dat"
	fi
done

# Check for runtime errors in QLB
if [ "$?" != "0" ]; then
	exit 1
fi

# Plotting
if [ "$plot_arg" != "" ]; then
	python_flags="--file=${plot_files#?} $python_flags --L=$L_list"
	python plot/plot_spread.py $python_flags
fi
