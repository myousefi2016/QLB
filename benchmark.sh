#!/bin/sh
#	
# Quantum Lattice Boltzmann 
# (c) 2015 Fabian Thüring, ETH Zürich
#
# This script benchmarks all the available compute models of QLB by running QLB
# with diffrent system sizes.

print_help() 
{
	echo "usage: $0 [options]"
	echo ""
	echo "options:"
	echo "   --Lmin=X        Set the minimal system size of the benchmark [default: 64]"
	echo "   --Lmax=X        Set the maximum system size of the benchmark [default: 1024]"
	echo "   --plot-timings  Plot system size vs. runtime with python"
	echo "   --plot-speedup  Plot system size vs. speedup with python"
	echo "   --save          Save the plot(s) as a pdf instead of showing them"
	echo "   --exe=PATH      Set the path of the QLB executable (e.g --exe=./QLB)"
	echo "   --bc=PATH       Set the path to bc (GNU Basic Calculator) [default : bc]"
	exit 1
}

exit_after_error()
{
	echo "$1"
	exit 1
}

QLB_exe=./QLB
QLB_args="--disable-progressbar --gui=none"

FILE_cpu_serial=data_cpu_serial.dat
FILE_cpu_thread=data_cpu_thread.dat
FILE_cuda=data_cuda.dat

plot=false
python_flags=""
Lmin=64
Lmax=1024
bc=bc

# Handle command-line arguments
for param in $*
	do case $param in
		"--plot-timings") plot=true ;
		                  python_flags="$python_flags --timings" ;;
		"--plot-speedup") plot=true ;
		                  python_flags="$python_flags --speedup" ;;
		"--Lmin="*) Lmin="${param##*=}" ;;
		"--Lmax="*) Lmax="${param##*=}" ;;
		"--save")   python_flags="$python_flags --save" ;;
		"--exe="*)  QLB_exe="${param##*=}" ;;
		"--bc="*) bc="${param##*=}" ;;
		*) 	print_help ;; 
	esac
done

# Check if the executable exists/is valid
$QLB_exe --version > /dev/null 2>&1
if [ "$?" != "0" ]; then
	exit_after_error "$0 : error : executable '$QLB_exe' is not valid"
fi

# Check if bc is working
skip_speedup=false
if [ $($bc --version | grep -c bc) != "1" ]; then
	echo "$0 : warning : calculator '$bc' not found. Skipping speedup calculations"
	skip_speedup=true
fi

# Check if QLB was compiled with CUDA
CUDA_is_present=true
if [ $($QLB_exe --version | grep -c CUDA) != "1" ]; then
	CUDA_is_present=false
fi

if [ "$plot" = "true" ]; then
	rm -f $FILE_cpu_serial $FILE_cpu_thread $FILE_cuda
fi

# ================================ Benchmarking ================================

echo " === CPU serial ==="
L=$Lmin
i=0

while [ $L -le $Lmax ]
do
	t=$($QLB_exe --L=$L $QLB_args --device=cpu-serial | grep -oP ':\s*\K\d+\.\d+')
	printf "%5i     %5.5f s\n" $L $t
	
	if [ "$plot" = "true" ]; then
		printf "%5i     %5.5f\n" $L $t >> $FILE_cpu_serial
	fi
	
	time_serial="$time_serial $t"
	L=$(expr $L '*' 2)
done

echo " === CPU thread ==="
L=$Lmin
i=1
while [ $L -le $Lmax ]
do
	t=$($QLB_exe --L=$L $QLB_args --device=cpu-thread | grep -oP ':\s*\K\d+\.\d+')
	
	# calculate speedup if possible
	time_serial_i=$(echo $time_serial | cut -d" " -f $i)
	
	if [ "$skip_speedup" = "false" ]; then
		speedup=$(echo "scale=2; $time_serial_i/$t" | $bc)
		printf "%5i     %5.5f s    %3.1f\n" $L $t $speedup
	else
		printf "%5i     %5.5f s\n" $L $t
	fi
	
	if [ "$plot" = "true" ]; then
		printf "%5i     %5.5f\n" $L $t >> $FILE_cpu_thread
	fi
	
	L=$(expr $L '*' 2)
	i=$(expr $i '+' 1)
done

if [ "$CUDA_is_present" = "true" ]; then
	echo " === CUDA ==="
	L=$Lmin
	i=1
	while [ $L -le $Lmax ]
	do
		t=$($QLB_exe --L=$L $QLB_args --device=gpu | grep -oP ':\s*\K\d+\.\d+')
	
		# calculate speedup if possible
		time_serial_i=$(echo $time_serial | cut -d" " -f $i)
		
		if [ "$skip_speedup" = "false" ]; then
			speedup=$(echo "scale=2; $time_serial_i/$t" | $bc)
			printf "%5i     %5.5f s    %3.1f\n" $L $t $speedup
		else
			printf "%5i     %5.5f s \n" $L $t 
		fi
		
		if [ "$plot" = "true" ]; then
			printf "%5i     %5.5f\n" $L $t >> $FILE_cuda
		fi
	
		L=$(expr $L '*' 2)
		i=$(expr $i '+' 1)
	done
fi

# Plotting
if [ "$plot" = "true" ]; then
	python_flags="$python_flags --cpu-serial=$FILE_cpu_serial \
	             --cpu-thread=$FILE_cpu_thread"
	
	if [ "$CUDA_is_present" = "true" ]; then
		python_flags="$python_flags --cuda=$FILE_cuda"
	fi
	
	python python/plot_benchmark.py $python_flags
fi
