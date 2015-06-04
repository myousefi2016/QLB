#!usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
# 
#  This script will plot the runtime against system sizes. The script
#  makes use of Python's Matplotlib and NumPy.

# General imports
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def Usage():
	""" Print the usage of this program """
	print "Usage: 	python plot_benchmark.py [Options]"
	print ""
	print "Options:"
	print "     --cpu-serial=FILE    File with the cpu-serial data stored as (N x 2)"
	print "     --cpu-thread=FILE    File with the cpu-thread data stored as (N x 2)"
	print "     --cuda=FILE          File with the cuda data stored as (N x 2)"
	print "     --save               Save the plot as a pdf"
	print "     --timings            Plot the timings vs. system size [default]"
	print "     --speedup            Plot the speedup vs. system size"
	print "     --help               Print this help statement"
	sys.exit(1)

def checkCmdArg(arg, is_multi = False):
	""" Check if the command-line argument 'arg' was passed """
	if is_multi:
		for args in sys.argv:
			if args.find(arg) != -1:
				sys.argv.remove(args)
				return [True,args]
		return [False,""]
	else:
		if arg in sys.argv:
			sys.argv.remove(arg)
			return True 
		else:
			return False

def getData(command):
	file = command[command.find('=')+1:]
	try:
		return np.loadtxt(file)
	except:
		print sys.argv[0],": error : '",file,"' is invalid"
		sys.exit(1)

# === Input validation ===

# check command-line arguments
if len(sys.argv) < 2 or checkCmdArg('--help'):
	Usage()

# '--save'
save = checkCmdArg('--save')

speedup = checkCmdArg('--speedup')
if speedup:
	timings = checkCmdArg('--timings')
else:
	timings = True

# '--cpu-serial=FILE'
cpuSerialArg  = checkCmdArg('--cpu-serial=', True)
cpuSerialData = []
if cpuSerialArg[0]:
	cpuSerialData = getData(cpuSerialArg[1])

# '--cpu-thread=FILE'
cpuThreadArg  = checkCmdArg('--cpu-thread=', True)
cpuThreadData = []
if cpuThreadArg[0]:
	cpuThreadData = getData(cpuThreadArg[1])
	
# '--cuda=FILE'
cudaArg  = checkCmdArg('--cuda=', True)
cudaData = []
if cudaArg[0]:
	cudaData = getData(cudaArg[1])	

if len(sys.argv) >= 2 and '--timings' not in sys.argv:
	print sys.argv[0],": error : unrecognized command-line option '",sys.argv[1],"'"
	exit(1)
	
markerSize=5

# === Plotting ===

if timings:
	print 'Plotting timings ...',

	# Plot timings
	fig = plt.figure()
#	fig.suptitle('Quantum Lattice Boltzmann - Benchmark',fontsize=13)
	ax = fig.add_subplot(111)

	if cpuSerialArg[0]:
		ax.plot(cpuSerialData[:,0],cpuSerialData[:,1],'-o',label=r'CPU Serial',\
		markersize=markerSize)
	if cpuThreadArg[0]:
		ax.plot(cpuThreadData[:,0],cpuThreadData[:,1],'-o',label=r'CPU Thread',\
		markersize=markerSize)
	if cudaArg[0]:
		ax.plot(cudaData[:,0],cudaData[:,1],'-o',label=r'CUDA',markersize=markerSize)

	ax.set_xlabel('system size $N$')
	ax.set_ylabel('time $s$')
	plt.legend(loc='upper left',fontsize=11)
	plt.grid()
	
	if save:
		print 'Done'
		fname = 'benchmark-'+time.strftime('%H-%M-%S')+'.pdf'
		print 'Saving file ... ', fname
		fig.savefig(fname)
	else:
		plt.show()
		print 'Done'
	
# Plot speedup
if speedup:
	print 'Plotting speedup ...',

	fig = plt.figure()
#	fig.suptitle('Quantum Lattice Boltzmann - Speedup',fontsize=13)
	ax = fig.add_subplot(111)

	if cpuSerialArg[0]:
		ax.plot(cpuSerialData[:,0],cpuSerialData[:,1]/cpuSerialData[:,1],\
		'-o',label=r'CPU Serial',markersize=markerSize)
	if cpuThreadArg[0]:
		ax.plot(cpuThreadData[:,0],cpuSerialData[:,1]/cpuThreadData[:,1],\
		'-o',label=r'CPU Thread',markersize=markerSize)
	if cudaArg[0]:
		ax.plot(cudaData[:,0],cpuSerialData[:,1]/cudaData[:,1],\
		'-o',label=r'CUDA',markersize=markerSize)

	ax.set_xlabel('system size $L$')
	ax.set_ylabel('speedup')
	plt.legend(loc='upper right',fontsize=11)
	plt.grid()

	if save:
		print 'Done'
		fname = 'speedup-'+time.strftime('%H-%M-%S')+'.pdf'
		print 'Saving file ... ', fname
		fig.savefig(fname)
	else:
		plt.show()
		print 'Done'
