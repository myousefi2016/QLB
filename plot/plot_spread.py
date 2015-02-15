#!usr/bin/env python
# -*- coding: utf-8 -*-
# 
#	Quantum Lattice Boltzmann 
#	(c) 2015 Fabian Thüring, ETH Zürich
# 
#	This script will plot the spreads of a particle against time. The script
#	makes use of Python's Matplotlib and NumPy.
#
#	Usage: 	python plot_spread.py INPUT_FILE [options]
#	
#	options:
#		INPUT_FILE        The file containing the data for plotting stored
#		                  as N x k matrix where k indicates the number of 
#		                  measured spreads (x,y,z) and N represent the number.
#		                  of performed time steps. 
#		--no-potential    The last column is interpreted as the exact solution
#		--help            Print this help statement

# General imports
import numpy as np
import sys
import matplotlib.pyplot as plt


# === Input validation ===

def Usage():
	""" Print the usage of this program """
	print "Usage: 	python plot_spread.py INPUT_FILE [options]"
	print ""
	print "options:"
	print "   INPUT_FILE       The file containing the data for plotting stored"
	print "                    as a N x k matrix where k indicates the number of"
	print "                    measured spreads (x,y,z) and N represent the number."
	print "                    of performed time steps."
	print "   --no-potential   The last column is interpreted as the exact solution"
	print "   --help           Print this help statement."
	sys.exit(1)

def checkCmdArg(arg):
	""" Check if the command-line argument 'arg' was passed """
	if arg in sys.argv:
		sys.argv.remove(arg)
		return True
	else:
		return False

# check command-line arguments
if len(sys.argv) < 2 or checkCmdArg('--help'):
	Usage()

# Read-in data
inputFile = sys.argv[1]
data = np.loadtxt(inputFile)

noPotential = checkCmdArg('--no-potential')
k = np.shape(data)[1]
label = ['$\Delta_x$','$\Delta_y$','$\Delta_z$','$Schroedinger$']

# === Plotting ===
print 'Plotting spread ...',

fig = plt.figure()
fig.suptitle('Quantum Lattice Boltzmann - Spread $\Delta$',fontsize=13)
ax = fig.add_subplot(111)

for ks in range(1,k - noPotential):
	ax.plot(data[:,0],data[:,ks],label=label[ks-1])

if noPotential:
	ax.plot(data[:,0],data[:,-1],label=label[-1])
	
ax.set_xlabel('$t$')
ax.set_ylabel('$\Delta$')
plt.legend(loc='best')
#fig.savefig('spread.pdf')
plt.show()

print 'Done'
