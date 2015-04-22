#!usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
# 
#  This script will plot the spreads of the particles against time. The script
#  makes use of Python's Matplotlib and NumPy.

# General imports
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def Usage():
	""" Print the usage of this program """
	print "Usage: 	python plot_spread.py [Options]"
	print ""
	print "Options:"
	print "   --file=S         The file(s) containing the data for plotting stored"
	print "                    as a N x k matrix where k indicates the number of"
	print "                    measured spreads (x,y,z) and N represent the number"
	print "                    of performed time steps. Multiple files are possible"
	print "                    if delimited e.g '--file=spread64.dat,spread128.dat'"
	print "   --L=X            Specify the grid size L, multiple values are possible"
	print "                    e.g '--L=64,128' (this is only used for labeling)"
	print "   --no-potential   The last column is interpreted as the exact solution"
	print "   --save           Save the plot as a pdf"
	print "   --help           Print this help statement"
	print ""
	print "Example:"
	print "    python plot_spread --file=spread.dat --noPotential --L=128"
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


# === Input validation ===

# check command-line arguments
if len(sys.argv) < 2 or checkCmdArg('--help'):
	Usage()

# '--no-potential'
noPotential = checkCmdArg('--no-potential')

# '--save'
save = checkCmdArg('--save')

# '--file=S'
filesArgs = checkCmdArg('--file=', True)
if filesArgs[0]:
	files =filesArgs[1][filesArgs[1].find('=')+1:].split(',')
else:
	print sys.argv[0],": error : missing command-line option '--file=S'"
	exit(1)

# '--L=X'
LArg = checkCmdArg('--L=',True)
if LArg[0]:
	L = LArg[1][LArg[1].find('=')+1:].split(',')
else:
	L = ""

if len(sys.argv) >= 2:
	print sys.argv[0],": error : unrecognized command-line option '",sys.argv[1],"'"
	exit(1)

# Read-in data
data = []
for f in files:
	data.append(np.loadtxt(f))

# === Plotting ===
print 'Plotting spread ...',

fig = plt.figure()
fig.suptitle('Quantum Lattice Boltzmann - Spread $\Delta$',fontsize=13)
ax = fig.add_subplot(111)

# There is only 1 file
if len(data) == 1:
	label = ['$\Delta_x$','$\Delta_y$','$\Delta_z$']
	k = np.shape(data[0])[1]
	
	for ks in range(1,k - noPotential):
		ax.plot(data[0][:,0],data[0][:,ks],label=label[ks-1])
	
	if noPotential:
		ax.plot(data[0][:,0],data[0][:,-1],label='$Schroedinger$')
# There are multiple files
else:
	for i in range(0, len(data)):
		if len(L) > i:
			ax.plot(data[i][:,0],data[i][:,1],label='$L='+L[i]+'$')
	
	if noPotential:
		ax.plot(data[0][:,0],data[0][:,-1],label='$Schroedinger$')

ax.set_xlabel('$t$')
ax.set_ylabel('$\Delta$')
plt.legend(loc='best')
plt.show()
print 'Done'

if save:
	fname = 'spread-'+time.strftime('%H-%M-%S')+'.pdf'
	print 'Saving file ... ', fname
	fig.savefig(fname)
