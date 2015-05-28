#!usr/bin python
# -*- coding: utf-8 -*-
#
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
#
#  This example demonstrates how to setup an initial condition in which the 
#  positive energy, spin-up component is a spherically symmetric Gaussian wave 
#  packet with spread delta0.
#
#  To run the generated output:
#  ./QLB --initial=gaussian-128.dat
#

# We first have to include the library path (if you move this script you have to
# adjust the relative path)
import sys, os
import numpy as np
scriptLocation = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scriptLocation+'/../')

# Import the library
import InputGenerator as ig

# Define the system size, spatial discretization, mass and initial spread
dx     = 1.5625
L      = 128
mass   = 0.1
delta0 = 14.0

# Initialize the library
InputObj = ig.InputGenerator(L, dx, mass, delta0)

# Set the spinor0 component (the others are initialzed with 0 by default if
# not explicitly set)
x0 = dx/2.0
y0 = dx/2.0
Ifunc = lambda x,y: np.sqrt(2*np.pi*delta0*delta0)*np.exp( -( ((x-x0)**2 + (y-y0)**2 )/(4*delta0*delta0) ) )
InputObj.setInitial(Ifunc, 0)

# Write the initial condition to an output-file.
InputObj.writeInitial('gaussian-%i.dat' % L)

# You can copy the file afterwards to a convenient location
InputObj.moveToInitialDB(scriptLocation+'/../initial')	
