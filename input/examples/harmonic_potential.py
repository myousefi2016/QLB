#!usr/bin python
# -*- coding: utf-8 -*-
#
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
#
#  This example demonstrates how to setup a harmonic potential using the QLB
#  InputGenerator library. For a detailed documentation of all the function 
#  consider to take a look at python/InputGenerator.py


# We first have to include the library path (if you move this script you have to
# adjust the relative path)
import sys, os
scriptLocation = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scriptLocation+'/../../python/')

# Import the library
import InputGenerator as ig

# Define the system size, spatial discretization, mass and initial spread
dx     = 1.5625
L      = 4
mass   = 0.1
delta0 = 14.0

# Initialize the library
InputObj = ig.InputGenerator(L, dx, mass, delta0)

# There are two ways of specifying the potential: Either you define a lambda
# function which will then be used to evaluate the potential at the grid points
# or you pass in the evaluated potential at the grid points directly.
#
# See also: setPotential, setPotentialArray

Vfunc = lambda x,y: 1.0/2*mass*( 1.0/(2*mass * delta0**2) )**2 * (x*x + y*y)
InputObj.setPotential(Vfunc)

# which is aquivalent to:
#  w0 = 1.0/(2*mass * delta0**2)
#  Vfunc = lambda x,y, m, w0: 1.0/2 * m * w0**2 * (x*x + y*y)
#  InputObj.setPotential(Vfunc, mass, w0)

# Write the potential to an output-file. Optionally one can also specify a title
# and a descrption of the potential.
InputObj.writePotential('potential-harmonic-%i.dat' % L)
InputObj.moveToPotentialDB(scriptLocation+'/../potential')
