#!usr/bin python
# -*- coding: utf-8 -*-
#
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
#
#  This example demonstrates how to setup a coulomb potential using the QLB
#  InputGenerator library. For a detailed documentation of all the function 
#  look at python/InputGenerator.py
#
#  To run the generated output:
#  ./QLB --potential=coulomb-100.dat

# We first have to include the library path (if you move this script you have to
# adjust the relative path)
import sys, os
scriptLocation = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scriptLocation+'/../')

# Import the library
import InputGenerator as ig

# Define the system size, spatial discretization
dx     = 1.5625
L      = 1024

# Initialize the library
InputObj = ig.InputGenerator(L, dx)

# There are two ways of specifying the potential: Either you define a lambda
# function which will then be used to evaluate the potential at the grid points
# or you pass in the evaluated potential at the grid points directly.
#
# See also: setPotential, setPotentialArray

Vfunc = lambda x,y: -1.0*L*L/(x*x + y*y)
InputObj.setPotential(Vfunc)

# Write the potential to an output-file. Optionally one can also specify a title
# and a descrption of the potential.
InputObj.writePotential('coulomb-%i.dat' % L)

# You can copy the file afterwards to a convenient location
InputObj.moveToPotentialDB(scriptLocation+'/../potential')
