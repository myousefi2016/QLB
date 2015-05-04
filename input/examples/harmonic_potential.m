%  Quantum Lattice Boltzmann 
%  (c) 2015 Fabian Thüring, ETH Zürich
% 
%  This example demonstrates how to setup a harmonic potential using the QLB
%  InputGenerator class. For a detailed documentation of all the function 
%  look at python/InputGenerator.py
% 
%  To run the generated output:
%  ./QLB --potential=harmonic-128.dat

% We first have to include the library path (if you move this script you 
% have to adjust the relative path)
addpath ../

% Define the system size, spatial discretization, mass and initial spread
L      = 128;
dx     = 1.5625;
mass   = 0.1;
delta0 = 14.0;

% Initialize the library
ig = InputGenerator(L, dx, mass, delta0);

% There are two ways of specifying the potential: Either you define a lambda
% function which will then be used to evaluate the potential at the grid 
% points or you pass in the evaluated potential at the grid points directly.
% 
% See also: setPotential, setPotentialArray
 
Vfunc = @(x,y) -1.0/2*mass*( 1.0/(2*mass * delta0^2) )^2 * (x*x + y*y);
ig = setPotential(ig, Vfunc);

% which is aquivalent to:
%  w0 = 1.0/(2*mass * delta0^2);
%  Vfunc = @(x,y, var) -1.0/2 * var{1} * var{2}^2 * (x*x + y*y);
%  ig = setPotential(ig, Vfunc, mass, w0);

% Write the potential to an output-file. Optionally one can also specify a 
% title and a descrption of the potential.
ig = writePotential(ig, sprintf('harmonic-%i.dat',L));

% You can copy the file afterwards to a convenient location
ig = moveToPotentialDB(ig, strcat(pwd, '/../potential'));