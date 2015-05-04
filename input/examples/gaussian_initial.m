%  Quantum Lattice Boltzmann 
%  (c) 2015 Fabian Thüring, ETH Zürich
% 
%  This example demonstrates how to setup an initial condition in which the 
%  positive energy, spin-up component is a spherically symmetric Gaussian wave 
%  packet with spread delta0.
%
%  To run the generated output:
%  ./QLB --initial=gaussian-128.dat

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

% Set the spinor0 component (the others are initialzed with 0 by default) 
x0 = dx/2.0;
y0 = dx/2.0;
Ifunc = @(x,y) exp( -( (x-x0)^2 + (y-y0)^2 )/(4*delta0*delta0) );
ig = setInitial(ig, Ifunc, 0);

% Write the potential to an output-file. Optionally one can also specify a 
% title and a descrption of the potential.
ig = writeInitial(ig, sprintf('gaussian-%i.dat',L));

% You can copy the file afterwards to a convenient location
ig = moveToInitialDB(ig, strcat(pwd, '/../initial'));
