# Quantum Lattice Boltzmann
Dirac Solver using the Quantum Lattice Boltzmann scheme.

# Building

Depending on your OS there are diffrent ways of building QLB

### Linux

To build the project on Linux make sure the following libraries are installed:
- libGL
- libGLU
- [libGLEW] [libGLEW]
- [libglut] [libglut]

All those libraries should be present in the repositories of your distribution. `libGLEW` can be built directly within this project with `make libGLEW`. 

To compile the project change in to the folder with the Makefile `QLB/' and type:

`make`

### Max OS X

To build the project on Mac OS X make sure the following library is installed:
- [libGLEW] [libGLEW] 

Alternatively `libGLEW` can be built directly within this project with `make libGLEW`. 

To compile the project change in to the folder with the Makefile `QLB/' and type:

`make`

### Windows

There are precompiled binaries for 32 and 64-bit in `bin/Windows/32-bit` and `bin/Windows/64-bit` respectively.
To build the project from source you can use the Visual Studio 2012 project with the solution `QLBwinVS2012'.
All libraries and headers needed during the compilation are residing in `inc/Windows/' and 'lib/Windows/'.

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
