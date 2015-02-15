# Quantum Lattice Boltzmann

Dirac Solver using the Quantum Lattice Boltzmann scheme and 3D visualisation with OpenGL.

# Building

QLB can be built with or without [CUDA][cudasdk] support.

### Linux

To build the project on Linux make sure the following libraries are installed:
- libGL
- libGLU
- [libGLEW][libGLEW]
- [libglut][libglut]

All other libraries should be present in the repositories of your distribution. Alternatively [libGLEW][libGLEW] can be built directly within this project with `make libGLEW`. In addition, if you plan to build against CUDA make sure you have installed the [CUDA SDK][cudasdk] properly i.e setup `LD_PATH` and `LIBRARY_PATH`.

To compile the project change in to the folder with the Makefile `QLB/` and type:

`make`

or building without CUDA support

`make NO_CUDA=true`

### Max OS X

To build the project on Mac OS X make sure the following library is installed:
- [libGLEW][libGLEW] 

You can build [libGLEW][libGLEW] directly within this project with `make libGLEW`. Any other dependencies should already be installed. In addition, if you plan to build against CUDA make sure you have installed the [CUDA SDK][cudasdk].

To compile the project change in to the folder with the Makefile `QLB/` and type:

`make`

or building without CUDA support

`make NO_CUDA=true`

### Windows

There are precompiled binaries in `bin/Windows/32-bit` and `bin/Windows/64-bit` respectively.
To build the project from source you can use the Visual Studio 2012 project with the solution `QLBwinVS2012`.
To build against CUDA you need the VisualStudio extension for CUDA shipped with the [CUDA SDK][cudasdk].
All libraries and headers needed during the compilation are residing in `inc/Windows/` and `lib/Windows/`.

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
[cudasdk]: https://developer.nvidia.com/cuda-downloads
