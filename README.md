# Quantum Lattice Boltzmann #

QLB is a Dirac Solver using the Quantum Lattice Boltzmann scheme. The goal of this project is to provide an efficient implementation which allows to visualize the solution of the Dirac equation in real-time. To achieve this goal QLB relies heavily on multi-threading and GPU acceleration through CUDA.

<p align="center">
  <img src="https://github.com/thfabian/QLB/blob/master/data/QLBtest.png?raw=true" alt="QLB"/>
</p>

# Building #

### Linux ###

To build QLB on Linux the following libraries are required:
- [libGLEW][libGLEW]
- [libglut][libglut]

All those libraries should be present in the repositories of your distribution. Alternatively [libGLEW][libGLEW] can be built directly within this project with `make libGLEW` (git required).
If you want to built QLB with CUDA you have to install the [CUDA SDK 7.0][cudasdk] and make sure the environment variables `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly. (Versions older than CUDA 6.5 might not work as the code makes use of C++11 which is only partially supported in prior versions).

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`
3. Compile with `make` which builds QLB by default with CUDA and assumes your CUDA installation is residing in `/usr/local/cuda/`. To disable CUDA run `make CUDA=false` instead.

### Mac OS X ###

To build QLB on Mac OS X the following libraries are required:
- [libGLEW][libGLEW]

You can build [libGLEW][libGLEW] directly within this project with `make libGLEW` (git required).
If you want to built QLB with CUDA you have to install the [CUDA SDK 7.0][cudasdk] and make sure the environment variables `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly. (Versions older than CUDA 6.5 might not work as the code makes use of C++11 which is only partially supported in prior versions).

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`
3. Install libGLEW if you haven't already `make libGLEW`
4. Compile with `make` which builds QLB by default with CUDA and assumes your CUDA installation is residing in `/usr/local/cuda/`. To disable CUDA run `make CUDA=false` instead.

### Windows ###

To build QLB on Windows you should use the Visual Studio 2012 project (QLBwinVS2012.sln). The current version can only build QLB __with__ CUDA and therefore requires the [CUDA SDK 7.0][cudasdk] Visual Studio integration.


There are precompiled binaries in `bin/Windows/32-bit` and `bin/Windows/64-bit`. These binaries are compiled with CUDA 7.0 and therefore require a NVIDIA Driver of version >= 346.

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
[cudasdk]: https://developer.nvidia.com/cuda-downloads
