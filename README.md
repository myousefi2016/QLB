# Quantum Lattice Boltzmann #

QLB is a 2D Dirac Solver using the Quantum Lattice Boltzmann scheme. The goal of this project is to provide an efficient implementation which allows to visualize the solution of the 2D Dirac equation in real-time using OpenGL and CUDA. This project is part of my bachelor thesis in computational science and engineering at ETH Zürich.

<p align="center">
  <img src="https://github.com/thfabian/QLB/blob/master/misc/QLBtest.png?raw=true" alt="QLB"/>
</p>

# Building #

### Linux ###

To build QLB on Linux the following libraries are required:
- [libGLEW][libGLEW]
- [libglut][libglut]

All those libraries should be present in the repositories of your distribution. Alternatively [libGLEW][libGLEW] can be built directly within this project with `make libGLEW` (git required).
If you want to built QLB with CUDA you have to install the [CUDA SDK 7.0][cudasdk] and make sure the environment variables `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly. (Versions older than CUDA 7.0 might not work as the code makes use of C++11 which is only partially supported in prior versions).

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`
3. Compile with `make` which builds QLB by default with CUDA and assumes your CUDA installation is residing in `/usr/local/cuda/`. To disable CUDA run `make CUDA=false` instead.

<span style="color:red">Note:</span> Ubuntu 14.04/14.10 might suffer from a linker regression that is exposed when linking against the system OpenGL library (Mesa) but running the application against the NVIDIA one. The bug can be fixed by directly linking against the OpenGL library from NVIDIA by adding `-L/usr/lib/nvidia-346/` to beginning of the `LDFLAGS` in the Makefile. See [here][bug1248642] for further information.

### Mac OS X ###

To build QLB on Mac OS X the following libraries are required:
- [libGLEW][libGLEW]

You can build [libGLEW][libGLEW] directly within this project with `make libGLEW` (git required).
If you want to built QLB with CUDA you have to install the [CUDA SDK 7.0][cudasdk] and make sure the environment variables `DYLD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly. (Versions older than CUDA 7.0 might not work as the code makes use of C++11 which is only partially supported in prior versions).

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`
3. Install libGLEW if you haven't already `make libGLEW`
4. Compile with `make` which builds QLB by default with CUDA and assumes your CUDA installation is residing in `/usr/local/cuda/`. To disable CUDA run `make CUDA=false` instead.

### Windows ###
To build QLB on Windows you should use the Visual Studio 2012 project (`QLBwinVS2012.sln`). You can either build QLB with/without CUDA depending on the configuration mode. The CUDA build requires the [CUDA SDK 7.0][cudasdk] Visual Studio integration.


There are precompiled binaries `QLBwin.exe` and `QLBwin-no-cuda.exe` in `bin/Windows/32-bit` and `bin/Windows/64-bit`. The CUDA binaries are compiled with CUDA 7.0 and therefore require a NVIDIA Driver of version 346 or higher.

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
[cudasdk]: https://developer.nvidia.com/cuda-downloads
[bug1248642]: https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-319/+bug/1248642
