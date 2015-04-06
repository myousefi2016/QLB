# Quantum Lattice Boltzmann

QLB is a Dirac Solver using the Quantum Lattice Boltzmann scheme. The goal of this project is to provide an efficient implementation which allows to visualize the solution of the Dirac equation in real-time. To achieve this goal QLB relies heavily on multi-threading and GPU acceleration through CUDA.

<p align="center">
  <img src="https://github.com/thfabian/QLB/blob/master/data/QLBtest.png?raw=true" alt="QLB"/>
</p>

# Building

### Linux

To built QLB on Linux the following libraries are required:
- [libGLEW][libGLEW]
- [libglut][libglut]

All those libraries should be present in the repositories of your distribution. Alternatively [libGLEW][libGLEW] can be built directly within this project with `make libGLEW`.
If you want to built QLB with CUDA you have to install the [CUDA SDK][cudasdk] and make sure the environment variables `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly.

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`
3. Compile `make` 

Which builts QLB by default with CUDA and assumes your CUDA installation is residing in `/usr/local/cuda/`. To disable CUDA run `make CUDA=false` instead.

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
[cudasdk]: https://developer.nvidia.com/cuda-downloads
