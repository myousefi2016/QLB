
QLB is a Dirac Solver using the Quantum Lattice Boltzmann scheme. The goal of this project is to provide an efficient implementation which allows to visualize the solution of the Dirac equation in real-time. To achieve this goal QLB relies heavily on multi-threading and GPU acceleration through CUDA.

<p align="center">
  <img src="https://github.com/thfabian/QLB/blob/master/data/QLBtest.png?raw=true" alt="QLB"/>
</p>



- [libGLEW][libGLEW]
- [libglut][libglut]

If you want to built QLB with CUDA you have to install the [CUDA SDK 7.0][cudasdk] and make sure the environment variables `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `PATH` are set accordingly. (Versions older than CUDA 6.5 might not work as the code make use of C++11 which is only partially supported in prior versions)

To compile the project:

1. Obtain the source `git clone https://github.com/thfabian/QLB`
2. Change into the QLB folder `cd QLB/`

[libGLEW]: http://glew.sourceforge.net/
[libglut]: http://freeglut.sourceforge.net/
[cudasdk]: https://developer.nvidia.com/cuda-downloads
