/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zürich
 *
 *	Function to handle CUDA runtime errors.
 *	The function follows the same API as the standard cassert.
 *	
 * [EXAMPLE] 
 *
 *	cuassert( cudaMalloc((void**)&device_array, 1024) );
 *
 */

#ifndef CUASSERT_HPP
#define CUASSERT_HPP

// System includes
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

// Do nothing if NDEBUG is defined
#ifndef NDEBUG
 #define cuassert(ans) { _cuassert((ans), __FILE__, __LINE__); }
#else
 #define cuassert(ans) { (ans); }
#endif
static inline void _cuassert(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess) 
	{
		std::cerr << file << ":" << line;
		std::cerr << " CudaError: " << cudaGetErrorString(code) << std::endl;
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#endif /* cuassert.hpp */
