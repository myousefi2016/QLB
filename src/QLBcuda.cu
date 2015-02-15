/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	[DESCRIPTION]
 *	This file contains all the CUDA kernels and device functions
 */

#include "QLB.hpp"
#include "cuassert.hpp"

void QLB::allocate_device_arrays()
{
	//cuassert( cudaMalloc((void**)&d_spinor_, sizeof(d_spinor_[0])) );
}	

void QLB::free_device_arrays()
{
	//cuassert( cudaFree(d_spinor_) );
	//cudaDeviceReset();
}

void QLB::init_device()
{
	// Print CUDA informations if requested
	if(verbose_)
	{
		std::cout << " === CUDA Info === " << std::endl;
		cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, 0);
		int dvVers = 0; cudaDriverGetVersion(&dvVers);
		int rtVers = 0; cudaRuntimeGetVersion(&rtVers);
		unsigned mem = (unsigned)deviceProp.totalGlobalMem;
		std::printf("CUDA Driver Version:  %d.%d\n", dvVers/1000, dvVers%100);
		std::printf("CUDA Runtime Version: %d.%d\n", rtVers/1000, rtVers%100);
		std::printf("Total GPU memory:     %u bytes\n", mem);
	}

	//cuassert( cudaMemcpy(d_spinor_, spinor1_.data() , sizeof(spinor1_[0]), cudaMemcpyHostToDevice) );
	//cuassert( cudaMemcpy(d_spinor_, X.data() , sizeof(X[0]), cudaMemcpyHostToDevice) );
}

void QLB::get_device_arrays()
{
	//cuassert( cudaMemcpy(spinor1_.data() , d_spinor_, sizeof(spinor1_[0]), cudaMemcpyDeviceToHost) );
}
