/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Query for different performance counters like CPU/GPU usage, memory usage 
 *  etc. The implementation is highly OS dependant although the interface
 *  is not.
 *
 *  Credit for GPU usage counter on Windows:
 *  Open Hardware Monitor (http://code.google.com/p/open-hardware-monitor) 
 */
 
#ifndef PERFORMANCE_COUNTER_HPP
#define PERFORMANCE_COUNTER_HPP

// System Includes
#ifdef _WIN32
 #include <windows.h>
 #include <psapi.h>
#elif defined(__linux__) 
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/sysinfo.h>
 #include <cstdlib>
 #include <cstdio>
 #include <cstring>
#else
 #include <unistd.h>
#endif

#include <cstring>
#include <exception>

#ifdef QLB_HAS_CUDA
 #include <cuda_runtime.h>
#endif

// Local includes
#include "error.hpp"

class PerformanceCounter
{
public:
	 PerformanceCounter(); 
	 
	 ~PerformanceCounter();
 	 
	/**
	 *	Get the maximal physical cpu memory (in bytes)
	 */
	std::size_t cpu_max_memory() const;
	
	/**
	 *	Get the maximal physical gpu memory (in bytes)
	 */
	std::size_t gpu_max_memory() const;
	
	/**
	 *	Get the physical memory (in bytes) currently used by current process 
	 */
	std::size_t cpu_memory();
	
	/**
	 *	Get the physical memory (in bytes) currently used by current process 
	 */
	std::size_t gpu_memory();
	
	/**
	 * Get the current CPU usage [0,100] by the current process
	 */
	double cpu_usage();
	
	/**
	 * Get the current GPU usage [0,100]
	 */
	double gpu_usage();
	
private:
	std::size_t cpu_max_memory_;
	std::size_t gpu_max_memory_;
	bool GPU_query_failed_;
};

#endif /* PerformanceCounter.hpp */
