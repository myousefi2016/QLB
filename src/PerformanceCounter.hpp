/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Query for different performance counters like CPU/GPU usage, memory usage 
 *  etc. The implementation is highly OS dependant although the interface
 *  is not.
 *
 *  Credit for GPU usage counter:
 *  Open Hardware Monitor (http://code.google.com/p/open-hardware-monitor) 
 */
 
#ifndef PERFORMANCE_COUNTER_HPP
#define PERFORMANCE_COUNTER_HPP

// By default we assume Intel HyperThreading
#define PC_HAS_HT 1

// System Includes
#ifdef _WIN32
 #include <windows.h>
 #include <psapi.h>
#else 
 #include <unistd.h>
#endif 

#include <cstring>
#include <exception>

// Local includes
#include "error.hpp"

class PerformanceCounter
{
public:
	 PerformanceCounter(); 
	 
	 ~PerformanceCounter();
 	 
	/**
	 *	Get the maximal physical memory (in bytes)
	 */
	std::size_t max_memory() const;
	
	/**
	 *	Get the physical memory (in bytes)currently used by current process 
	 */
	std::size_t used_memory();
	
	/**
	 * Get the current CPU usage [0,100] by the current process
	 */
	double cpu_usage();
	
	/**
	 * Get the current GPU usage [0,100]
	 */
	double gpu_usage();
	
	// == Getter ===
	inline int num_processor() const { return num_processor_; }
	
private:
	std::size_t max_memory_;
	int num_processor_;
	bool GPU_query_failed_;
};

#endif /* PerformanceCounter.hpp */
