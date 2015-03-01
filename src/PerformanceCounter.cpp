/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Query for different performance counters like CPU/GPU usage, memory usage 
 *  etc. The implementation is highly OS dependent although the interface
 *  is not.
 *
 *  Credit for GPU usage counter:
 *  Open Hardware Monitor (http://code.google.com/p/open-hardware-monitor) 
 */

#include "PerformanceCounter.hpp"
 
#ifdef _WIN32

// === GPU ===
#define NVAPI_MAX_PHYSICAL_GPUS   64
#define NVAPI_MAX_USAGES_PER_GPU  34

static int          gpuCount = 0;
static int*         gpuHandles[NVAPI_MAX_PHYSICAL_GPUS] = { NULL };
static unsigned int gpuUsages[NVAPI_MAX_USAGES_PER_GPU] = { 0 };
 
// function pointer types
typedef int* (*NvAPI_QueryInterface_t)(unsigned int offset);
typedef int  (*NvAPI_Initialize_t)();
typedef int  (*NvAPI_EnumPhysicalGPUs_t)(int **handles, int *count);
typedef int  (*NvAPI_GPU_GetUsages_t)(int *handle, unsigned int *usages);

// nvapi.dll internal function pointers
NvAPI_QueryInterface_t      NvAPI_QueryInterface     = NULL;
NvAPI_Initialize_t          NvAPI_Initialize         = NULL;
NvAPI_EnumPhysicalGPUs_t    NvAPI_EnumPhysicalGPUs   = NULL;
NvAPI_GPU_GetUsages_t       NvAPI_GPU_GetUsages      = NULL;

// === CPU ===
struct CPUstats
{
	ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
	HANDLE process;
};

CPUstats* CPU;
 
PerformanceCounter::PerformanceCounter()
:	max_memory_(0), num_processor_(0), GPU_query_failed_(false)
{
	// maximum memory
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	if( GlobalMemoryStatusEx(&memInfo) == 0)
		WARNING("GlobalMemoryStatusEx failed")
	else
		max_memory_ =  std::size_t(memInfo.ullTotalPhys);
	
	// === CPU ===
	CPU = new CPUstats;
	SYSTEM_INFO sysInfo;
	FILETIME ftime, fsys, fuser;

	// Unfortunately there is no way I now of to detect the physical
	// processors we just assume HyperThreading and divide the cores by 2
 	GetSystemInfo(&sysInfo);
	num_processor_ = sysInfo.dwNumberOfProcessors;
#ifdef PC_HAS_HT
	num_processor_ /= 2;
#endif
	
	GetSystemTimeAsFileTime(&ftime);
	memcpy(&CPU->lastCPU, &ftime, sizeof(FILETIME));
	
	CPU->process = GetCurrentProcess();
	GetProcessTimes(CPU->process, &ftime, &ftime, &fsys, &fuser);
	memcpy(&CPU->lastSysCPU, &fsys, sizeof(FILETIME));
	memcpy(&CPU->lastUserCPU, &fuser, sizeof(FILETIME));
	
	// === GPU ===
	try
	{
#ifdef _WIN64
		HMODULE hmod = LoadLibraryA("nvapi64.dll");
#else
		HMODULE hmod = LoadLibraryA("nvapi.dll");
#endif
		if(hmod == NULL)
			throw 1;

		// nvapi_QueryInterface is a function used to retrieve other internal 
		// functions in nvapi.dll
		NvAPI_QueryInterface = (NvAPI_QueryInterface_t) GetProcAddress(hmod, 
		                                                "nvapi_QueryInterface");

		// some useful internal functions that aren't exported by nvapi.dll
		NvAPI_Initialize = (NvAPI_Initialize_t) (*NvAPI_QueryInterface)(0x0150E828);
		NvAPI_EnumPhysicalGPUs = (NvAPI_EnumPhysicalGPUs_t) (*NvAPI_QueryInterface)(0xE5AC921F);
		NvAPI_GPU_GetUsages = (NvAPI_GPU_GetUsages_t) (*NvAPI_QueryInterface)(0x189A1FDF);

		if (NvAPI_Initialize == NULL || NvAPI_EnumPhysicalGPUs == NULL ||
			NvAPI_EnumPhysicalGPUs == NULL || NvAPI_GPU_GetUsages == NULL)
			throw 1; 
			
		(*NvAPI_Initialize)();

		// gpuUsages[0] must be this value, otherwise NvAPI_GPU_GetUsages won't work
		gpuUsages[0] = (NVAPI_MAX_USAGES_PER_GPU * 4) | 0x10000;
		(*NvAPI_EnumPhysicalGPUs)(gpuHandles, &gpuCount);
	}
	catch(...)
	{
#ifdef _WIN64
		WARNING("Cannot find 'nvapi64.dll'");
#else
		WARNING("Cannot find 'nvapi.dll'");
#endif
		GPU_query_failed_ = true;
	}
}

PerformanceCounter::~PerformanceCounter()
{
	delete CPU;
}

std::size_t PerformanceCounter::max_memory() const
{
	return max_memory_;
}

std::size_t PerformanceCounter::used_memory()
{
	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, 
	                     sizeof(pmc));
	return std::size_t(pmc.WorkingSetSize);
}

double PerformanceCounter::cpu_usage()
{
 	FILETIME ftime, fsys, fuser;
	ULARGE_INTEGER now, sys, user;
	double percent = 0;

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&now, &ftime, sizeof(FILETIME));

	GetProcessTimes(CPU->process, &ftime, &ftime, &fsys, &fuser);
	memcpy(&sys, &fsys, sizeof(FILETIME));
	memcpy(&user, &fuser, sizeof(FILETIME));
	
	percent =   double((sys.QuadPart  - CPU->lastSysCPU.QuadPart) +
		               (user.QuadPart - CPU->lastUserCPU.QuadPart));

	// Make sure we do not divide by 0 if we have no contention
	double den = double(now.QuadPart - CPU->lastCPU.QuadPart);
	percent = den != 0.0 ? percent/den : 0.0;
	
	percent /= num_processor_;
	CPU->lastCPU = now;
	CPU->lastUserCPU = user;
	CPU->lastSysCPU = sys;
	
	return (double) percent * 100;
}

double PerformanceCounter::gpu_usage()
{	
	if(GPU_query_failed_)
		return 0;
	else 
	{
		(*NvAPI_GPU_GetUsages)(gpuHandles[0], gpuUsages);
		return gpuUsages[3];
	}
}
 
#elif defined(__linux__) /* Linux */

PerformanceCounter::PerformanceCounter()
:	max_memory_(0), num_processor_(0), GPU_query_failed_(false)
{
	
}

PerformanceCounter::~PerformanceCounter()
{
	
}

std::size_t PerformanceCounter::max_memory() const
{
	return max_memory_;
}

std::size_t PerformanceCounter::used_memory()
{
	return 0;
}

double PerformanceCounter::cpu_usage()
{
	return 0;
}

double PerformanceCounter::gpu_usage()
{
	return GPU_query_failed_ ? 0 : 0;
}
 
#else /* UNIX */

PerformanceCounter::PerformanceCounter()
:	max_memory_(0), num_processor_(0), GPU_query_failed_(false)
{
	
}

PerformanceCounter::~PerformanceCounter()
{
	
}

std::size_t PerformanceCounter::max_memory() const
{
	return max_memory_;
}

std::size_t PerformanceCounter::used_memory()
{
	return 0;
}

double PerformanceCounter::cpu_usage()
{
	return 0;
}

double PerformanceCounter::gpu_usage()
{
	return GPU_query_failed_ ? 0 : 0;
}
 
#endif
