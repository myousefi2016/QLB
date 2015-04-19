/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Query for different performance counters like CPU/GPU usage, memory usage 
 *  etc. The implementation is highly OS dependent although the interface
 *  is not.
 *
 *  Credit for GPU usage counter on Windows:
 *  Open Hardware Monitor (http://code.google.com/p/open-hardware-monitor) 
 */

#include "PerformanceCounter.hpp"
 
#ifdef _WIN32

//=============================== WINDOWS ====================================== 

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
	ULARGE_INTEGER  ul_sys_idle_old, ul_sys_kernel_old, ul_sys_user_old;
};

static CPUstats* CPU;
 
PerformanceCounter::PerformanceCounter()
:	cpu_max_memory_(0), gpu_max_memory_(0), GPU_query_failed_(false)
{
	// maximum memory
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	if( GlobalMemoryStatusEx(&memInfo) == 0)
		WARNING("GlobalMemoryStatusEx failed")
	else
		cpu_max_memory_ =  std::size_t(memInfo.ullTotalPhys);
	
	// === CPU ===
	CPU = new CPUstats;

	FILETIME ft_sys_idle;
	FILETIME ft_sys_kernel;
	FILETIME ft_sys_user;

	ULARGE_INTEGER  ul_sys_idle;
	ULARGE_INTEGER  ul_sys_kernel;
	ULARGE_INTEGER  ul_sys_user;

	GetSystemTimes(&ft_sys_idle, &ft_sys_kernel, &ft_sys_user);

	CopyMemory(&ul_sys_idle  , &ft_sys_idle  , sizeof(FILETIME));
	CopyMemory(&ul_sys_kernel, &ft_sys_kernel, sizeof(FILETIME));
	CopyMemory(&ul_sys_user  , &ft_sys_user  , sizeof(FILETIME));

	CPU->ul_sys_idle_old.QuadPart   = ul_sys_idle.QuadPart;
    CPU->ul_sys_user_old.QuadPart   = ul_sys_user.QuadPart;
    CPU->ul_sys_kernel_old.QuadPart = ul_sys_kernel.QuadPart;
	
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
		WARNING(": PerformanceCounter : Cannot find 'nvapi64.dll'");
#else
		WARNING(": PerformanceCounter : Cannot find 'nvapi.dll'");
#endif
		GPU_query_failed_ = true;
	}
}

PerformanceCounter::~PerformanceCounter()
{
	delete CPU;
}

std::size_t PerformanceCounter::cpu_max_memory() const
{
	return cpu_max_memory_;
}

std::size_t PerformanceCounter::gpu_max_memory() const
{
	return gpu_max_memory_;
}

std::size_t PerformanceCounter::cpu_memory()
{
	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, 
	                     sizeof(pmc));
	return std::size_t(pmc.WorkingSetSize);
}

std::size_t PerformanceCounter::gpu_memory()
{
#ifdef QLB_HAS_CUDA
	std::size_t free;
	std::size_t total;
	cudaMemGetInfo( &free, &total );
	return (total - free);
#else
	return 0;
#endif
}

double PerformanceCounter::cpu_usage()
{
	// We query the filetime and calculate the usage
    FILETIME ft_sys_idle;
    FILETIME ft_sys_kernel;
    FILETIME ft_sys_user;

    ULARGE_INTEGER ul_sys_idle;
    ULARGE_INTEGER ul_sys_kernel;
    ULARGE_INTEGER ul_sys_user;

    GetSystemTimes(&ft_sys_idle, &ft_sys_kernel, &ft_sys_user);
	ULONGLONG usage = 0;

    CopyMemory(&ul_sys_idle  , &ft_sys_idle  , sizeof(FILETIME));
    CopyMemory(&ul_sys_kernel, &ft_sys_kernel, sizeof(FILETIME));
    CopyMemory(&ul_sys_user  , &ft_sys_user  , sizeof(FILETIME));

    usage  = ( ( ( (ul_sys_kernel.QuadPart - CPU->ul_sys_kernel_old.QuadPart) + 
		           (ul_sys_user.QuadPart   - CPU->ul_sys_user_old.QuadPart) )
			     - (ul_sys_idle.QuadPart   - CPU->ul_sys_idle_old.QuadPart) ) * (100) ) /
             ( (ul_sys_kernel.QuadPart - CPU->ul_sys_kernel_old.QuadPart) + 
			   (ul_sys_user.QuadPart   - CPU->ul_sys_user_old.QuadPart) );

    CPU->ul_sys_idle_old.QuadPart   = ul_sys_idle.QuadPart;
    CPU->ul_sys_user_old.QuadPart   = ul_sys_user.QuadPart;
    CPU->ul_sys_kernel_old.QuadPart = ul_sys_kernel.QuadPart;
	
	return double(usage);
}

double PerformanceCounter::gpu_usage()
{	
	if(GPU_query_failed_)
	{
		return 0;
	}
	else 
	{
		(*NvAPI_GPU_GetUsages)(gpuHandles[0], gpuUsages);
		return gpuUsages[3];
	}
}
 
//=============================== LINUX ========================================

#elif defined(__linux__)

/**
 *  This class will be used to query for the GPU Utilization using the 
 *  'nvidia-settings' which are distributed with he Nvidia binary driver for Linux.
 *
 *  The program is highly dependant on the output format of 'nvidia-settings'
 *  which should look more or less the following to work porperly:
 *  invoke 'nvidia-settings -q GPUUtilization'
 *  >
 *  >  Attribute 'GPUUtilization' (usaername-PC:0[gpu:0]): graphics=28, memory=25,
 *  >  video=0, PCIe=2
 *  >
 */
class GPUUtilization
{
public:

	/** 
	 * Constructor - Check for usability
	 */
	GPUUtilization()
		: gpu_query_failed_(false), fp_(nullptr), gpu_util_(0)
	{
		fp_ = popen("nvidia-settings -q GPUUtilization 2>&1", "r");
		
		if(fp_ == nullptr)
			gpu_query_failed_ = true;
			
		pclose(fp_);	
	}
	
	/**
	 * Query the GPU through 'nvidia-settings' to obtained current utilization.
	 * This call might have a non-trivial cost and should not be called extensivly
	 * @return current utilization in percent [0-99]
	 */
	int query_utilization()
	{
		fp_ = nullptr;
		if(!gpu_query_failed_ &&  
		   (fp_ = popen("nvidia-settings -q GPUUtilization 2>&1", "r")) != nullptr)
		{
			char path[1024];
			char* pp;
			gpu_util_ = -1;

			while(std::fgets(path, sizeof(path) - 1, fp_) != nullptr)
			{
				pp = std::strstr(path,"graphics=");
				if(pp != nullptr)
				{
					try {
						gpu_util_ = std::stoi(std::string(pp+9,pp+11));
					} catch(...) {}
				}
			}
			
			// query failed
			if(gpu_util_ == -1)
			{
				gpu_util_ = 0;
				gpu_query_failed_ = true;
			}
		}
		else
			gpu_query_failed_ = true;
			
		if(fp_ != nullptr)
			pclose(fp_);
		
		return gpu_util_;
	}
	
	bool gpu_query_failed() const { return gpu_query_failed_; }
	
private:
	bool gpu_query_failed_;
	std::FILE *fp_;
	int gpu_util_;
};

/**
 * Line parsing used by cpu_usage() and cpu_max_memory()
 */
static int parse_line(char* line)
{
    int i = std::strlen(line);
    while(*line < '0' || *line > '9') 
    	line++;
    line[i-3] = '\0';
    i = std::atoi(line);
    return i;
}
    
struct CPUstats
{
	unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;
};

static CPUstats*  CPU                = nullptr;
#if defined(QLB_HAS_CUDA) && defined(QLB_HAS_GPU_COUNTER) 
static GPUUtilization* GPU           = nullptr; 
static std::thread* GPU_query_thread = nullptr;
static int gpu_util                  = 0;
#endif

PerformanceCounter::PerformanceCounter()
:	cpu_max_memory_(0), gpu_max_memory_(0), GPU_query_failed_(false)
{

	// === CPU ===
	
	CPU = new CPUstats;
	try
	{
		std::FILE* file = fopen("/proc/stat", "r");
		
		if(!file) throw 1;
		
		if(!std::fscanf(file, "cpu %Ld %Ld %Ld %Ld", &CPU->lastTotalUser, 
		            &CPU->lastTotalUserLow, &CPU->lastTotalSys, &CPU->lastTotalIdle))
			throw 1;
		fclose(file);
	}
	catch(...)
	{}

 	// === GPU ===
#if defined(QLB_HAS_CUDA) && defined(QLB_HAS_GPU_COUNTER) 
 	GPU = new GPUUtilization;
 	
	if( (GPU_query_failed_ = GPU->gpu_query_failed()) )
 		WARNING(": PerformanceCounter : Cannot exectute 'nvidia-settings'")
 	
 	// we launch a thread and continue otherwise the query is too expensive
	GPU_query_thread = new std::thread([&](){ gpu_util = GPU->query_utilization(); });
#else
	GPU_query_failed_ = false;
#endif
}

PerformanceCounter::~PerformanceCounter()
{
	delete CPU;
#if defined(QLB_HAS_CUDA) && defined(QLB_HAS_GPU_COUNTER) 
	GPU_query_thread->join();
	delete GPU;
	delete GPU_query_thread;
#endif
}

std::size_t PerformanceCounter::cpu_max_memory() const
{
	return cpu_max_memory_;
}

std::size_t PerformanceCounter::gpu_max_memory() const
{
	return gpu_max_memory_;
}

std::size_t PerformanceCounter::cpu_memory()
{
	std::size_t result = 0;
	
	try
	{
		std::FILE* file = std::fopen("/proc/self/status", "r");
		char line[128];
		
		if(!file) throw 1;

		while (std::fgets(line, 128, file) != NULL)
		{
		    if (std::strncmp(line, "VmRSS:", 6) == 0){
		        result = parse_line(line);
		        break;
		    }
		}
		
		fclose(file);
	}
	catch(...)
	{}
	
	return result*1000;
}

std::size_t PerformanceCounter::gpu_memory()
{
#ifdef QLB_HAS_CUDA
	std::size_t free;
	std::size_t total;
	cudaMemGetInfo( &free, &total );
	return (total - free);
#else
	return 0;
#endif
}

double PerformanceCounter::cpu_usage()
{
    double usage = 0.0;
	try
	{
		std::FILE* file;
		unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

		file = std::fopen("/proc/stat", "r");
		if(!file) throw 1;
		
		if(!std::fscanf(file, "cpu %Ld %Ld %Ld %Ld", &totalUser, &totalUserLow,
		                &totalSys, &totalIdle))
		   throw 1;
		std::fclose(file);

		if (totalUser < CPU->lastTotalUser || totalUserLow < CPU->lastTotalUserLow ||
		    totalSys < CPU->lastTotalSys || totalIdle < CPU->lastTotalIdle)
		{
		    usage = -1.0;
		}
		else
		{
		    total = (totalUser    - CPU->lastTotalUser)    + 
		            (totalUserLow - CPU->lastTotalUserLow) +
		            (totalSys     - CPU->lastTotalSys);
		    usage  = total;
		    total += (totalIdle - CPU->lastTotalIdle);
		    usage /= total;
		    usage *= 100;
		}
		CPU->lastTotalUser = totalUser;
		CPU->lastTotalUserLow = totalUserLow;
		CPU->lastTotalSys = totalSys;
		CPU->lastTotalIdle = totalIdle;
	}
	catch(...)
	{}
	
	return usage;
}

double PerformanceCounter::gpu_usage()
{
#if defined(QLB_HAS_CUDA) && defined(QLB_HAS_GPU_COUNTER) 
	GPU_query_thread->join();
	delete GPU_query_thread;
	GPU_query_thread = new std::thread([&](){ gpu_util = GPU->query_utilization(); });
	return gpu_util;
#else
	return 0;
#endif
}
 

//=================================== UNIX ===================================== 
 
#else

PerformanceCounter::PerformanceCounter()
:	cpu_max_memory_(0), gpu_max_memory_(0), GPU_query_failed_(false)
{
	
}

PerformanceCounter::~PerformanceCounter()
{
	
}

std::size_t PerformanceCounter::cpu_max_memory() const
{
	return cpu_max_memory_;
}

std::size_t PerformanceCounter::gpu_max_memory() const
{
	return gpu_max_memory_;
}

std::size_t PerformanceCounter::cpu_memory()
{
	return 0;
}

std::size_t PerformanceCounter::gpu_memory()
{
#ifdef QLB_HAS_CUDA
	std::size_t free;
	std::size_t total;
	cudaMemGetInfo( &free, &total );
	return (total - free);
#else
	return 0;
#endif
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
