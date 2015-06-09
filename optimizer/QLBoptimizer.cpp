/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  This file contains the program 'QLBoptimizer' which is used to self-tune the
 *  QLB program. QLBoptimizer will try to find the optimal CUDA settings, in the
 *  form of block and grid dimensions, for your device. 
 */

#include <array>
#include <cstdio>
#include <algorithm>

#include "QLB.hpp"
#include "utility.hpp"

#define NUM_DIMS   6
#define NUM_STEPS  9

static Progressbar* p;

/**
 *	Seek the optimal configuration by iterating over different configurations.
 *	@param   QLB_system   Initialized QLB object
 *	@param   cur_dim      Initial guess of the block dimension
 *	@param   other        Either Block1 or Block4 block dimension. This is needed
 *	                      as QLB uses two kinds of kernels (L x L) and (L x L x 4) 
 *	                      and we can only optimize one at a time.
 *	@param	func_block    Function pointer to the member function of QLB to set
 *                        the block dimensions.
 *	@param  func_grid     Function pointer to the member function of QLB to set
 *	                      the grid dimensions.
 */
template < class func_ptr >
void perform_optimization_step(QLB& QLB_system, dim3& cur_dim, const dim3 other,
                               func_ptr func_block, func_ptr func_grid)
{
	cudaDeviceProp deviceProp; 
	cuassert(cudaGetDeviceProperties(&deviceProp, 0));
	const unsigned L = QLB_system.L();
	const unsigned max_threads = deviceProp.maxThreadsPerBlock;
	const unsigned warp_size  = deviceProp.warpSize;
	
	// Bind the member functions
	auto set_block = std::bind(func_block, &QLB_system, std::placeholders::_1);
	auto set_grid  = std::bind(func_grid,  &QLB_system, std::placeholders::_1);

	auto prod = [](const dim3& d) -> unsigned { return d.x*d.y*d.z; };
	
	// get a valid configuration
	while( prod(cur_dim) > max_threads )
		cur_dim.x /= 2;
	
	// get the grid dimension
	auto grid = [=] (const dim3& d) -> dim3 {	
		return dim3( L % d.x == 0 ? L / d.x : L / d.x + 1,
		             L % d.y == 0 ? L / d.y : L / d.y + 1, 1);
	};
	
	// step functions 
	// Note: We might shuffle sometimes to prevent being trapped in certain 
	// configurations.
	auto left = [&](const dim3& d) -> dim3 { 
		unsigned cur = d.x*d.y*d.z/2;
		if(cur >= warp_size && d.y > 1)
			return dim3(d.x, d.y/2, d.z); 
		else if(d.y > 1) // shuffle
			return dim3(2*d.x, d.y/2, d.z);
		else
			return dim3(d.x, d.y, d.z);
	};
	
	auto right = [&](const dim3& d) -> dim3 { 
		unsigned cur =  d.x*d.y*d.z*2;
		if(cur <= max_threads)
			return dim3(d.x, 2*d.y, d.z); 
		else // shuffle
			return dim3(d.x/2, 2*d.y, d.z);
	};
	
	auto up = [&](const dim3& d) -> dim3 { 
		unsigned cur =  d.x*d.y*d.z*2;
		if(cur <= max_threads)
			return dim3(2*d.x, d.y, d.z); 
		else // shuffle
			return dim3(2*d.x, d.y/2, d.z);
	};
	
	auto down = [&](const dim3& d) -> dim3 { 
		unsigned cur =  d.x*d.y*d.z/2;
		if(cur >= warp_size && d.x > 1)
			return dim3(d.x/2, d.y, d.z);
		else if(d.x > 1) // shuffle
			return dim3(d.x/2, 2*d.y, d.z);
		else
			return dim3(d.x, d.y, d.z);
	};

	// Benchmark function
	const int n_runs = (1 << 14) / L;
	auto benchmark = [=](QLB& QLB_system) -> double {
		Timer t;
		t.start();
		for(int i = 0; i < n_runs; ++i)
			QLB_system.evolution_GPU();
		return t.stop();
	};

	std::array<double, 4> timings;
	double cur_time;
	
	if(cur_dim.z == 1)
	{	
		QLB_system.set_block4(other);
		QLB_system.set_grid4(grid(other));
	}
	else
	{
		QLB_system.set_block1(other);
		QLB_system.set_grid1(grid(other));
	}
	
	for(int i = 0; i < NUM_STEPS; ++i)
	{
		// current
		set_block(cur_dim);
		set_grid(grid(cur_dim));
		cur_time = benchmark(QLB_system);
		
		// up
		set_block(up(cur_dim));
		set_grid(grid(up(cur_dim)));
		timings[0] = benchmark(QLB_system);
		
		// right
		set_block(right(cur_dim));
		set_grid(grid(right(cur_dim)));
		timings[1] = benchmark(QLB_system);
		
		// down
		set_block(down(cur_dim));
		QLB_system.set_grid1(grid(down(cur_dim)));
		timings[2] = benchmark(QLB_system);
		
		// left
		set_block(left(cur_dim));
		set_grid(grid(left(cur_dim)));
		timings[3] = benchmark(QLB_system);
		
		// Select new block dimension
		auto min_idx = 
		     std::min_element(timings.begin(), timings.end()) - timings.begin();

		dim3 old_dim = cur_dim;
		if(cur_time > timings[min_idx])
		{
			switch(min_idx)
			{
				case 0:
					cur_dim = up(cur_dim);
					break;
				case 1:
					cur_dim = right(cur_dim);
					break;
				case 2:
					cur_dim = down(cur_dim);
					break;
				case 3:
					cur_dim = left(cur_dim);
					break;
			}
		}
		else
		{
			p->progress(NUM_STEPS-i);
			break;
		}
		
		// We stop if we do not make any progress
		if(cur_dim.x == old_dim.x && cur_dim.y == old_dim.y)
		{
			p->progress(NUM_STEPS-i);
			break;
		}
		
		p->progress();
	}
}

/**
 *	Optimze the block dimensions for a given system size 'L'
 *	@param  L          length of the grid size
 *	@param  dim_z      z-dimension of the block [dim_z <= 4] 
 *	@return optimal block dimensions
 */
dim3 optimize(unsigned L, unsigned dim_z)
{
	// Setup the system
	QLBparser parser("","");
	QLBopt opt(0, 0, 2, 1, "__QLBoptimizerTest__", 1.0); 
	
	QLB QLB_system(L, 1.5, 0.1, 1.5, 14, 100, 0, parser, opt);
	
	dim3 cur_dim = dim3(16, 16, dim_z);
	if(dim_z == 1)
		perform_optimization_step(QLB_system, cur_dim, dim3(1, 16, 4), 
		                          &QLB::set_block1, &QLB::set_grid1);
	else // dim_z == 4
		perform_optimization_step(QLB_system, cur_dim, dim3(1, 32, 1), 
		                          &QLB::set_block4, &QLB::set_grid4);
	
	return cur_dim;
}

/**
 *	Write the contents of the blocks1 and blocks4 arrays to a config file in the
 *	appropriate format.
 *	@param  blocks1   array of std::pair with the grid size and block dimensions 
 *	@param  blocks4   array of std::pair with the grid size and block dimensions 
 */
template < class T >
void write_to_config_file(const T& blocks1, const T& blocks4)
{
	std::cout << "Writing to config file 'QLBconfig.conf' ... ";
	std::ofstream fout("QLBconfig.conf");
	
	if(!fout.is_open() && !fout.good())
	{
		std::cout << "FAILED" << std::endl;	
		FATAL_ERROR("cannot open file: 'QLBconfig.conf'");
	}
	
	cudaDeviceProp deviceProp; 
	cuassert(cudaGetDeviceProperties(&deviceProp, 0));
	std::string device(deviceProp.name);

	fout << "QLBoptimizer - v1.0" << std::endl;
	fout << "Config file was generated for device : [" << device << "]\n" << std::endl;
	
	fout << "$BEGIN_CONFIG" << std::endl;
	fout << "$BEGIN_BLOCK1" << std::endl;
	fout << blocks1.size() << std::endl;
	
	for(const auto& b : blocks1)
		fout << std::right << std::setw(15) << b.first
		                   << std::setw(15) << b.second.x
		                   << std::setw(15) << b.second.y
		                   << std::setw(15) << b.second.z << std::endl;
	
	fout << "$END_BLOCK1" << std::endl;
	fout << "$BEGIN_BLOCK4" << std::endl;
	fout << blocks4.size() << std::endl;
	
	for(const auto& b : blocks4)
		fout << std::right << std::setw(15) << b.first
		                   << std::setw(15) << b.second.x
		                   << std::setw(15) << b.second.y
		                   << std::setw(15) << b.second.z << std::endl;
	
	fout << "$END_BLOCK4" << std::endl;
	fout << "$END_CONFIG" << std::endl;
	
	std::cout << "Done" << std::endl;	
	fout.close();  	
}

int main(int argc, char* argv[])
{
	int size[NUM_DIMS] = {32, 64, 128, 256, 512, 1024};

	std::array< std::pair<int,dim3>, NUM_DIMS > blocks1;
	std::array< std::pair<int,dim3>, NUM_DIMS > blocks4;

	p = new Progressbar(2 * NUM_DIMS * NUM_STEPS);

	cudaDeviceProp deviceProp; 
	cuassert(cudaGetDeviceProperties(&deviceProp, 0));
	std::printf("QLBoptimizer - v1.0\n");
	std::printf("Config file will be generated for device : [%s]\n",deviceProp.name);
	std::printf("\nNOTE: To obtain accurate results, close all other" 
	            " applications which might\nuse the GPU!\n");
	std::printf("\n-------------------------------------------------------\n");
	std::printf("| %-15s | %-15s | %-15s |\n", "Grid Size (L)", "Block1", 
	            "Block4");
	std::printf("-------------------------------------------------------\n");
	            
	for(int i = 0; i < NUM_DIMS; ++i)
	{
		blocks1[i].first = size[i];
		blocks4[i].first = size[i];
		
		// Run benchmark
		blocks1[i].second = optimize(size[i], 1);
		blocks4[i].second = optimize(size[i], 4);
		
		p->pause();
		std::printf("| %-15i |  (%3i,%3i,%3i)  |  (%3i,%3i,%3i)  |\n", 
		            size[i], 
		            blocks1[i].second.x, blocks1[i].second.y, blocks1[i].second.z,
		            blocks4[i].second.x, blocks4[i].second.y, blocks4[i].second.z);
	}

	p->pause();	
	delete p;
	std::printf("-------------------------------------------------------\n");
	
	write_to_config_file(blocks1, blocks4);
	return 0;
}
