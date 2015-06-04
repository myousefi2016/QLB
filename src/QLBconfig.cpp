/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  This file contains the parser for the config files used to set the block sizes
 *  and grid sizes of the CUDA kernels.  
 */

#include "QLB.hpp"

#ifdef QLB_HAS_CUDA 

void QLB::set_block_and_grid_size(std::string filename) 
{
	std::ifstream fin;
	bool use_default_args = false;
	
	// The optimzer skips this and sets the blocks & grids himself
	if(filename == "__QLBoptimizerTest__")
		return;

	if(filename.empty())
		use_default_args = true;
	else
	{
		fin.open(filename);
	
		if(!fin.is_open() && !fin.good())
		{
			use_default_args = true;
			WARNING("Cannot open config file '"+filename+"'."+
			        "\nConsider running 'QLBoptimizer' to generate a config file"+
			        ", or pass an existing one with the option '--config=FILE'.");
		}
	}
	
	// Parse config file
	if(!use_default_args)
	{
		std::array<std::string, 6> keywords;
		keywords[0] = "$BEGIN_CONFIG";
		keywords[1] = "$BEGIN_BLOCK1";
		keywords[2] = "$END_BLOCK1";
		keywords[3] = "$BEGIN_BLOCK4";
		keywords[4] = "$END_BLOCK4";
		keywords[5] = "$END_CONFIG";
	
		std::vector< std::pair<unsigned,dim3> > block1;
		std::vector< std::pair<unsigned,dim3> > block4;
		
		bool begin_found = false;
		bool end_found   = false;
		
		bool block1_begin_found = false;
		bool block1_end_found   = false;
		bool block4_begin_found = false;
		bool block4_end_found   = false;
		
		std::string line;
		std::size_t pos;
		
		unsigned x,y,z;
		unsigned L;
		
		while(!std::getline(fin, line).eof())
		{
			if(!begin_found)
			{
				begin_found = line.find(keywords[0]) != std::string::npos; 
			}
			
			if(!block1_begin_found)
			{
				pos = line.find(keywords[1]);
				block1_begin_found = pos != std::string::npos;
				
				if(block1_begin_found)
				{
					unsigned num_blocks;
					fin >> num_blocks;
					block1.resize(num_blocks);
				
					for(unsigned i = 0; i < num_blocks; ++i)
					{	
						fin >> L; fin >> x; fin >> y; fin >> z;
						block1[i] = std::make_pair(L, dim3(x,y,z));
					}
				}
			}
			
			if(!block1_end_found)
			{
				pos = line.find(keywords[2]);
				block1_end_found = pos != std::string::npos;
			}
			
			if(!block4_begin_found)
			{
				pos = line.find(keywords[3]);
				block4_begin_found = pos != std::string::npos;
				
				if(block4_begin_found)
				{
					unsigned num_blocks;
					fin >> num_blocks;
					block4.resize(num_blocks);
				
					for(unsigned i = 0; i < num_blocks; ++i)
					{	
						fin >> L; fin >> x; fin >> y; fin >> z;
						block4[i] = std::make_pair(L, dim3(x,y,z));
					}
				}
			}
			
			if(!block4_end_found)
			{
				pos = line.find(keywords[4]);
				block4_end_found = pos != std::string::npos;
			}
			
			if(!end_found)
			{
				end_found = line.find(keywords[5]) != std::string::npos; 
			}
		}
		
		// If any errors occured, we fall back to default values
		if(!begin_found || !end_found || !block1_begin_found ||
		   !block1_end_found || !block4_begin_found || !block4_end_found)
		{
			use_default_args = true;
			WARNING("Config file '"+filename+"' is invalid.");	
		}
		else
		{
			// Use the best matching data
			if(L_ <= block1[0].first)
			{
				block1_ = block1[0].second;
				block4_ = block4[0].second;
			}
			else
			{
				for(auto it = block1.begin(), end = block1.end(); it != end; 
				    ++it)
				{
					if(std::next(it) != end)
					{		
						if(L_ >= it->first && L_ <= std::next(it)->first)
						{
							unsigned diff_next = std::next(it)->first - L_;
							unsigned diff_curr = L_ - it->first;
				
							if(diff_next >= diff_curr)
							{
								block1_ = it->second;
								block4_ = block4[it - block1.begin()].second;
								break;
							}
							else
							{
								block1_ = std::next(it)->second;
								block4_ = block4[std::next(it) - 
									             block1.begin()].second;
								break;
							}
						}		
					}
					else
					{
						block1_ = it->second;
						block4_ = block4[block4.size() - 1].second;
					} 
				}
			}
		}
	}
	
	if(use_default_args)
	{
		// Default values (obtained via 'QLBoptimizer' for GeForce GTX 770)
		cudaDeviceProp deviceProp; 
		cuassert(cudaGetDeviceProperties(&deviceProp, 0));
		
		block4_.x = 1;
		block4_.y = deviceProp.warpSize / 2;
		block4_.z = 4;
	

		block1_.x = 1;
		block1_.y = deviceProp.warpSize;
		block1_.z = 1;
	}
	
	// Grid dimensions are fixed by L
	grid1_  = dim3( L_ % block1_.x == 0 ? L_ / block1_.x : L_ / block1_.x + 1,
	                L_ % block1_.y == 0 ? L_ / block1_.y : L_ / block1_.y + 1, 1);
	
	grid4_  = dim3( L_ % block4_.x == 0 ? L_ / block4_.x : L_ / block4_.x + 1,
	                L_ % block4_.y == 0 ? L_ / block4_.y : L_ / block4_.y + 1, 1);
	                
	// cleanup
	if(fin.is_open()) fin.close();
}

#else

void QLB::set_block_and_grid_size(std::string filename)
{
	FATAL_ERROR("QLB was compiled without CUDA support");
}

#endif
