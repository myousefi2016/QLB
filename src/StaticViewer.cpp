/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Load and setup visualization of a dump file produced by 'QLB::dump_simulation()' 
 *  and passed by the option '--dump-load=file'.
 *  The file should have the following format:
 *  -  QLB_MAJOR       ( int )
 *  -  QLB_MINOR       ( int )
 *  -  L_              ( int )
 *  -  V_indx_         ( int )
 *  -  scaling_        ( float )
 *	-  dx_             ( float )
 *  -  mass_           ( float )
 *  -  array_vertex_   ( float[3*L_*L_] )
 *  -  array_normal_   ( float[3*L_*L_] )
 */

#include "StaticViewer.hpp" 

static std::size_t get_file_size(std::string filename)
{
    std::ifstream in(filename, std::ios::ate | std::ios::binary);
    return in.tellg(); 
}

QLB* StaticViewerLoader(CmdArgParser* cmd)
{
	std::string filename(cmd->static_viewer_file());
	std::cout << "Loading '" << filename << "' ... " << std::flush;

	// Setup filestreams
	std::ifstream fin(filename, std::ios::out | std::ios::binary);
	
	if(!fin.is_open() && !fin.good())
	{
		std::cout << "FAILED" << std::endl;
		FATAL_ERROR("Cannot open '"+filename+"'");
	}

	std::vector<int>   int_args(4);
	std::vector<float> float_args(3);

	if(get_file_size(filename) < (sizeof(int_args[0]) * int_args.size()))
	{
		std::cout << "FAILED" << std::endl;
		FATAL_ERROR("'"+filename+"' is not a valid input file");
	}

	// Read the file
	fin.read(reinterpret_cast<char*>(int_args.data()),   
	         int_args.size() * sizeof(int_args[0]));

	// Perform some checks if the file is valid         
	if(int_args[0] != QLB_MAJOR && int_args[1] != QLB_MINOR)
	{
		std::cout << "FAILED" << std::endl;
		FATAL_ERROR("'"+filename+"' is not a valid input file");
	}        
	
	fin.read(reinterpret_cast<char*>(float_args.data()), 
	         float_args.size() * sizeof(float_args[0]));
	
	unsigned L = int_args[2];
	std::vector<float> array_vertex(3*L*L);
	std::vector<float> array_normal(3*L*L);
	
	fin.read(reinterpret_cast<char*>(array_vertex.data()), 
	         array_vertex.size() * sizeof(array_vertex[0]));
	fin.read(reinterpret_cast<char*>(array_normal.data()), 
	         array_normal.size() * sizeof(array_normal[0])); 

	fin.close();
	std::cout << "Done" << std::endl;
	
	QLBopt opt;
	opt.set_plot(0); 
	opt.set_verbose(cmd->verbose());
	opt.set_device(1);
	if(cmd->device() == 2) WARNING("ignoring option '--device=gpu'");
	opt.set_nthreads(cmd->max_threads());
	
	// Setup QLB
	QLB* QLB_system = new QLB(L, int_args[3], float_args[1], float_args[2], 
	                          float_args[0], array_vertex, array_normal, opt);
	
	return QLB_system;
}


