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

QLB* StaticViewerLoader(CmdArgParser* cmd)
{
	std::string filename(cmd->static_viewer_file());
	std::cout << "Loading '" << filename << "' ... " << std::flush;

	std::ifstream fin(filename, std::ios::out | std::ios::binary);
	
	if(!fin.is_open() && !fin.good())
	{
		std::cout << "FAILED" << std::endl;
		FATAL_ERROR("Cannot open '"+filename+"'");
	}

	std::vector<int>   int_args(4);
	std::vector<float> float_args(3);

	// Read-in the file
	fin.read(reinterpret_cast<char*>(int_args.data()),   
	         int_args.size() * sizeof(int_args[0]));
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
	
	QLBopt opt;
	opt.set_plot(0); 
	opt.set_verbose(cmd->verbose());
	opt.set_device(cmd->device());
	opt.set_nthreads(cmd->nthreads_value());
	
	// Setup QLB
	QLB* QLB_system = new QLB(L, int_args[3], float_args[1], float_args[2], 
	                          float_args[0], array_vertex, array_normal, opt);
	
	std::cout << "Done" << std::endl;
	
	return QLB_system;
}


