/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Write the current state of the simulation to a binary file.
 *  The format will look as follows:
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
 
#include "QLB.hpp"

#define x(i,j) 	3*((i)*L_ + (j))
#define y(i,j)  3*((i)*L_ + (j)) + 1
#define z(i,j)  3*((i)*L_ + (j)) + 2

void QLB::dump_simulation(bool static_viewer)
{
	Timer t; t.start();
	
	std::string filename("dump" + std::to_string(L_) + ".bin");
	std::cout << "Writing to '" << filename << "' ... " << std::flush;
	
	if(!static_viewer)
	{
		const float dx = float(dx_);
		// Setup vertex arrays
		for(unsigned i = 0; i < L_; ++i)
			for(unsigned j = 0; j < L_; ++j)
			{
				array_vertex_[x(i,j)] = dx*(i-0.5f*(L_-1));
				array_vertex_[y(i,j)] = 0.0f;                 
				array_vertex_[z(i,j)] = dx*(j-0.5f*(L_-1));
			}
	
		// Calculate current vertices and normals
		calculate_vertex(0, 1);
		calculate_normal();
	}
	
	// Write to binary file
	fout.open(filename, std::ios::out | std::ios::binary);

	if(!fout.is_open() && !fout.good())
	{
		std::cout << "FAILED" << std::endl;
		FATAL_ERROR("Cannot open '"+filename+"'");
	}
	
	int int_args[4]     = { QLB_MAJOR, QLB_MINOR, static_cast<int>(L_), V_indx_ };
	float float_args[3] = { static_cast<float>(scaling_), static_cast<float>(dx_),
	                        static_cast<float>(mass_) };

	fout.write(reinterpret_cast<char*>(int_args),   sizeof(int_args));
	fout.write(reinterpret_cast<char*>(float_args), sizeof(float_args));
	fout.write(reinterpret_cast<char*>(array_vertex_.data()), 
	           array_vertex_.size() * sizeof(array_vertex_[0]));
	fout.write(reinterpret_cast<char*>(array_normal_.data()), 
	           array_normal_.size() * sizeof(array_normal_[0]));
	
	fout.close();
	std::cout << "Done" << std::endl;

	double t_write = t.stop();
	if(opt_.verbose())
	{
		std::size_t size = 3*sizeof(int) + (1 + 2*array_vertex_.size())*sizeof(float);
		std::printf("  Size : %5.2f MB\n", size * 1e-6 );
		std::printf("  Time : %5.2f s\n", t_write );
	}
}

#undef x
#undef y
#undef z
