/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Setup the simulation and call the appropriate functions for launching
 *  - QLB_run_no_gui (to run without a GUI) pass 'gui=none' (default)
 *  - QLB_run_glut   (to run with GLUT) pass '--gui=glut'
 *
 *  For further assistance try '--help'
 */
 
// System includes 
#include <iostream>
#include <cstdio>
#include <cstdlib>

// Local includes
#include "QLB.hpp"
#include "error.hpp"
#include "utility.hpp"
#include "PerformanceCounter.hpp"
#include "CmdArgParser.hpp"
#include "GLUTmain.hpp"
#include "barrier.hpp"

// local functions
static void QLB_run_no_gui(const CmdArgParser& cmd);

int main(int argc, char* argv[])
{
	// Parse command-line arguments
	CmdArgParser cmd(argc, argv);

	switch(cmd.gui())
	{
		case 0: // none
			QLB_run_no_gui(cmd);
			break;
		case 1:  // glut
			QLB_run_glut(argc, argv);
			break;
	}	

	return 0;
}

void QLB_run_no_gui(const CmdArgParser& cmd)
{
	// We set some default value if nothing was passed
	const unsigned L = cmd.L() ? cmd.L_value() : 128;
	const QLB::float_t dx = cmd.dx() ? cmd.dx_value() : 1.5625;
	const QLB::float_t mass = cmd.mass() ? cmd.mass_value() : 0.1;
	const QLB::float_t dt = cmd.dt() ? cmd.dt_value() : 1.5625;
	unsigned tmax = cmd.tmax() ? cmd.tmax_value() : 100;
	
	// Setup threadpool 
	std::vector<std::thread> threadpool(cmd.nthreads_value());

	// Set QLB options
	QLBopt opt;
	opt.set_plot(cmd.plot()); 
	opt.set_verbose(cmd.verbose());
	opt.set_device(cmd.device());
	opt.set_nthreads(cmd.nthreads_value());

	// Setup the system
	QLB QLB_system(L, dx, mass, dt, cmd.V(), opt);
	
	Timer t;
	t.start();
	
	// Run simulation in [0, dt*tmax]
	switch(opt.device())
	{
		case 0: // CPU serial
		{
			for(unsigned t = 0; t < tmax; ++t)
			{
				QLB_system.evolution_CPU_serial();
				if(UNLIKELY(cmd.dump() && cmd.dump_value() == t))
					QLB_system.dump_simulation(false);
			}
			break;
		}
		case 1: // CPU multi threaded
		{
			for(unsigned t = 0; t < tmax; ++t)
			{
				for(std::size_t tid = 0; tid < threadpool.size(); ++tid)
					threadpool[tid] = std::thread( &QLB::evolution_CPU_thread, 
					                               &QLB_system, 
					                               int(tid) ); 
				for(std::thread& t : threadpool)
					t.join();
				
				if(UNLIKELY(cmd.dump() && cmd.dump_value() == t))
					QLB_system.dump_simulation(false);
			}
			break;
		}
		case 2: // GPU CUDA
			FATAL_ERROR("CUDA version is not yet implemented");
			break;
	}
	double tsim = t.stop();
	
	// Write values to file
	QLB_system.print_spread();
	QLB_system.write_content_to_file();	
	
	std::cout << "Simulation time : " << tsim << " s" << std::endl;
}


