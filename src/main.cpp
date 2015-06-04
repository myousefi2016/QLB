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

	if(cmd.gui())
		QLB_run_glut(argc, argv);
	else
		QLB_run_no_gui(cmd);
	
#if defined(QLB_HAS_CUDA) && ( !defined (__APPLE__) && !defined(MACOSX) )
	cudaDeviceReset();
#endif
	return 0;
}

void QLB_run_no_gui(const CmdArgParser& cmd)
{
	// We set some default value if nothing was passed
	unsigned L = cmd.L() ? cmd.L_value() : 128;
	QLB::float_t dx     = cmd.dx()     ? cmd.dx_value() : 1.5625;
	QLB::float_t mass   = cmd.mass()   ? cmd.mass_value() : 0.1;
	QLB::float_t dt     = cmd.dt()     ? cmd.dt_value() : 1.5625;
	QLB::float_t delta0 = cmd.delta0() ? cmd.delta0_value() : 14.0;
	unsigned tmax = cmd.tmax() ? cmd.tmax_value() : 100;
	
	// Setup threadpool 
	std::vector<std::thread> threadpool(cmd.nthreads_value());

	// Setup QLB options
	QLBopt opt;
	opt.set_plot(cmd.plot()); 
	opt.set_verbose(cmd.verbose());
	opt.set_device(cmd.device());
	opt.set_nthreads(cmd.nthreads_value());
	opt.set_config_file(cmd.config_file());
	
	// Setup QLBparser
	QLBparser parser(cmd.potential_file(), cmd.initial_file());
	parser.parse_input(&cmd);
	
	// Adjust arguments
	if(parser.is_valid())
	{
		L      = parser.L();
		dx     = parser.dx(); 
		mass   = parser.mass_is_present() ? parser.mass() : mass;
		delta0 = parser.delta0_is_present() ? parser.delta0() : delta0;
	}
	
	// Setup the system
	QLB QLB_system(L, dx, mass, dt, delta0, tmax, cmd.potential(), parser, opt);

	// Setup utility classes
	Progressbar p(tmax);
	
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
				
				if(UNLIKELY(cmd.dump() && cmd.dump_value() == t+1))
				{
					p.pause();
					QLB_system.dump_simulation(false);
				}
				
				if(cmd.progressbar())
					p.progress();
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
			
				if(cmd.progressbar())
					p.progress();
					
				if(UNLIKELY(cmd.dump() && cmd.dump_value() == t+1))
				{
					p.pause();
					QLB_system.dump_simulation(false);
				}

			}
			break;
		}
		case 2: // GPU CUDA
			for(unsigned t = 0; t < tmax; ++t)
			{
				QLB_system.evolution_GPU();

				if(UNLIKELY(cmd.dump() && cmd.dump_value() == t+1))
				{
					p.pause();
					QLB_system.get_device_arrays();
					QLB_system.dump_simulation(false);
				}
			
				if(cmd.progressbar())
					p.progress();
			}
	}
	double tsim = t.stop();
	p.pause();
	
	if(cmd.device() == 2)
		QLB_system.get_device_arrays();
		
	// Write values to file (if requested)
	QLB_system.print_spread();
	QLB_system.write_content_to_file();	
	
	std::cout << "Simulation time : " << tsim << " s" << std::endl;
}


