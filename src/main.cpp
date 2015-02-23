/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Setup the simulation and call the appropriate functions for launching
 *	- QLB_run_no_gui (to run without a GUI) pass 'gui=none' (default)
 *	- QLB_run_glut   (to run with GLUT) pass '--gui=glut'
 */
 
// System includes 
#include <iostream>
#include <cstdio>
#include <cstdlib>

// Local includes
#include "QLB.hpp"
#include "error.hpp"
#include "utility.hpp"
#include "CmdArgParser.hpp"
#include "GLUTmain.hpp"

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
	
	QLB  QLB_system(L, dx, mass, dt, cmd.V(), cmd.plot(), cmd.verbose());
	
	Timer t;
	
	t.start();
	
	// Run simulation
	while(tmax--)
		QLB_system.evolution();

	double tsim = t.stop();
	std::cout << "Simulation time : " << tsim << std::endl;
	
	// Write values to file
	QLB_system.write_content_to_file();	
}


