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

#ifndef STATIC_VIEWER_HPP
#define STATIC_VIEWER_HPP

// System Includes
#include <string>
#include <fstream>

// Local Includes
#include "QLB.hpp"
#include "QLBopt.hpp"
#include "CmdArgParser.hpp"
#include "error.hpp"

/**
 *	Load and setup the QLB system from the dump file
 *	@param  cmd      pointer to CmdArgParser
 *	@return pointer to the constructed QLB system
 */
QLB* StaticViewerLoader(CmdArgParser* cmd);

#endif
