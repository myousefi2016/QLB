/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Visualize the simulation using GLUT for window management. 
 */

#ifndef GLUT_MAIN_HPP
#define GLUT_MAIN_HPP

// System includes 
#include <iostream>
#include <cstdio>
#include <cstdlib>

// Local includes
#include "QLB.hpp"
#include "error.hpp"
#include "utility.hpp"
#include "CmdArgParser.hpp"
#include "GLUTui.hpp"

void QLB_run_glut(int argc, char* argv[]);
void init_GL(int argc, char* argv[]);
void cleanup();

// GLUT callback functions
void callback_display();
void callback_reshape(int width, int height);
void callback_keyboard(unsigned char key, int x, int y);
void callback_keyboard_2(int key, int x, int y);
void callback_mouse(int button, int state, int x, int y);
void callback_mouse_motion(int x, int y);

#endif /* GLUTmain.hpp */
