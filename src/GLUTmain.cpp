/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Th�ring, ETH Zurich
 *
 *  Visualize the simulation using GLUT for window management. 
 */

#include "GLUTmain.hpp"

// Local class pointers (needed to access classes during glut callback functions)
QLB* QLB_system		= nullptr;
UserInterface* UI	= nullptr;
CmdArgParser* cmd	= nullptr;

std::vector<std::thread> threadpool;

/// GLUT main function
void QLB_run_glut(int argc, char* argv[])
{
	// Reparse command-line arguments
	cmd = new CmdArgParser(argc, argv);
	
	// Setup threadpool
	threadpool.resize(cmd->nthreads_value());

	const unsigned L = cmd->L() ? cmd->L_value() : 128;
	const QLB::float_t dx   = cmd->dx()   ? cmd->dx_value() : 1.5625;
	const QLB::float_t mass = cmd->mass() ? cmd->mass_value() : 0.1;
	const QLB::float_t dt   = cmd->dt()   ? cmd->dt_value() : 1.5625;
	
	QLBopt opt;
	opt.set_plot(cmd->plot()); 
	opt.set_verbose(cmd->verbose());
	opt.set_device(cmd->device());
	opt.set_nthreads(cmd->nthreads_value());
	
	// Setup QLB 
	QLB_system = new QLB(L, dx, mass, dt, cmd->V(), opt);

	// Setup UserInterface engine
	int width = 800, height = 800;
	UI = new UserInterface(width, height, "QLB - v1.0", 
	                       float(-1.5*QLB_system->L()*QLB_system->dx()));

	// Setup OpenGL & GLUT	
	init_GL(argc, argv);
	QLB_system->init_GL();

	glutMainLoop();
	cleanup_and_exit();
}

/****************************
 *         init_GL          *
 ****************************/
void init_GL(int argc, char* argv[])
{
	// Use the main display
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

	// Initialize glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	// Scale the window properly (go fullscreen if requested)
	int screen_query_successful = 1; 
	if(cmd->fullscreen())
	{
		int width  = glutGet(GLUT_SCREEN_WIDTH);
		int height = glutGet(GLUT_SCREEN_HEIGHT);
		screen_query_successful = width != 0 && height != 0;
		UI->set_width(screen_query_successful ? width : UI->width() );
		UI->set_height(screen_query_successful ? height : UI->height() );
		
		// GameMode doesn't really work with multiple displays on Linux
#if defined(__linux__) && defined(QLB_MULTI_DISPLAY)
		screen_query_successful = 2;
#endif
	}
		
	if(cmd->fullscreen() && screen_query_successful == 1)
	{
		std::stringstream res;
		res << UI->width() << "x" << UI->height() << ":32@60"; 
		glutGameModeString(res.str().c_str());
		glutEnterGameMode();
	}
	else
	{
		glutInitWindowSize(UI->width(),UI->height());
		if(screen_query_successful != 1)
			WARNING("Cannot change to fullscreen mode, falling back to windowed mode.");
		if(glutCreateWindow(UI->title()) < 1)
			FATAL_ERROR("glutCreateWindow() failed, something is seriously wrong.");
	}

	// Initialize necessary OpenGL extensions
	if(glewInit() != GLEW_OK) 
		FATAL_ERROR("glewInit() failed, something is seriously wrong.")
	
	if(!glewIsSupported("GL_VERSION_2_0"))
		FATAL_ERROR("Support for necessary OpenGL 2.0 extensions is missing.");

	if(cmd->verbose())
	{
		std::cout << " === OpenGL Info === " << std::endl;
		std::printf("Version : %s\n", glGetString(GL_VERSION));
		std::printf("Device  : %s\n", glGetString(GL_RENDERER));
	}

	// GLUT callback registration
	glutDisplayFunc(callback_display);
	glutReshapeFunc(callback_reshape);
	glutMouseFunc(callback_mouse);
	glutMotionFunc(callback_mouse_motion);
	glutKeyboardFunc(callback_keyboard);
	glutSpecialFunc(callback_keyboard_2);

	// Viewport
	glViewport(0, 0, UI->width(),UI->height());

	// Perspective
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, GLdouble(UI->width()) / UI->height(), 0.1, 10000.0);

	// Enables
//	glEnable(GL_CULL_FACE); 
//	glCullFace(GL_FRONT); 
//	glEnable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glShadeModel(GL_SMOOTH);

	// Set view matrix for the first time
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, UI->translate_z());
	glRotatef(UI->rotate_x(), 1.0f, 0.0f, 0.0f);
	glRotatef(UI->rotate_y(), 0.0f, 1.0f, 0.0f);

	UI->init_light();

	glClear(GL_COLOR_BUFFER_BIT);
}

/****************************
 *        display           *
 ****************************/
void callback_display()
{
	if(UI->restart())
		QLB_system = UI->reset(QLB_system);
	
	// Adjust simulation parameter if something has changed
	if(UI->param_has_changed())
	{
		QLB_system->change_scaling(UI->change_scaling());
		UI->reset_change_scaling();
	
		QLB_system->set_current_scene(UI->current_scene());
		QLB_system->set_current_render(UI->current_render());
		QLB_system->set_draw_potential(UI->draw_potential());
		
		UI->reset_param_has_changed();
	}
		
	// Reset color's and clear bits	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Update cpu/gpu usage/memory etc..
	UI->update_performance_counter();

	// Set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	if( UI->rotating() ) UI->set_rotate_y(UI->rotate_y() - 0.15f);
	glTranslatef(0.0f, 0.0f, UI->translate_z());
	glRotatef(UI->rotate_x(), 1.0f, 0.0f, 0.0f);
	glRotatef(UI->rotate_y(), 0.0f, 1.0f, 0.0f);

	// Advance system
	switch(QLB_system->opt().device())
	{
		case 0: // CPU serial
			if(!UI->paused())
				QLB_system->evolution_CPU_serial();
			QLB_system->calculate_vertex(0, 1);
			QLB_system->calculate_normal();
			QLB_system->render();
			break;
		case 1: // CPU multi threaded
		{ 
			for(std::size_t tid = 0; tid < threadpool.size(); ++tid)
				threadpool[tid] = std::thread( &QLB::calculate_vertex, 
				                                QLB_system, 
				                                int(tid),
				                                int(threadpool.size()) );
			for(std::thread& t : threadpool)
				t.join();
				
			if(!UI->paused())
			{
				// Overlap computation and drawing
				for(std::size_t tid = 0; tid < threadpool.size(); ++tid)
					threadpool[tid] = std::thread( &QLB::evolution_CPU_thread, 
					                               QLB_system, 
					                               int(tid) ); 

				QLB_system->calculate_normal();
				QLB_system->render();
		
				for(std::thread& t : threadpool)
					t.join();
			}
			else
			{				
				QLB_system->calculate_normal();		
				QLB_system->render();
			}
			break;
		}
		case 2: // GPU CUDA
			FATAL_ERROR("CUDA version is not yet implemented");
			break;
	}

	// Draw text boxes
	UI->draw();

	glutSwapBuffers();
	glutPostRedisplay();
}

/****************************
 *        reshape           *
 ****************************/
void callback_reshape(int width, int height) 
{
	if(height == 0) height = 1;
	if(width == 0) width = 1;
	
	UI->set_width(width);
	UI->set_height(height);

	// Viewport
	glViewport(0, 0, UI->width(),UI->height());

	// Perspective
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, GLdouble(UI->width()) / UI->height(), 0.1, 10000.0);

	// Set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, UI->translate_z());
	glRotatef(UI->rotate_x(), 1.0f, 0.0f, 0.0f);
	glRotatef(UI->rotate_y(), 0.0f, 1.0f, 0.0f);
	UI->init_light();
}

/****************************
 *        keyboard          *
 ****************************/
void callback_keyboard(unsigned char key, int x, int y)
{
	if(key == 27) cleanup_and_exit(); // Esc
	UI->keyboard(int(key),x,y);
}	

/****************************
 *        keyboard_2        *
 ****************************/
void callback_keyboard_2(int key, int x, int y)
{
	if(key == 27) cleanup_and_exit(); // Esc
	UI->keyboard(key,x,y);
}

/****************************
 *          mouse           *
 ****************************/
void callback_mouse(int button, int state, int x, int y)
{
	UI->mouse(button,state,x,y);
}

/****************************
 *       mouse_motion       *
 ****************************/
void callback_mouse_motion(int x, int y)
{
	UI->mouse_motion(x,y);
}

/****************************
 *         cleanup          *
 ****************************/
void cleanup_and_exit()
{
	// Delete the local class pointers explicitly, otherwise the destructors are 
	// not called if we leave the glutMainLoop() with exit()
	if(UI != nullptr)
	{
		if(cmd->verbose())
		{
			std::cout << " === Clean up === " << std::endl;
			std::printf("cleaning up ... UI at %p\n", UI); 
		}
		delete UI;
		UI = nullptr;
	}
	
	if(QLB_system != nullptr) 
	{	
		if(cmd->verbose())
			std::printf("cleaning up ... QLB_system at %p\n", QLB_system); 
		
		delete QLB_system;
		QLB_system = nullptr;
	}
	
	if(cmd != nullptr)
	{
		if(cmd->verbose()) 
			std::printf("cleaning up ... cmd at %p\n", cmd); 
		delete cmd;
		cmd = nullptr;
	}
	
	// We exit now
	exit(EXIT_SUCCESS);
}	
