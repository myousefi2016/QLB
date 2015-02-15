/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Implements a class to handle various user interactions.
 */
 
#include "GLUTui.hpp"

UserInterface::UserInterface(int width, int height, const char* title,
	                         float translate_z)
	:	
		// === Window variables ===
		width_(width),
		height_(height),
		title_(const_cast<char*>(title)),
		// === Camera variables ===
		translate_z_(translate_z),
		rotate_x_(25.0f),
		rotate_y_(-40.0f),
		// === Mouse variables ===
		mouse_old_x_(0),
		mouse_old_y_(0),
		mouse_button_(0),
		// === Retart/Pause variables ===
		paused_(false),
		restart_(false),
		// === FPS ===
		frame_count_(0),
		time_(0),
		fps_(0.0f),
		// === Font ===
		
		// === Set parameters ===
		param_has_changed_(false),
		change_scaling_(0),
		current_scene_(QLB::spinor0),
		current_render_(QLB::SOLID),
		// === Light ===
		light_()
{}

UserInterface::~UserInterface()
{}

void UserInterface::keyboard(int key, int x, int y)
{
	switch(key)
	{
		case 27:    // Esc
			exit(EXIT_SUCCESS);
			break;
		case 8:     // Backspace
			restart_ = !restart_;
			break;
		case 114:   // r
			restart_ = !restart_;
			break;
		case 32:    // Space
			paused_ = !paused_;
			break;
		case 43:    // +
			param_has_changed_ = true;
			change_scaling_ = 1;
			break;
		case 45:    // -
			param_has_changed_ = true;
			change_scaling_ = -1;
			break;
		case 49:    // 1
			param_has_changed_ = true;
			current_scene_ = QLB::spinor0;
			break;
		case 50:    // 2
			param_has_changed_ = true;
			current_scene_ = QLB::spinor1;
			break;
		case 51:    // 3
			param_has_changed_ = true;
			current_scene_ = QLB::spinor2;
			break;
		case 52:    // 4
			param_has_changed_ = true;
			current_scene_ = QLB::spinor3;
			break;
		case 118:   // v
			param_has_changed_ = true;
			current_scene_ = QLB::potential;
			break;
		case 119:   // w
			param_has_changed_ = true;
			if(current_render_ == QLB::SOLID)
				current_render_ = QLB::WIRE;
			else
				current_render_ = QLB::SOLID;
			break;
	}
}

void UserInterface::eye_position(GLdouble& x, GLdouble& y, GLdouble& z) const
{
	GLint viewport[4];
	GLdouble modelMatrix[16];
	GLdouble projMatrix[16];

	// Get the matrices
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix ); 
	glGetDoublev(GL_PROJECTION_MATRIX, projMatrix  ); 
	glGetIntegerv(GL_VIEWPORT, viewport); 
	
	const GLdouble winx = (viewport[2]-viewport[0])/2.0;
	const GLdouble winy = (viewport[3]-viewport[1])/2.0;
	const GLdouble winz = 0.0;
	
	// Unproject
	gluUnProject(winx, winy, winz, modelMatrix, projMatrix, viewport, 
		         &x, &y, &z);
}

void UserInterface::mouse(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
        mouse_button_ = button;
    else if(state == GLUT_UP)
        mouse_button_ = 0;

	// Handle mouse wheel
	if(button == WHEEL_UP)
		translate_z_ +=  0.01f*std::abs(translate_z_);
	else if(button == WHEEL_DOWN)
		translate_z_ -=  0.01f*std::abs(translate_z_);

	mouse_old_x_ = x;
	mouse_old_y_ = y;
}

void UserInterface::mouse_motion(int x, int y)
{
	// Translation of the mouse pointer since GLUT_DOWN
    float dx = float(x - mouse_old_x_);
    float dy = float(y - mouse_old_y_);

	switch(mouse_button_)
	{
		case MOUSE_LEFT:
			rotate_x_ += dy * 0.2f;
			rotate_y_ += dx * 0.2f;
			break;
		case MOUSE_MIDDLE:
			break;
		case MOUSE_RIGHT:
			break;
	}
	
    mouse_old_x_ = x;
    mouse_old_y_ = y;
}

float UserInterface::compute_fps()
{
	frame_count_++;
	
	// Get time in ms since glutInit()
	int cur_time = glutGet(GLUT_ELAPSED_TIME); 
	int time_interval = cur_time - time_;

	// Calculate fps after FPS_UPDATE_FRQ (ms) passed
	if(time_interval > FPS_UPDATE_FRQ)
	{
		fps_  = frame_count_ /(time_interval / 1000.0f);
		time_ = cur_time;
		frame_count_ = 0;
	}

	return fps_;
}

QLB* UserInterface::reset(QLB* qlb_old)
{
	QLB* new_qlb = new QLB(qlb_old->L(), 
	                       qlb_old->dx(), 
	                       qlb_old->mass(), 
	                       qlb_old->dt(), 
	                       qlb_old->V(),
	                       qlb_old->plot(),
	                       qlb_old->verbose());
	delete qlb_old;
	new_qlb->init_GL();	
	
	restart_ = !restart_;
	current_render_ = new_qlb->current_render();
	current_scene_  = new_qlb->current_scene();
	
	return new_qlb;
}

void UserInterface::init_light()
{	
	GLdouble x,y,z;
	eye_position(x, y, z);
	light_.init(GLfloat(x), GLfloat(y), GLfloat(z));
}
