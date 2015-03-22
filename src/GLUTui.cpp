/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Implements a class to handle various user interactions.
 */
 
#include "GLUTui.hpp"

UserInterface::UserInterface(int width, int height, const char* title,
	                         float translate_z, bool static_viewer)
	:	
		// === Window variables ===
		width_(width),
		height_(height),
		title_(const_cast<char*>(title)),
		static_viewer_(static_viewer),
		// === Camera variables ===
		translate_z_(translate_z),
		rotate_x_(25.0f),
		rotate_y_(-40.0f),
		rotating_(false),
		// === Mouse variables ===
		mouse_old_x_(0),
		mouse_old_y_(0),
		mouse_button_(0),
		// === Restart/Pause variables ===
		paused_(false),
		restart_(false),
		// === PerformanceCounter ===
		frame_count_(0),
		time_(0),
		fps_(0.0f),
		// === Set parameters ===
		param_has_changed_(false),
		change_scaling_(0),
		current_scene_(QLB::spinor0),
		current_render_(QLB::SOLID),
		draw_potential_(false),
		dump_simulation_(false),
		// === Light ===
		light_()
{

	text_boxes_.resize(4);		

	// BOX_HELP_DETAIL
	TextBox::svec_t text(15);
	text[0] = "Esc    - Exit program      ";
	text[1] = "Space  - Pause/unpause     ";
	text[2] = "+/-    - Change scaling    ";
	text[3] = "R      - Restart           ";
	text[4] = "W      - Activate Wireframe";
	text[5] = "1    - Draw spinor 1 "; 
	text[6] = "2    - Draw spinor 2 ";
	text[7] = "3    - Draw spinor 3 ";
	text[8] = "4    - Draw spinor 4 ";
	text[9] = "V    - Draw potential";
	text[10] = "S    - Show Performance   "; 
	text[11] = "C    - Rotating camera    ";
	text[12] = "D    - Dump the simulation";
	text[13] = "                          ";
	text[14] = "                          ";
	
	text_boxes_[BOX_HELP_DETAIL].init(-0.985f, -0.975f, 1.90f, 0.25f, 5, 3, 
	                                  true, true, true, true, 1.0f);
	text_boxes_[BOX_HELP_DETAIL].add_text(text.begin(), text.end());
	text_boxes_[BOX_HELP_DETAIL].deactivate();

	// BOX_HELP_ASK
	text.resize(1);	
	text[0] = "Press H for detailed help";

	text_boxes_[BOX_HELP_ASK].init(-0.99f, -0.99f, 0.5f, 0.06f, 1, 1, 
	                               false, false, false, false, 1.0f);
	text_boxes_[BOX_HELP_ASK].add_text(text.begin(), text.end());
	
	
	// BOX_PERFORMANCE
	text.resize(5);	
	text[0] = "FPS           59    ";
	text[1] = "CPU memory    1.0 GB";
	text[2] = "CPU usage     59 %  ";
	text[3] = "GPU memory    2.0 GB";
	text[4] = "GPU usage     59 %  ";

	text_boxes_[BOX_PERFORMANCE].init(-0.97f, 0.75f, 0.5f, 0.22f, 5, 1, 
	                                  true, true, false, true, -0.10f);
	text_boxes_[BOX_PERFORMANCE].add_text(text.begin(), text.end());
	text_boxes_[BOX_PERFORMANCE].deactivate();
	
	// BOX_STATIC_VIEWER
	text.resize(1);
	text[0] = "Static Viewer";

	text_boxes_[BOX_STATIC_VIEWER].init(-0.13f, 0.91f, 0.5f, 0.06f, 1, 1, 
	                               false, false, false, false, 1.0f);
	text_boxes_[BOX_STATIC_VIEWER].add_text(text.begin(), text.end());
	if(!static_viewer_)
		text_boxes_[BOX_STATIC_VIEWER].deactivate();
}

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
		case 100:   // d
			param_has_changed_ = true;
			dump_simulation_ = true;
			break;
		case 99:    // c
			rotating_ = !rotating_;
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
			draw_potential_ = !draw_potential_;
			break;
		case 119:   // w
			param_has_changed_ = true;
			if(current_render_ == QLB::SOLID)
				current_render_ = QLB::WIRE;
			else
				current_render_ = QLB::SOLID;
			break;
		case 104:   // h
			if(text_boxes_[BOX_HELP_ASK].is_active())
			{
				text_boxes_[BOX_HELP_ASK].deactivate();
				text_boxes_[BOX_HELP_DETAIL].activate();
			}
			else
			{
				text_boxes_[BOX_HELP_ASK].activate();
				text_boxes_[BOX_HELP_DETAIL].deactivate();
			}
			break;
		case 115:   // s
			if(text_boxes_[BOX_PERFORMANCE].is_active())
				text_boxes_[BOX_PERFORMANCE].deactivate();
			else
				text_boxes_[BOX_PERFORMANCE].activate();
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
	rotating_ = false;

	if(state == GLUT_DOWN)
		mouse_button_ = button;
	else if(state == GLUT_UP)
		mouse_button_ = 0;

	// Handle mouse wheel
	if(button == WHEEL_UP)
		translate_z_ +=  0.015f*std::abs(translate_z_);
	else if(button == WHEEL_DOWN)
		translate_z_ -=  0.01f*std::abs(translate_z_);

	mouse_old_x_ = x;
	mouse_old_y_ = y;
}

void UserInterface::mouse_motion(int x, int y)
{
	rotating_ = false;

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

void UserInterface::draw() const
{
	for(std::size_t i = 0; i < text_boxes_.size(); ++i)
		if(text_boxes_[i].is_active()) 
			text_boxes_[i].draw(width_, height_);
}

void UserInterface::update_performance_counter()
{
	frame_count_++;
	
	// Get time in ms since glutInit()
	int cur_time = glutGet(GLUT_ELAPSED_TIME); 
	int time_interval = cur_time - time_;

	// Update all PerformanceCounter variables and calculate fps after 
	// FPS_UPDATE_FRQ (ms) passed
	if(time_interval > FPS_UPDATE_FRQ)
	{
		fps_  = frame_count_ /(time_interval / 1000.0f);
		time_ = cur_time;
		frame_count_ = 0;

		// Update FPS
		char entry[100]; 
		SPRINTF(entry, "FPS             %2.0f   ", fps_);
		text_boxes_[BOX_PERFORMANCE].add_text(0, entry);
		
		// Update CPU memory
		SPRINTF(entry, "CPU memory    %4.0f MB ", pc_.cpu_memory()*1e-6);
		text_boxes_[BOX_PERFORMANCE].add_text(1, entry);
		
		// Update CPU usage
		SPRINTF(entry, "CPU usage       %2.0f %% ", pc_.cpu_usage());
		text_boxes_[BOX_PERFORMANCE].add_text(2, entry);
	
		// Update GPU memory
#ifdef QLB_HAS_CUDA
		SPRINTF(entry, "GPU memory    %4.0f MB ", pc_.gpu_memory()*1e-6);
#else
		SPRINTF(entry, "GPU memory     %s   ", "N/A");
#endif
		text_boxes_[BOX_PERFORMANCE].add_text(3, entry);
	
		// Update GPU usage
#if defined(QLB_HAS_CUDA) && (defined(_WIN32) || \
                             (defined(__linux__) && defined(QLB_HAS_GPU_COUNTER)))
		SPRINTF(entry, "GPU usage       %2.0f %% ", pc_.gpu_usage());
#else 
		SPRINTF(entry, "GPU usage      %s   ", "N/A");
#endif
		text_boxes_[BOX_PERFORMANCE].add_text(4, entry);
	}	
}

QLB* UserInterface::reset(QLB* qlb_old)
{
	QLB* new_qlb = new QLB(qlb_old->L(), 
	                       qlb_old->dx(), 
	                       qlb_old->mass(), 
	                       qlb_old->dt(), 
	                       qlb_old->V(),
	                       qlb_old->opt());
	delete qlb_old;
	new_qlb->init_GL(static_viewer_);	
	
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
