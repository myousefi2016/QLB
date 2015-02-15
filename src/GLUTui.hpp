/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	User Interface for the GLUT implementation to handle various user
 *	interaction.
 */
 
#ifndef GLUT_UI_HPP
#define GLUT_UI_HPP

#define _USE_MATH_DEFINES

// System incldues
#include <string>
#include <sstream>
#include <cmath>

// Local includes
#include "GLerror.hpp"
#include "error.hpp"
#include "QLB.hpp"
#include "GLUTlight.hpp"

// Defines
#define MOUSE_LEFT		0x0000
#define MOUSE_MIDDLE	0x0001
#define MOUSE_RIGHT		0x0002
#define WHEEL_UP		0x0003
#define WHEEL_DOWN		0x0004

#define FPS_UPDATE_FRQ	500 // ms

/****************************
 *     UserInterface        *
 ****************************/
class UserInterface
{
public:
	// === Initialization ===

	/** 
	 *	Constructor 
	 *	@param 	width		width of the initial window in pixels
	 *	@param	height 		height of the initial window in pixels
	 *	@param	title		title of the window
	 *	@param 	translate_z initial distance in z direction
	 */
	UserInterface(int width, int height, const char* title, float translate_z);
	
	~UserInterface();

	// === Methods ===

	/**
	 *	Caluclate current 'eye' position
	 *	@param	x	x-coordinate [out]
	 *	@param	y   y-coordinate [out]
	 *	@param	z   z-coordinate [out]
	 */
	void eye_position(GLdouble& x, GLdouble& y, GLdouble& z) const;
	
		/** 
	 *	Register keyboard actions and parses them 
	 *	@param 	key		integer value of the ASCII character of the pressed key
	 *	@param	x		current mouse position (x axis)
	 *  @param 	y		current mouse position (y axis)
	 */
	void keyboard(int key, int x, int y);

	/** 
	 *	Register mouse actions and parses them 
	 *	@param 	button	which button triggered the function
	 *	@param	state	which action was performed (GLUT_UP or GLUT_DOWN)
	 *	@param	x		current mouse position (x axis)
	 *  @param 	y		current mouse position (y axis)
	 */
	void mouse(int button, int state, int x, int y);

	/** 
	 *	Adjust the camera variables according to current mouse position 
	 *	and the old one
	 *	@param	x		current mouse position (x axis)
	 *  @param 	y		current mouse position (y axis)
	 */
	void mouse_motion(int x, int y);

	/**
	 *	Calculate frame's per second (FPS) using glut's built-in timer
	 *	@return	 current FPS
	 */
	float compute_fps();

	/**
	 *	Reset the whole simulation by initializing a new system with the old
	 *	one and deleting the old system aferwards
	 *	@param 	 qlb_old	system used to construct the new one
	 *	@return  qlb_new 	newly constructed system
	 */
	QLB* reset(QLB* qlb_old);

	/**
	 *	Setup lightning
	 */
	void init_light();
	
	// === Getter ===
	inline int height() const { return height_; }
	inline int width()  const { return width_;  }
	inline char* title()  const { return title_;  }

	inline float translate_z() const { return translate_z_; }
	inline float rotate_x() const { return rotate_x_; }
	inline float rotate_y() const { return rotate_y_; }

	inline bool paused()  const { return paused_;  }
	inline bool restart() const { return restart_; }
	
	inline bool param_has_changed() const { return param_has_changed_; }
	inline int change_scaling() const { return change_scaling_;  }
	inline QLB::scene_t current_scene() const { return current_scene_; }
	inline QLB::render_t current_render() const { return current_render_; }

	// === Setter ===
	inline void set_height(int height) { height_ = height; }
	inline void set_width(int width)   { width_  = width;  }
	
	// === Reset === 
	inline void reset_param_has_changed() { param_has_changed_ = false; }
	inline void reset_change_scaling() { change_scaling_ = 0; }

private:
	// === Window variables ===
	int width_;
	int height_;
	char* title_;

	// === Camera variables ===
	float translate_z_;
	float rotate_x_;
	float rotate_y_;

	// === Mouse variables ===
	int mouse_old_x_;
	int mouse_old_y_;
	int mouse_button_;
	
	// === Restart/Pause variables ===
	bool paused_;
	bool restart_;

	// === FPS ===
	int frame_count_;
	int time_;
	float fps_;
	
	// === Font ===
	
	// === Set parameters ===
	bool param_has_changed_;
	int change_scaling_; // -1: decrease 0: false 1: increase
	QLB::scene_t current_scene_;
	QLB::render_t current_render_;

	// === Light ===
	Light light_;
};

#endif /* GLUTui.hpp */
