/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Class to handle Light and shading, this class is part of
 *  the class UserInterface from GLUTui.hpp
 */

#ifndef GLUT_LIGHT_HPP
#define GLUT_LIGHT_HPP

// System includes
#include <vector>

// Local includes
#include "GLerror.hpp"

/********************
 *      Light       *
 ********************/
class Light
{
public:
	typedef std::vector<GLfloat> fvec_t;

	Light();

	/**
	 *	Setup all necessary glEnable's - this call needs a existing
	 *	OpenGL context to work
	 *	@param	x	x-coordinate of the position of the light
 	 *	@param	y	y-coordinate of the position of the light
	 *	@param	z	z-coordinate of the position of the light
	 */
	void init(GLfloat x, GLfloat y, GLfloat z);

	/**
	 *	Enable light (adjust the position)
	 *	@param	x	x-coordinate
 	 *	@param	y	y-coordinate
	 *	@param	z	z-coordinate
	 */
	void enable(GLfloat x, GLfloat y, GLfloat z);

	/**
	 *	Disable light
	 */
	void disable() const;

private:
	fvec_t light_ambient_;
	fvec_t light_diffuse_;
	fvec_t light_specular_;
	fvec_t light_position_;
 
	fvec_t mat_ambient_;
	fvec_t mat_diffuse_;
	fvec_t mat_specular_;
	fvec_t high_shininess_;
};

#endif
