/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Class to handle lightning
 */

#include "GLUTlight.hpp"

static const GLfloat light_ambient[4]  = { 0.0f, 0.0f, 0.0f, 1.0f };
static const GLfloat light_diffuse[4]  = { 1.0f, 1.0f, 1.0f, 1.0f };
static const GLfloat light_specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
static const GLfloat light_position[4] = { 10.0f, 10.0f, 10.0f, 0.0f }; 
 
static const GLfloat mat_ambient[4]    = { 0.7f, 0.7f, 0.7f, 1.0f };
static const GLfloat mat_diffuse[4]    = { 0.8f, 0.8f, 0.8f, 1.0f };
static const GLfloat mat_specular[4]   = { 1.0f, 1.0f, 1.0f, 1.0f };
static const GLfloat high_shininess[1] = { 25.0f };

Light::Light()
	:
	light_ambient_(light_ambient,light_ambient+4),
	light_diffuse_(light_diffuse,light_diffuse+4),
	light_specular_(light_specular,light_specular+4),
	light_position_(light_position,light_position+4),
	mat_ambient_(mat_ambient,mat_ambient+4),
	mat_diffuse_(mat_diffuse,mat_diffuse+4),
	mat_specular_(mat_specular,mat_specular+4),
	high_shininess_(high_shininess,high_shininess+1)
{}

void Light::init(GLfloat x, GLfloat y, GLfloat z)
{
	light_position_[0] = x;
	light_position_[1] = y;
	light_position_[2] = z;

	glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING); 
 
    glLightfv(GL_LIGHT0, GL_AMBIENT,  &light_ambient_[0]);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  &light_diffuse_[0]);
    glLightfv(GL_LIGHT0, GL_SPECULAR, &light_specular_[0]);
    glLightfv(GL_LIGHT0, GL_POSITION, &light_position_[0]); 
 
    glMaterialfv(GL_FRONT, GL_AMBIENT,   &mat_ambient_[0]);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   &mat_diffuse_[0]);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  &mat_specular_[0]);
    glMaterialfv(GL_FRONT, GL_SHININESS, &high_shininess_[0]);
    
    glDisable(GL_NORMALIZE);
}

void Light::enable(GLfloat x, GLfloat y, GLfloat z)
{
	init(x,y,z);
}

void Light::disable() const
{
	glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING); 
}
