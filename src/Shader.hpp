/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Providing a classes to handle GLSL Shaders.
 */

#ifndef SHADER_LOADER
#define SHADER_LOADER

// System includes
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <exception>

// Local includes
#include "GLerror.hpp"
#include "error.hpp"

class ShaderException : public std::exception
{
public:
	/**
	 *	Constructor - Handle exception thrown during shader compilation/linking
	 *	@param	msg		warning message
	 */
	ShaderException(std::string msg)
		: msg_(msg)
	{}

	virtual ~ShaderException() throw() {}

	/**
	 *	Exception handling
	 *	@return warning message passed during throw
	 */
	virtual const char* what() const throw()
	{
		return msg_.c_str(); 
	}
private:
	std::string msg_;
};


class ShaderLoader
{
public:
	/**
	 *	Constructor
	 */
	ShaderLoader();

	/**
	 *	Load a shader source code (GLSL) from file and compile it
	 *	@param	filename      file with shader source code
	 *	@param	shader_type   can be either:
	 *	                      GL_COMPUTE_SHADER 
	 *	                      GL_VERTEX_SHADER 
	 *	                      GL_TESS_CONTROL_SHADER 
	 *	                      GL_TESS_EVALUATION_SHADER 
	 *	                      GL_GEOMETRY_SHADER 
	 *	                      GL_FRAGMENT_SHADER
	 *	@param	debug	print the shader source to std::cout (optional)
	 */
	void load_from_file(const char* filename, GLenum shader_type, bool debug = false);

	/**
	 *	Load a shader source code (GLSL) from a string and compile it
	 *	@param	src            string with shader source code (const char*)
	 *	@param	shader_type    can be either:
	 *	                       GL_COMPUTE_SHADER 
	 *	                       GL_VERTEX_SHADER 
	 *	                       GL_TESS_CONTROL_SHADER 
	 *	                       GL_TESS_EVALUATION_SHADER 
	 *	                       GL_GEOMETRY_SHADER 
	 *	                       GL_FRAGMENT_SHADER
	 *	@param	debug	print the shader source to std::cout (optional)
	 */
	void load_from_string(const char* src, GLenum shader_type, bool debug = false);

	/**
	 *	Destructor
	 */
	~ShaderLoader();

	/**
	 *	Check if compilation was successful, throw if not
	 *	@param	shader		id of the shader
	 */
	void check_compilation(GLuint shader) const;

	// === Getter ===
	inline GLuint shader() const  { return shader_;   }
	inline bool is_valid() const  { return is_valid_; }

private:
	GLuint shader_;
	bool is_valid_;
};


class Shader
{
public:
	enum state_t { INVALID, VALID, UNINITIALIZED }; 

	/**
	 *	Constructor - default
	 */
	Shader();

	/**
	 *	Destructor
	 */
	~Shader();

	/**
	 * 	Link a single compiled shader into a program which can then be used with 
	 * 	Shader::use_shader(). If the program was already in-use this call will 
	 * 	destroy the existing program and create a new one.
	 * 	The shader should be compiled with the class ShaderLoader.
	 *	
	 * 	@param    shader   compiled shader
	 */
	void add_shader(const ShaderLoader& shader);

	/**
	 *	Link two compiled shaders (vertex/fragment) into a program which can then be 
	 *	used with Shader::use_shader(). The shaders should be compiled with the class 
	 *	ShaderLoader. If the program was already in-use this call will destroy the 
	 *	existing program and create a new one.
	 *
	 *	@param	shader_vertex	compiled vertex shader
	 *	@param	shader_fragment	compiled fragment shader
	 */
	void add_shaders(const ShaderLoader& shader_vertex, 
	                 const ShaderLoader& shader_fragment);

	/**
	 *	Installs the program and the attached shaders in the current rendering state 
	 */
	void use_shader() const;

	/**
	 *	Check if linking was successful, throw if not
	 *	@param	program		id of the program
	 */
	void check_linking(GLuint program) const;

	// === Getter ===
	inline GLuint program() const { return program_; }
	inline bool is_valid() const  { return program_state_ == VALID; }

private:
	GLuint program_;
	state_t program_state_;
};

#endif /* ShaderLoader.hpp */
