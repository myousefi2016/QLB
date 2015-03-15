/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Providing a class to handle OpenGL Shaders (GLSL)
 */

#include "Shader.hpp"

static inline void get_error_log(GLuint object, GLuint type)
{
	GLint max_len = 0;	
	std::vector<GLchar> error_log;

	std::string warning_msg;

	switch(type)
	{
		case 0: // Shader Info
			std::cerr << " === SHADER COMPILATION LOG === " << std::endl;
			glGetShaderiv(object, GL_INFO_LOG_LENGTH, &max_len);
			error_log.resize(max_len);
			glGetShaderInfoLog(object, max_len, &max_len, &error_log[0]);
			warning_msg = "Shader compilation failed";
			break;
		case 1: // Program Info
			std::cerr << " === SHADER LINKING LOG === " << std::endl;
			glGetProgramiv(object, GL_INFO_LOG_LENGTH, &max_len);
			error_log.resize(max_len);
			glGetProgramInfoLog(object, max_len, &max_len, &error_log[0]);
			warning_msg = "Shader linking failed";
			break;
	}

	// Print the Log
	for(std::size_t i = 0; i < error_log.size(); ++i)
		std::cerr << error_log[i];
	std::cerr << std::endl;

	throw ShaderException(warning_msg);
}


ShaderLoader::ShaderLoader()
	:	shader_(0), is_valid_(false)
{}

void ShaderLoader::load_from_file(const char* filename, GLenum shader_type, bool debug)
{
	is_valid_ = true;
	try
	{
		std::stringstream sout; 
		std::ifstream fin(filename, std::ios::in);
		
		if(!fin.is_open())
		{
			std::string warn = "Cannot open shader source file : ";
			warn += std::string(filename == NULL ? "no such file" : filename);
			throw ShaderException(warn);
		}
		
		// Read the file into memory
		while(fin.good()) sout << (GLchar)fin.get();
		std::string shader_src = sout.str();

		if(shader_src.size() == 1)
			throw ShaderException("Empty shader source file");

		if(debug) 
			std::cout << shader_src << std::endl;

		fin.close();

		// Create the shader
		shader_ = glCreateShader(shader_type);
		glCheckLastError();

		// Load the shader from the shader source file
		const GLchar* src = shader_src.c_str();
		const GLint len = GLint(shader_src.size() - 1); 
		glShaderSource(shader_, 1, &src, &len);
	
		// Compile the shader
		glCompileShader(shader_); 

		check_compilation(shader_);
	}
	catch(std::exception &e)
	{
		// An error occurred (we issue a warning and continue)
		is_valid_ = false;
		WARNING(e.what());
	}
}

void ShaderLoader::load_from_string(const char* src, GLenum shader_type, bool debug)
{
	is_valid_ = true;
	try
	{
		if(src == NULL)
			throw ShaderException("Empty shader source");
	
		if(debug)
		{
			std::string shader_src(src);
			std::cout << shader_src << std::endl;
		}

		// Create the shader
		shader_ = glCreateShader(shader_type);
		glCheckLastError();

		// Load the shader from the shader source file
		const GLint len = GLint(std::strlen(src));
		glShaderSource(shader_, 1, &src, &len);
	
		// Compile the shader
		glCompileShader(shader_); 

		check_compilation(shader_);
	}
	catch(std::exception &e)
	{
		// An error occurred (we issue a warning and continue)
		is_valid_ = false;
		WARNING(e.what());
	}
}

void ShaderLoader::check_compilation(GLuint shader) const
{
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if(!status) get_error_log(shader, 0);
}

ShaderLoader::~ShaderLoader()
{
	glDeleteShader(shader_);
}


Shader::Shader()
	: program_(0), program_state_(UNINITIALIZED)
{}

void Shader::add_shader(const ShaderLoader& shader)
{
	// If the program was used previously we destroy the program
	if(program_state_ != UNINITIALIZED)
		glDeleteProgram(program_);
	program_state_ = VALID;

	try
	{
		if(!shader.is_valid()) 
			throw ShaderException("Shader is invalid");
	
		// Attach the shader to the program
		program_ = glCreateProgram();
		glAttachShader(program_, shader.shader());
		glCheckLastError();

		// Link the shader
		glLinkProgram(program_);
		glCheckLastError();
	
		check_linking(program_);
		
		glDetachShader(program_, shader.shader());
	}
	catch(std::exception &e)
	{
		// An error occured (we issue a warning and continue)
		program_state_ = INVALID;
		WARNING(e.what());
	}
}

void Shader::add_shaders(const ShaderLoader& shader_vertex, 
                         const ShaderLoader& shader_fragment)
{		
	// If the program was used previously we destroy the program
	if(program_state_ != UNINITIALIZED)
		glDeleteProgram(program_);
	program_state_ = VALID;

	try
	{
		// Check if everything is fine
		if(!shader_vertex.is_valid()) 
			throw ShaderException("Vertex shader is invalid");
			
		if(!shader_fragment.is_valid()) 
			throw ShaderException("Fragment shader is invalid");
	
		// Attach the shaders to the program
		program_ = glCreateProgram();
		glAttachShader(program_, shader_vertex.shader());
		glAttachShader(program_, shader_fragment.shader());
		glCheckLastError();

		// Link the shaders
		glLinkProgram(program_);
		glCheckLastError();
	
		check_linking(program_);

		glDetachShader(program_, shader_vertex.shader());
		glDetachShader(program_, shader_fragment.shader());
	}
	catch(std::exception &e)
	{
		// An error occured (we issue a warning and continue)
		program_state_ = INVALID;
		WARNING(e.what());
	}
}

Shader::~Shader()
{
	if(program_state_ != UNINITIALIZED)
		glDeleteProgram(program_);
}

void Shader::use_shader() const
{
	glUseProgram(program_);
}

void Shader::check_linking(GLuint program) const
{
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if(!status) get_error_log(program, 1);
}
