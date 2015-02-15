/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zürich
 *
 *	Error handling code for fatal_error's and warning's with support
 *	for coloring of terminal messages (Windows/UNIX).
 *	
 *	The functions should be called with FATAL_ERROR("message") and 
 *	WARNING("message").
 */

#ifndef ERROR_HPP
#define ERROR_HPP

#include <iostream>
#include <vector>
#include <exception>
#include <cstdlib>

#ifdef _WIN32
 #include <windows.h>
#else 
 #include <string>
 #include <unistd.h>
#endif 

#ifdef _WIN32
 #define NO_RETURN		__declspec(noreturn)
#else
 #define NO_RETURN		__attribute__((noreturn))
#endif

#define COLOR_RED       0
#define COLOR_GREEN     1
#define COLOR_MAGENTA   2
#define COLOR_WHITE	    3

// API to use the functions
#define FATAL_ERROR(msg) { fatal_error((msg), __FILE__, __LINE__); }
#define WARNING(msg)     { warning((msg), __FILE__, __LINE__); }

/****************************
 *        Coloring          *
 ****************************/
class ConsoleColorAPI
{
public:
	/**
	 *	Set the console color
	 *	@param colour	0: red
	 *	                1: green
	 *	                2: magenta
	 *					3: strong white
	 */
	virtual void set_color(int color) = 0;
	
	/**
	 *	Reset the console color to the state before calling the constructor or
	 *	to black if no value was obtained
	 */
	virtual void reset_color() = 0; 
};

#ifdef _WIN32

class ConsoleColor : public ConsoleColorAPI
{
public:
	ConsoleColor()
	{
		hstdout_ = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hstdout_, &console_state_ );	
	
		color_table_.push_back(0x0C); // red
		color_table_.push_back(0x0A); // green
		color_table_.push_back(0x0D); // magenta
		color_table_.push_back(0x0F); // strong white
	}
	
	~ConsoleColor() 
	{
		reset_color();
	}
	
	void set_color(int color)
	{
		// We don't do anything if color index is out of bounds
		try
		{
			SetConsoleTextAttribute(hstdout_, color_table_.at(color));
		}
		catch(...) {}
	}
	
	void reset_color()
	{	
		SetConsoleTextAttribute(hstdout_, console_state_.wAttributes );
	}
	
private:
	HANDLE hstdout_;	
	CONSOLE_SCREEN_BUFFER_INFO console_state_;

	std::vector<WORD> color_table_;
};

#else /* UNIX */ 

class ConsoleColor : public ConsoleColorAPI
{
public:
	ConsoleColor()
	{
		is_terminal_ = isatty(STDOUT_FILENO);
		
		color_table_.push_back("\x1b[1;31m"); 	// red
		color_table_.push_back("\x1b[1;32m"); 	// green
		color_table_.push_back("\x1b[1;35m"); 	// magenta
		color_table_.push_back("\x1b[1m"); 		// strong white
	}
	
	~ConsoleColor() 
	{
		reset_color();
	}
	
	void set_color(int color)
	{
		if(is_terminal_)
		{
			try
			{
				std::cout << color_table_.at(color) << std::flush;
				std::cerr << color_table_.at(color) << std::flush;		
			}
			catch(...) {}
		}
	}
	
	void reset_color()
	{	
		if(is_terminal_)
		{
			std::cout << "\x1b[0m" << std::flush;
			std::cerr << "\x1b[0m" << std::flush;
		}
	}
	
private:
	bool is_terminal_;
	
	std::vector<std::string> color_table_;
};

#endif

/****************************
 *     Error handling       *
 ****************************/
template< typename msg_t >
NO_RETURN static inline void fatal_error(const msg_t errmsg , const char *file, int line)
{
	ConsoleColor cc;
	cc.set_color(COLOR_WHITE);
	std::cerr << file << ":" << line;
	cc.set_color(COLOR_RED);
	std::cerr << " error: ";
	cc.reset_color();
	std::cerr << errmsg << std::endl;
	exit(EXIT_FAILURE);
}

/****************************
 *        Warnings          *
 ****************************/
template< typename msg_t >
static inline void warning(const msg_t warmsg , const char *file, int line)
{
	ConsoleColor cc;
	cc.set_color(COLOR_WHITE);
	std::cerr << file << ":" << line;
	cc.set_color(COLOR_MAGENTA);
	std::cerr << " warning: ";
	cc.reset_color();
	std::cerr << warmsg << std::endl;
}

#undef COLOR_RED
#undef COLOR_GREEN
#undef COLOR_MAGENTA
#undef COLOR_WHITE

#endif /* error.hpp */