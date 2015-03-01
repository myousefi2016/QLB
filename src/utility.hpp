/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zürich
 *
 *	Several useful utility functions and classes (Timer)
 */

#ifndef UTILITY_HPP
#define UTILITY_HPP

// System includes 
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "error.hpp"

#ifdef _WIN32
 #define NOMINMAX
 #include <windows.h>
 #undef NOMINMAX
#else
 #include <unistd.h>
 #include <sys/time.h>
#endif

// Compiler Hints
#ifdef _WIN32
 #define FORCE_INLINE
 #define NO_RETURN     __declspec(noreturn)
#else
 #define FORCE_INLINE  __attribute__((always_inline)) 
 #define NO_RETURN     __attribute__((noreturn))
#endif

// Compile-time warnings #pragma message WARN("message")
#if _MSC_VER
 #define STRINGIFICATION_IMPL(x) #x
 #define STRINGIFICATION(x) STRINGIFICATION_IMPL(x)
 #define FILE_LINE_LINK __FILE__ "(" STRINGIFICATION(__LINE__) ") : "
 #define WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#else
 #define WARN(exp) ("WARNING: " exp)
#endif

// Branch prediction
#if defined(__GNUC__) && __GNUC__ >= 4
 #define LIKELY(x)   (__builtin_expect((x), 1))
 #define UNLIKELY(x) (__builtin_expect((x), 0))
#else
 #define LIKELY(x)   (x)
 #define UNLIKELY(x) (x)
#endif

// Windows workaround
#ifdef SPRINTF
#undef SPRINTF
#endif
#ifdef _MSC_VER
 #define SPRINTF(buf, ...) sprintf_s((buf), __VA_ARGS__)
#else
 #define SPRINTF(buf, ...) std::sprintf((buf), __VA_ARGS__)
#endif

/****************************
 *       Timer class        *
 ****************************/
#ifdef _WIN32 // Windows 
class Timer
{
public:
	Timer()
		:	freq_(0.0), t_start_(0), start_was_called_(false), t_total_(0), N_(0)
	{}
	
	/**
	 *	Start the timer
	 */
	inline void start()
	{
		LARGE_INTEGER lpFrequency;

		// Get CPU frequency [counts/sec]
		if(!QueryPerformanceFrequency(&lpFrequency))
			FATAL_ERROR("QueryPerformanceFrequency failed");	
		freq_ = double(lpFrequency.QuadPart);

		// Get current performance count [counts]
		if(!QueryPerformanceCounter(&lpFrequency))
			FATAL_ERROR("QueryPerformanceCounter failed");

		start_was_called_ = true;
		t_start_ = lpFrequency.QuadPart; 
	}
	
	/**
	 *	Stop the timer and update total time
	 *	@return	the time in seconds since start()
	 */
	inline double stop()
	{
		if(!start_was_called_) 
			WARNING("calling stop() without previously calling start()");

		LARGE_INTEGER lpFrequency;
		QueryPerformanceCounter(&lpFrequency);
		__int64 t_end = lpFrequency.QuadPart;

		// Update total variables
		t_total_ += double(t_end - t_start_)/freq_;
		N_++;

		start_was_called_ = false;
		return double(t_end - t_start_)/freq_;
	}

	/**
	 *	Reset all variables
	 */
	inline void reset()
	{
		freq_ = 0.0;
		t_start_ = 0;
		start_was_called_ = false;
		t_total_ = 0;
		N_ = 0;
	}
	
	/**
	 *	Total time since last reset
	 */
	inline double total()   const { return t_total_; }
	
	/**
	 *	Average time since last reset
	 */
	inline double average() const { return t_total_/N_; }

private:
	double  freq_;
	__int64	t_start_;
	bool start_was_called_;

	double t_total_;
	__int64 N_; 
};

#else // Linux / Max OSX 
class Timer
{
public:
	Timer()
		: t_start_(0.0), start_was_called_(false), t_total_(0.0), N_(0)
	{
    	gettimeofday(&t_, NULL);
	}
	
	/**
	 *	Start the timer
	 */
	inline void start()
	{
		gettimeofday(&t_, NULL);
		t_start_ = t_.tv_sec + (t_.tv_usec/1000000.0);

		start_was_called_ = true;
	}
	
	/**
	 *	Stop the timer and update total time
	 *	@return	The time in seconds since start()
	 */
	inline double stop()
	{
		if(!start_was_called_) 
			WARNING("calling stop() without previously calling start()");

		gettimeofday(&t_, NULL);
		double t_cur =  t_.tv_sec + (t_.tv_usec/1000000.0) - t_start_;

		// Update total variables
		t_total_ += t_cur;
		N_++;

		start_was_called_ = false;
		return t_cur;
	}

	/**
	 *	Reset all variables
	 */
	inline void reset()
	{
		gettimeofday(&t_, NULL);	
		t_start_ = 0.0;
		start_was_called_ = false;
		t_total_ = 0.0;
		N_ = 0; 
	}

	/**
	 *	Total time since last reset
	 */
	inline double total()   const { return t_total_; }
	
	/**
	 *	Average time since last reset
	 */
	inline double average() const { return t_total_/N_; }

private:
	struct timeval t_;	
	double t_start_;
	bool start_was_called_;

	double t_total_;
	std::size_t N_; 
};
#endif

#endif /* utility.hpp */
