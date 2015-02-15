/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zürich
 *
 *	Wrapper class for N dimensional square matrices (N x N) aswell as 
 *	N-dimensional square matrices with 4-dimensional vector as elements. 
 *	The matrix is stored in ROW MAJOR order, which means the LAST index is 
 *	varying the most.
 *	 
 *	The implementation will depend on the macro's MATRIX_USE_STL and 
 *	MATRIX_USE_CARRAY. If MATRIX_USE_CARRAY is defined plain c-arrays are used 
 *	as a container, otherwise std::vector is used i.e MATRIX_USE_STL is defined 
 *	by default. 
 *
 *	[EXAMPLE]
 *	Creating a simple 2x2 matrix
 *	
 *		a	b
 *		c	d
 *
 *	Initialization :
 *
 *	matND<double> matrix(2);	
 *	
 *	to use default initialization (i.e fill everything with 0.0)
 *
 *	matND<double> matrix(2,0.0)
 *
 *	To access the elements e.g get the value of c :
 *
 *	double c = matrix(1,0);
 *
 *	or
 *
 *	double c = matrix[2];
 *
 *	which are both equivalent.
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

// System includes
#include <cstddef>
#include <algorithm>

#if defined(_MSC_VER) && !defined(__clang__)
 #define MATRIX_FORCE_INLINE
 #define MATRIX_FORCE_ALIGNED_64  	__declspec(align(64))
#else
 #define MATRIX_FORCE_INLINE		__attribute__((always_inline)) 
 #define MATRIX_FORCE_ALIGNED_64	__attribute__((aligned(64)))
#endif

// VEC2D_USE_STL is default
#ifndef MATRIX_USE_CARRAY
 #undef  MATRIX_USE_STL
 #define MATRIX_USE_STL
#endif

/*****************************
 *      Matrix (N x N)       *
 *****************************/
#if defined( MATRIX_USE_STL )
#include <vector>

template < class value_t >
class matND : private std::vector<value_t>
{
public:

	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	val		initial value (default : 0)
	 */
	matND(std::size_t N, value_t val = value_t(0))
		: std::vector<value_t>(N*N,val), N_(N)
	{}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	begin	pointer to the begin of an array of size N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matND(std::size_t N, const value_t* begin, const value_t* end)
		: std::vector<value_t>(begin, end), N_(N)
	{}

	/**
	 *	Copy-Constructor
	 *	@param	v		matND used for copy constructing
	 */
	matND(const matND& v)
		: std::vector<value_t>(v.N()*v.N()), N_(v.N())
	{
		std::copy(v.begin(), v.end(), std::vector<value_t>::begin());
	}
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator[](std::size_t i) const
	{
		return std::vector<value_t>::operator[](i);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t& operator[](std::size_t i) 
	{
		return std::vector<value_t>::operator[](i);
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator()(std::size_t i, std::size_t j) const
	{
		return std::vector<value_t>::operator[](i*N_ + j);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE
	inline value_t& operator()(std::size_t i, std::size_t j)
	{
		return std::vector<value_t>::operator[](i*N_ + j);
	}
		
	// === Data access ===
	
	/**
	 *	Return one dimension of the matrix
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const value_t* data() const 
	{
		return &std::vector<value_t>::operator[](0); 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline value_t* data() 
	{ 
		return &std::vector<value_t>::operator[](0); 
	} 
	
private:
	std::size_t N_;
};

#else

template < class value_t >
class matND
{
public:
	
	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	val		initial value (default : 0)
	 */
	matND(std::size_t N, value_t val = value_t(0))
		: value_(new value_t[N*N]), N_(N)
	{
		std::fill(value_, value_+N*N, val);
	}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	begin	pointer to the begin of an array of size N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matND(std::size_t N, const value_t* begin, const value_t* end)
		: value_(new value_t[N*N]), N_(N)
	{
		// Visual Studio has a diffrent version of std::copy (from xutility)
		// we want to avoid using that
#ifdef _MSC_VER
		const std::size_t NN_const = N_*N_;
		for(std::size_t i = 0; i < NN_const; ++i)
			value_[i] = *(begin+i);
#else
		std::copy(begin, end, value_);
#endif
	}
	
	/**
	 *	Copy-Constructor
	 *	@param	v		matND used for copy constructing
	 */
	matND(const matND& v)
		: value_(new value_t[v.N()*v.N()]), N_(v.N())
	{
#ifdef _MSC_VER
		const std::size_t NN_const = N_*N_;
		for(std::size_t i = 0; i < NN_const; ++i)
			value_[i] = v[i];
#else
		std::copy(v.data(), v.data()+v.N()*v.N(), value_);
#endif
	}
	
	// === Destructor ===
	
	~matND() { delete value_; }
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE
	inline value_t const& operator()(std::size_t i, std::size_t j) const
	{
		return value_[i*N_ + j];
	}
	
	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t& operator()(std::size_t i, std::size_t j)
	{
		return value_[i*N_ + j];
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator[](std::size_t i) const
	{
		return value_[i];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t& operator[](std::size_t i) 
	{
		return value_[i];
	}

	// === Data access ===
	
	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const value_t* data() const 
	{ 
		return value_; 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline value_t* data() 
	{ 
		return value_; 
	} 
	
	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}
	
	/**
	 *	Size of the underlying array
	 *	@return N*N
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_; 
	}

private:
	MATRIX_FORCE_ALIGNED_64 value_t* value_;
	std::size_t N_;
};

#endif /* MATRIX_USE_STL */


/*****************************
 *    Matrix (N x N x 4)     *
 *****************************/
#if defined( MATRIX_USE_STL )

template < class value_t >
class matN4D : private std::vector<value_t>
{
public:

	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N x 4)
	 *	@param	val		initial value (default : 0)
	 */
	matN4D(std::size_t N, value_t val = value_t(0))
		: std::vector<value_t>(N*N*4,val), N_(N)
	{}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the vector (N x N x 4)
	 *	@param	begin	pointer to the begin of an array of size 4*N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matN4D(std::size_t N, const value_t* begin, const value_t* end)
		: std::vector<value_t>(begin, end), N_(N)
	{}
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator()(std::size_t i, std::size_t j, std::size_t k) const
	{
		return std::vector<value_t>::operator[](4*(N_*i + j) + k);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE
	inline value_t& operator()(std::size_t i, std::size_t j, std::size_t k)
	{
		return std::vector<value_t>::operator[](4*(N_*i + j) + k);
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator[](std::size_t i) const
	{
		return std::vector<value_t>::operator[](i);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t& operator[](std::size_t i) 
	{
		return std::vector<value_t>::operator[](i);
	}
	
		
	// === Data access ===
	
	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N*4
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_*4; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const value_t* data() const 
	{
		return &std::vector<value_t>::operator[](0); 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline value_t* data() 
	{ 
		return &std::vector<value_t>::operator[](0); 
	} 

private:
	std::size_t N_;
};

#else

template < class value_t >
class matN4D
{
public:
	
	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N x 4)
	 *	@param	val		initial value (default : 0)
	 */
	matN4D(std::size_t N, value_t val = value_t(0))
		: value_(new value_t[N*N*4]), N_(N)
	{
		std::fill(value_, value_+N*N*4, val);
	}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the vector (N x N x 4)
	 *	@param	begin	pointer to the begin of an array of size 4*N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matN4D(std::size_t N, const value_t* begin, const value_t* end)
		: value_(new value_t[N*N*4]), N_(N)
	{
		// Visual Studio has a diffrent version of std::copy (from xutility)
		// we want to avoid using that
#ifdef _MSC_VER
		const std::size_t NN_const = N*N*4;
		for(std::size_t i = 0; i < NN_const; ++i)
			value_[i] = *(begin+i);
#else
		std::copy(begin, end, value_);
#endif
	}
	
	// === Destructor ===
	
	~matN4D() { delete value_; }
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator()(std::size_t i, std::size_t j, std::size_t k) const
	{
		return value_[4*(N_*i + j) + k];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE
	inline value_t& operator()(std::size_t i, std::size_t j, std::size_t k)
	{
		return value_[4*(N_*i + j) + k];
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t const& operator[](std::size_t i) const
	{
		return value_[i];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline value_t& operator[](std::size_t i) 
	{
		return value_[i];
	}
	
	// === Data access ===

	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N*4
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_*4; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const value_t* data() const 
	{ 
		return value_; 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline value_t* data() 
	{ 
		return value_; 
	} 

private:
	MATRIX_FORCE_ALIGNED_64 value_t* value_;
	std::size_t N_;
};

#endif /* MATRIX_USE_STL */

#endif /* matrix.hpp */
