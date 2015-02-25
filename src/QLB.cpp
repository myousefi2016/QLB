/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Implementations of the quantum lattice Boltzmann methods which are shared
 *	by the CPU and GPU implementation.
 */
 
#include "QLB.hpp"

// === CONSTRUCTOR === 

QLB::QLB(unsigned L, float_t dx, float_t mass, float_t dt, int V_indx, QLBopt opt)
	:	
		// === Simulation variables ===
		L_(L),
		dx_(dx),
		mass_(mass),
		t_(0),
		dt_(dt),
		deltax_(0),
		deltay_(0),
		delta0_(14.0),
		V_indx_(V_indx),
		// === Arrays CPU ===
		spinor_(L),
		spinoraux_(L),
		spinorrot_(L),
		currentX_(L),
		currentY_(L),
		veloX_(L),
		veloY_(L),
		wrot_(L),
		rho_(L),
		// === OpenGL context ===
		GL_is_initialzed_(false),
		current_scene_(spinor0),
		current_render_(SOLID),
		scaling_(L/2.0),
		array_index_(2*L*(L-1)),
		array_vertex_(3*L*L),
		array_normal_(3*L*L), 
		// === IO ===
		opt_(opt)
{
	// Set initial condition
	initial_condition_gaussian();
	calculate_macroscopic_vars();
	
	// Copy arrays to device
#ifndef QLB_NO_CUDA
	allocate_device_arrays();
	init_device();
#endif
}

// === DESTRUCTOR ====

QLB::~QLB()
{
#ifndef QLB_NO_CUDA
	free_device_arrays();
#endif
}

// === INITIALIZATION ===

void QLB::initial_condition_gaussian()
{
	float_t gaussian;
	float_t x, y;
	const float_t stddev = 2*delta0_*delta0_;
	//const float_t C = 1.0 / (2*M_PI * stddev); (?)
	
	for(unsigned i = 0; i < L_; ++i)
	{
		for(unsigned j = 0; j < L_; ++j)
		{
			x = dx_*(i-0.5*(L_-1));
			y = dx_*(j-0.5*(L_-1));	
			gaussian = std::exp( -(x*x + y*y)/(2*stddev) );
			                           
			spinor_(i,j,0) = gaussian;
			spinor_(i,j,1) = 0;
			spinor_(i,j,2) = 0;
			spinor_(i,j,3) = 0;
		}
	}
}


// === SIMULATION ===

void QLB::evolution()
{
	evolution_CPU();

	// Update time;
	t_ += 1.0;
}


// === PRINTING ===

void QLB::print_spread()
{
	calculate_spread();
	
	// Schrödinger solution
	float_t deltax_t = std::sqrt( delta0_*delta0_ + t_*dt_*t_*dt_ / 
		                         (4.0*mass_*mass_*delta0_*delta0_) );

	std::cout << std::left << std::setprecision(6);

	std::cout << std::setw(15) << "time";
	std::cout << std::setw(15) << "deltaX";
	std::cout << std::setw(15) << "deltaY";
	if(V_indx_ == 0) std::cout << std::setw(15) << "Schrödinger";
	std::cout << std::endl;

	std::cout << std::setw(15) << t_*dt_;
	std::cout << std::setw(15) << deltax_;
	std::cout << std::setw(15) << deltay_;
	if(V_indx_ == 0) std::cout << std::setw(15) << deltax_t;	
	std::cout << std::endl;
}

/**
 *	Generic matrix print function (use with wrapper functions)
 *	@param	m		object which provides an operator[] e.g cmat_t
 *	@param	N		size (dimension)
 *	@param 	M		size (dimension)
 *	@param 	d 		size (dimension)
 *	@param 	k 		offset in the matrix (k-th vector element) 
 *	@param	out		stream to which the matrix will be written to
 */
template< class mat_t, class stream_t >
static void print_mat(const mat_t& m, const std::size_t N, const std::size_t M, 
                      const std::size_t d, const std::size_t k, stream_t& out)
{
	out << std::fixed;
	out << std::setprecision(5);
	out << std::left << std::endl;
	for(std::size_t i = 0; i < N; ++i)
	{
		for(std::size_t j = 0; j < M; ++j)
			out << std::setw(20) << m[d*(N*i + j) + k];
		out << std::endl;
	}
	out << std::endl;
}


/**
 *	Like print_mat but with special evaluation (this function does only work if
 *	the underlining type of mat_t is std::complex<T>)
  *	@param	m		object which provides an operator[] e.g cmat_t
 *	@param	N		size (dimension)
 *	@param 	M		size (dimension)
 *	@param 	d 		size (dimension)
 *	@param 	k 		offset in the matrix (k-th vector element) 
 *	@param	out		stream to which the matrix will be written to
 *	@param 	eval	special evaluation of the elements
 *	                0: m(i,j) 
 *	                1: std::real( m(i,j) )
 *	                2: std::imag( m(i,j) )
 *	                3: std::norm( m(i,j) )
 */

template< class mat_t, class stream_t >
static void print_mat_eval(const mat_t& m, const std::size_t N, const std::size_t M, 
                           const std::size_t d, const std::size_t k, stream_t& out,
                           const int eval)
{
	const int eval_flag = eval; 
	out << std::fixed;
	out << std::setprecision(5);
	out << std::left << std::endl;
	for(std::size_t i = 0; i < N; ++i)
	{
		for(std::size_t j = 0; j < M; ++j)
		{
			if(eval_flag == 0) 
				out << std::setw(20) << m[d*(N*i + j) + k];
			else if(eval_flag == 1)
				out << std::setw(20) << std::real(m[d*(N*i + j) + k]);
			else if(eval_flag == 2)
				out << std::setw(20) << std::imag(m[d*(N*i + j) + k]);
			else if(eval_flag == 3)
				out << std::setw(20) << std::norm(m[d*(N*i + j) + k]);
		}
		out << std::endl;
	}
	out << std::endl;
}

void QLB::print_matrix(const cmat_t& m) const
{
	print_mat(m, m.N(), m.N(), 1, 0, std::cout);
}

void QLB::print_matrix(const fmat_t& m) const
{
	print_mat(m, m.N(), m.N(), 1, 0, std::cout);
}

void QLB::print_matrix(const c4mat_t& m, std::size_t k) const
{
	print_mat(m, m.N(), m.N(), 4, k, std::cout);
}

/**
 *	Inform which file is written to
 *	@param	file	filename
 */
template< typename msg_t >
static inline void verbose_write_to_file(msg_t filename)
{
	std::cout << "Writing to ... " << filename << std::endl;
}

void QLB::write_content_to_file()
{
	calculate_macroscopic_vars();

	// We have to mask and shift to get the bit
	
	// spinor1
	if((opt_.plot() & QLBopt::spinor1) >> 2 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("spinor1.dat");
		if(opt_.verbose()) verbose_write_to_file("spinor1.dat");
		print_mat(spinor_, L_, L_, 4, 0, fout);
		fout.close();	
	}
	
	// spinor2
	if((opt_.plot() & QLBopt::spinor2) >> 3 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("spinor2.dat");
		if(opt_.verbose()) verbose_write_to_file("spinor2.dat");
		print_mat(spinor_, L_, L_, 4, 1, fout);
		fout.close();	
	}

	// spinor3
	if((opt_.plot() & QLBopt::spinor3) >> 4 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("spinor3.dat");
		if(opt_.verbose()) verbose_write_to_file("spinor3.dat");
		print_mat(spinor_, L_, L_, 4, 2, fout);
		fout.close();	
	}

	// spinor4
	if((opt_.plot() & QLBopt::spinor4) >> 5 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("spinor4.dat");
		if(opt_.verbose()) verbose_write_to_file("spinor4.dat");
		print_mat(spinor_, L_, L_, 4, 3, fout);
		fout.close();	
	}

	// density
	if((opt_.plot() & QLBopt::density) >> 6 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("density.dat");
		if(opt_.verbose()) verbose_write_to_file("density.dat");
		print_mat_eval(rho_, L_, L_, 1, 0, fout, 3);
		fout.close();
	}

	// currentX
	if((opt_.plot() & QLBopt::currentX) >> 7 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("currentX.dat");
		if(opt_.verbose()) verbose_write_to_file("currentX.dat");
		print_mat_eval(currentX_, L_, L_, 1, 0, fout, 1);
		fout.close();	
	}

	// currentY
	if((opt_.plot() & QLBopt::currentY) >> 8 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("currentY.dat");
		if(opt_.verbose()) verbose_write_to_file("currentY.dat");
		print_mat_eval(currentY_, L_, L_, 1, 0, fout, 1);
		fout.close();	
	}

	// veloX
	if((opt_.plot() & QLBopt::veloX) >> 9 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("veloX.dat");
		if(opt_.verbose()) verbose_write_to_file("veloX.dat");
		print_mat_eval(veloX_, L_, L_, 1, 0, fout, 1);
		fout.close();	
	}

	// veloY
	if((opt_.plot() & QLBopt::veloY) >> 10 || (opt_.plot() & QLBopt::all))
	{ 
		fout.open("veloY.dat");
		if(opt_.verbose()) verbose_write_to_file("veloY.dat");
		print_mat_eval(veloY_, L_, L_, 1, 0, fout, 1);
		fout.close();	
	}
}

// === Constants === 

const QLB::complex_t QLB::one(1.0,0.0);
const QLB::complex_t QLB::img(0.0,1.0);

// X
static const QLB::complex_t X_[] =  
{-QLB::one, -QLB::one, -QLB::one, -QLB::one, 
  QLB::one, -QLB::one, -QLB::one,  QLB::one, 
 -QLB::one,         0,         0,  QLB::one, 
  QLB::one,  QLB::one, -QLB::one, -QLB::one};
const QLB::cmat_t QLB::X(4, X_, X_+16);

// Y
static const QLB::complex_t Y_[] =  
{-QLB::one, -QLB::one, -QLB::img,  QLB::img, 
  QLB::one, -QLB::one, -QLB::img, -QLB::img, 
 -QLB::one,         0,         0, -QLB::img, 
  QLB::one,  QLB::one, -QLB::img,  QLB::img};
const QLB::cmat_t QLB::Y(4, Y_, Y_+16);

#ifndef QLB_SINGLE_PRECISION 

// Xinv
static const QLB::complex_t Xinv_[] =   
{-QLB::one/4.0,  QLB::one/4.0, -QLB::one/2.0,             0, 
 -QLB::one/4.0, -QLB::one/4.0,  QLB::one/2.0,  QLB::one/2.0, 
 -QLB::one/4.0, -QLB::one/4.0, -QLB::one/2.0, -QLB::one/2.0, 
 -QLB::one/4.0,  QLB::one/4.0,  QLB::one/2.0,             0};
const QLB::cmat_t QLB::Xinv(4, Xinv_, Xinv_+16);

// Yinv
static const QLB::complex_t Yinv_[] =   
{-QLB::one/4.0,  QLB::one/4.0, -QLB::one/2.0,             0, 
 -QLB::one/4.0, -QLB::one/4.0,  QLB::one/2.0,  QLB::one/2.0, 
  QLB::img/4.0,  QLB::img/4.0,  QLB::img/2.0,  QLB::img/2.0, 
 -QLB::img/4.0,  QLB::img/4.0,  QLB::img/2.0,             0};
const QLB::cmat_t QLB::Yinv(4, Yinv_, Yinv_+16);

#else

// Xinv
static const QLB::complex_t Xinv_[] =   
{-QLB::one/4.0f,  QLB::one/4.0f, -QLB::one/2.0f,             0, 
 -QLB::one/4.0f, -QLB::one/4.0f,  QLB::one/2.0f,  QLB::one/2.0f, 
 -QLB::one/4.0f, -QLB::one/4.0f, -QLB::one/2.0f, -QLB::one/2.0f, 
 -QLB::one/4.0f,  QLB::one/4.0f,  QLB::one/2.0f,             0};
const QLB::cmat_t QLB::Xinv(4, Xinv_, Xinv_+16);

// Yinv
static const QLB::complex_t Yinv_[] =   
{-QLB::one/4.0f,  QLB::one/4.0f, -QLB::one/2.0f,             0, 
 -QLB::one/4.0f, -QLB::one/4.0f,  QLB::one/2.0f,  QLB::one/2.0f, 
  QLB::img/4.0f,  QLB::img/4.0f,  QLB::img/2.0f,  QLB::img/2.0f, 
 -QLB::img/4.0f,  QLB::img/4.0f,  QLB::img/2.0f,             0};
const QLB::cmat_t QLB::Yinv(4, Yinv_, Yinv_+16);

#endif

// alphaX
static const QLB::complex_t alphaX_[] = 
{        0,         0,         0,  QLB::one, 
         0,         0,  QLB::one,         0, 
         0,  QLB::one,         0,         0, 
  QLB::one,         0,         0,         0};
const QLB::cmat_t QLB::alphaX(4, alphaX_, alphaX_+16);

// alphaY
static const QLB::complex_t alphaY_[] = 
{        0,         0,         0, -QLB::img, 
         0,         0,  QLB::img,         0, 
         0, -QLB::img,         0,         0, 
  QLB::img,         0,         0,         0};
const QLB::cmat_t QLB::alphaY(4, alphaY_, alphaY_+16);

// Beta
static const QLB::complex_t beta_[] =
{ QLB::one,         0,         0,         0, 
         0,  QLB::one,         0,         0, 
         0,         0, -QLB::one,         0, 
         0,         0,         0, -QLB::one};
const QLB::cmat_t QLB::beta(4, beta_, beta_+16);
