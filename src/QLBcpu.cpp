/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  This file contains the CPU implementations of the QLB scheme
 *	
 *  Based on the implementation of M. Mendoza (ETH Zurich)
 */

#include "QLB.hpp"

// === POTENTIAL ===

QLB::float_t QLB::V_harmonic(int i, int j) const 
{
	const float_t w0 = 1.0 / (2.0*mass_*delta0_*delta0_);
	const float_t x  = dx_*(i-0.5*(L_-1));
	const float_t y  = dx_*(j-0.5*(L_-1));

	return -0.5*mass_*w0*w0*(x*x + y*y);
}

QLB::float_t QLB::V_free(int i, int j) const
{
	return float_t(0);
}

QLB::float_t QLB::V_barrier(int i, int j) const
{
	const float_t delta0_2 = delta0_ * delta0_;
	const float_t V0 = L_*L_ / (32.0 * mass_ * delta0_2);
	
	return float_t(i < 1./3.*L_ && i > 1./6. * L_) * V0;
}

// === SIMULATION ===

void QLB::Qhat_X(int i, int j, QLB::cmat_t& Q) const
{
	// Precompute frequently used values
	const float_t m = 0.5 * mass_* dt_;
	const float_t g = 0.5 * V_(i,j) * dt_;
	const float_t omega = m*m - g*g;

	const complex_t a = (one - float_t(0.25)*omega) / 
	                    (one + float_t(0.25)*omega - img*g);
	const complex_t b = m / (one + float_t(0.25)*omega - img*g);

	// Qhat = X^(-1) * Q * X
	Q(0,0) =  a;
	Q(0,1) =  0;
	Q(0,2) =  0;
	Q(0,3) = -b*img;

	Q(1,0) =  0;
	Q(1,1) =  a;
	Q(1,2) =  b*img;
	Q(1,3) =  0;

	Q(2,0) =  0;
	Q(2,1) =  b*img;
	Q(2,2) =  a;
	Q(2,3) =  0;

	Q(3,0) = -b*img;
	Q(3,1) =  0;
	Q(3,2) =  0;
	Q(3,3) =  a;
}


void QLB::Qhat_Y(int i, int j, QLB::cmat_t& Q) const
{
	// Precompute frequently used values
	const float_t m = 0.5 * mass_* dt_;
	const float_t g = 0.5 *  V_(i,j) * dt_;
	const float_t omega = m*m - g*g;

	const complex_t a = (one - float_t(0.25)*omega) / 
	                    (one + float_t(0.25)*omega - img*g);
	const complex_t b = m / (one + float_t(0.25)*omega - img*g);

	// Qhat = Y^(-1) * Q * Y
	Q(0,0) =  a;
	Q(0,1) =  0;
	Q(0,2) =  0;
	Q(0,3) = -b*img;

	Q(1,0) =  0;
	Q(1,1) =  a;
	Q(1,2) = -b*img;
	Q(1,3) =  0;

	Q(2,0) =  0;
	Q(2,1) = -b*img;
	Q(2,2) =  a;
	Q(2,3) =  0;

	Q(3,0) = -b*img;
	Q(3,1) =  0;
	Q(3,2) =  0;
	Q(3,3) =  a;
}

// === SERIAL ===

void QLB::evolution_CPU_serial()
{
	const int L = L_;

	cmat_t Q(4);
	int ia, ja;
	int ik, jk;

	// Rotate with X^(-1)
	for(int i = 0; i < L; ++i)
	{
		for(int j=0; j < L; ++j)
		{

			for(int mk = 0; mk < 4; ++mk)
				spinorrot_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinorrot_(i,j,mk) += Xinv(mk,nk) * spinor_(i,j,nk);
		}
	}

	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			spinoraux_(i,j,0) = spinorrot_(i,j,0);
			spinoraux_(i,j,1) = spinorrot_(i,j,1);
			spinoraux_(i,j,2) = spinorrot_(i,j,2);
			spinoraux_(i,j,3) = spinorrot_(i,j,3);
		}
	}
	
	// collide & stream with Q_X 
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			ia = (i + 1) % L;
			ik = (i - 1 + L) % L;

			spinorrot_(ia,j,0) = 0;
			spinorrot_(ia,j,1) = 0;
			spinorrot_(ik,j,2) = 0;
			spinorrot_(ik,j,3) = 0;

			Qhat_X(i, j, Q);
	
			for(int nk = 0; nk < 4; ++nk)
			{
				spinorrot_(ia,j,0) += Q(0,nk) * spinoraux_(i,j,nk);
				spinorrot_(ia,j,1) += Q(1,nk) * spinoraux_(i,j,nk);
				spinorrot_(ik,j,2) += Q(2,nk) * spinoraux_(i,j,nk);
				spinorrot_(ik,j,3) += Q(3,nk) * spinoraux_(i,j,nk);
			}

		}
	}
	
	// Rotate back with X
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{

			for(int mk = 0; mk < 4; ++mk)
				spinor_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinor_(i,j,mk) += X(mk,nk)*spinorrot_(i,j,nk);
		}
	}

	// Rotate with Y^(-1)
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			for(int mk = 0; mk < 4; ++mk)
				spinorrot_(i,j,mk) = 0;

			for(int mk=0; mk < 4; ++mk)
				for(int nk=0; nk < 4; ++nk)
					spinorrot_(i,j,mk) += Yinv(mk,nk) * spinor_(i,j,nk);
		}
	}
	
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			spinoraux_(i,j,0) = spinorrot_(i,j,0);
			spinoraux_(i,j,1) = spinorrot_(i,j,1);
			spinoraux_(i,j,2) = spinorrot_(i,j,2);
			spinoraux_(i,j,3) = spinorrot_(i,j,3);
		}
	}
		
	// collide & stream with Q_Y
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{

			ja = (j + 1) % L;
			jk = (j - 1 + L) % L;

			spinorrot_(i,ja,0) = 0;
			spinorrot_(i,ja,1) = 0;
			spinorrot_(i,jk,2) = 0;
			spinorrot_(i,jk,3) = 0;

			Qhat_Y(i, j, Q);

			for(int nk = 0; nk < 4; ++nk)
			{
				spinorrot_(i,ja,0) += Q(0,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,ja,1) += Q(1,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,jk,2) += Q(2,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,jk,3) += Q(3,nk) * spinoraux_(i,j,nk);
			}
		}
	}
	
	// Rotate back with Y
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			for(int mk = 0; mk < 4; ++mk)
				spinor_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinor_(i,j,mk) += Y(mk,nk)*spinorrot_(i,j,nk);
		}
	}
	
	// Write to spread.dat (if requested)
	if( (opt_.plot() & QLBopt::spread) >> 1 || (opt_.plot() & QLBopt::all) )
		calculate_spread();
	
	// Update time;
	t_ += 1;
}

void QLB::calculate_spread()
{
	float_t deltax_nom = 0.0, deltax_den = 0.0;
	float_t deltay_nom = 0.0, deltay_den = 0.0;

	float_t x = 0.0, y = 0.0;
	const float_t dV = dx_*dx_;
	
	for(unsigned i = 0; i < L_; ++i)
		for(unsigned j = 0; j < L_; ++j)
		{
			x = dx_*(i-0.5*(L_-1));
			y = dx_*(j-0.5*(L_-1));

			// Delta X
			deltax_nom += x*x*std::norm(spinor_(i,j,0))*dV;
			deltax_den += std::norm(spinor_(i,j,0))*dV;

			// Delta Y
			deltay_nom += y*y*std::norm(spinor_(i,j,0))*dV;
			deltay_den += std::norm(spinor_(i,j,0))*dV;
		}

	// Update global variables
	deltax_[t_] = std::sqrt(deltax_nom/deltax_den);
	deltay_[t_] = std::sqrt(deltay_nom/deltay_den);
}

void QLB::calculate_macroscopic_vars()
{
	for(unsigned i = 0; i < L_; ++i)
	{
		for(unsigned j = 0; j < L_; ++j)
		{
			currentX_(i,j) = 0;
			currentY_(i,j) = 0;

			// Calculate current:
			// currentX = [s1, s2, s3, s4].H * beta *  alphaX * [s1, s2, s3, s4]  	
			for(int is = 0; is < 4; ++is)
				for(int js = 0; js < 4; ++js)
				{
					currentX_(i,j) += 
					  std::conj(spinor_(i,j,is))*beta(is,js)*alphaX(is,js)*spinor_(i,j,js);
					currentY_(i,j) +=
					  std::conj(spinor_(i,j,is))*beta(is,js)*alphaY(is,js)*spinor_(i,j,js);
				}
		
			// Calculate density
			rho_(i,j) = std::norm(spinor_(i,j,0)) + std::norm(spinor_(i,j,1)) + 
			            std::norm(spinor_(i,j,2)) + std::norm(spinor_(i,j,3));
			
			veloX_(i,j) = currentX_(i,j)/rho_(i,j);
			veloY_(i,j) = currentY_(i,j)/rho_(i,j);
		}
    }
}

// === MULTITHREAD ===

void QLB::evolution_CPU_thread(int tid)
{
	const int L_per_thread = L_ / opt_.nthreads();
	const int lower = tid*L_per_thread;
	const int upper = (tid + 1) != int(opt_.nthreads()) ? (tid+1)*L_per_thread : L_;
	const int L = L_;
	
	// Reset flag
	flag_ = 1;

	// thread private
	cmat_t Q(4);
	int ia, ja;
	int ik, jk;

	// Rotate with X^(-1)
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{

			for(int mk = 0; mk < 4; ++mk)
				spinorrot_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinorrot_(i,j,mk) += Xinv(mk,nk) * spinor_(i,j,nk);
		}
	}

	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			spinoraux_(i,j,0) = spinorrot_(i,j,0);
			spinoraux_(i,j,1) = spinorrot_(i,j,1);
			spinoraux_(i,j,2) = spinorrot_(i,j,2);
			spinoraux_(i,j,3) = spinorrot_(i,j,3);
		}
	}
	
	// Sync threads
	barrier.wait();

	// collide & stream with Q_X
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			ia = (i + 1) % L;
			ik = (i - 1 + L) % L;

			spinorrot_(ia,j,0) = 0;
			spinorrot_(ia,j,1) = 0;
			spinorrot_(ik,j,2) = 0;
			spinorrot_(ik,j,3) = 0;

			Qhat_X(i, j, Q);
	
			for(int nk = 0; nk < 4; ++nk)
			{
				spinorrot_(ia,j,0) += Q(0,nk) * spinoraux_(i,j,nk);
				spinorrot_(ia,j,1) += Q(1,nk) * spinoraux_(i,j,nk);
				spinorrot_(ik,j,2) += Q(2,nk) * spinoraux_(i,j,nk);
				spinorrot_(ik,j,3) += Q(3,nk) * spinoraux_(i,j,nk);
			}

		}
	}
	
	// Sync threads
	barrier.wait();

	// Rotate back with X
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{

			for(int mk = 0; mk < 4; ++mk)
				spinor_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinor_(i,j,mk) += X(mk,nk)*spinorrot_(i,j,nk);
		}
	}

	// Rotate with Y^(-1)
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			for(int mk = 0; mk < 4; ++mk)
				spinorrot_(i,j,mk) = 0;

			for(int mk=0; mk < 4; ++mk)
				for(int nk=0; nk < 4; ++nk)
					spinorrot_(i,j,mk) += Yinv(mk,nk) * spinor_(i,j,nk);
		}
	}
	
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			spinoraux_(i,j,0) = spinorrot_(i,j,0);
			spinoraux_(i,j,1) = spinorrot_(i,j,1);
			spinoraux_(i,j,2) = spinorrot_(i,j,2);
			spinoraux_(i,j,3) = spinorrot_(i,j,3);
		}
	}
		
	// collide & stream with Q_Y
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{

			ja = (j + 1) % L;
			jk = (j - 1 + L) % L;

			spinorrot_(i,ja,0) = 0;
			spinorrot_(i,ja,1) = 0;
			spinorrot_(i,jk,2) = 0;
			spinorrot_(i,jk,3) = 0;

			Qhat_Y(i, j, Q);

			for(int nk = 0; nk < 4; ++nk)
			{
				spinorrot_(i,ja,0) += Q(0,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,ja,1) += Q(1,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,jk,2) += Q(2,nk) * spinoraux_(i,j,nk);
				spinorrot_(i,jk,3) += Q(3,nk) * spinoraux_(i,j,nk);
			}
		}
	}
	
	// Rotate back with Y
	for(int i = lower; i < upper; ++i)
	{
		for(int j = 0; j < L; ++j)
		{
			for(int mk = 0; mk < 4; ++mk)
				spinor_(i,j,mk) = 0;

			for(int mk = 0; mk < 4; ++mk)
				for(int nk = 0; nk < 4; ++nk)
					spinor_(i,j,mk) += Y(mk,nk)*spinorrot_(i,j,nk);
		}
	}

	// Sync threads
	barrier.wait();

	// Update time & calculate spreads (only one thread executes this);
	if(std::atomic_fetch_sub(&flag_, 1) == 1)
	{
		if( (opt_.plot() & QLBopt::spread) >> 1 || (opt_.plot() & QLBopt::all) )
			calculate_spread();
		
		t_ += 1;
	}
}
