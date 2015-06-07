/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  This file contains all the CUDA kernels and function that make use of the 
 *  CUDA runtime API
 */

// Local includes
#include "QLB.hpp"

// ==== CONSTANTS ====

__constant__ unsigned int d_L;
__constant__ float d_dx;
__constant__ float d_dt;
__constant__ float d_mass;
__constant__ float d_g;
__constant__ unsigned int d_t;

__constant__ float d_scaling;
__constant__ int d_current_scene;

// ==== INITIALIZATION ====

void QLB::allocate_device_arrays()
{
	cuassert(cudaMalloc(&d_X, X.size() * sizeof(d_X[0])));
	cuassert(cudaMalloc(&d_Y, Y.size() * sizeof(d_Y[0])));
	cuassert(cudaMalloc(&d_Xinv, Xinv.size() * sizeof(d_Xinv[0])));
	cuassert(cudaMalloc(&d_Yinv, Yinv.size() * sizeof(d_Yinv[0])));
	cuassert(cudaMalloc(&d_alphaX, alphaX.size() * sizeof(d_alphaX[0])));
	cuassert(cudaMalloc(&d_alphaY, alphaY.size() * sizeof(d_alphaY[0])));
	cuassert(cudaMalloc(&d_beta, beta.size() * sizeof(d_beta[0])));

	cuassert(cudaMalloc(&d_spinor_, spinor_.size() * sizeof(d_spinor_[0])));
	cuassert(cudaMalloc(&d_spinoraux_, spinoraux_.size()*sizeof(d_spinoraux_[0])));
	cuassert(cudaMalloc(&d_spinorrot_, spinorrot_.size()*sizeof(d_spinorrot_[0])));
	cuassert(cudaMalloc(&d_V_, V_.size() * sizeof(d_V_[0])));
	
#ifdef QLB_CUDA_GL_WORKAROUND
	cuassert(cudaMalloc(&d_vertex_ptr_, array_vertex_.size() * sizeof(float)));
	cuassert(cudaMalloc(&d_normal_ptr_, array_normal_.size() * sizeof(float)));
#endif	
	
	cuassert(cudaDeviceSynchronize());
}	

void QLB::free_device_arrays()
{
	cuassert(cudaFree((void*) d_X));
	cuassert(cudaFree((void*) d_Y));
	cuassert(cudaFree((void*) d_Xinv));
	cuassert(cudaFree((void*) d_Yinv));
	cuassert(cudaFree((void*) d_alphaX));
	cuassert(cudaFree((void*) d_alphaY));
	cuassert(cudaFree((void*) d_beta));

	cuassert(cudaFree((void*) d_spinor_));
	cuassert(cudaFree((void*) d_spinoraux_));
	cuassert(cudaFree((void*) d_spinorrot_));
	cuassert(cudaFree((void*) d_V_));
	
#ifdef QLB_CUDA_GL_WORKAROUND
	cuassert(cudaFree((void*) d_vertex_ptr_));
	cuassert(cudaFree((void*) d_normal_ptr_));
#endif
}

/**
 *	Print version information
 *	@param  grid1     Grid dimensions for (L x L) kernels
 *	@param  grid4     Grid dimensions for (L x L x 4) kernels
 *	@param  block1    Block dimensions for (L x L) kernels
 *	@param  block4    Block dimensions for (L x L x 4) kernels
 */
static void print_version_information(dim3 grid1, dim3 grid4, dim3 block1, dim3 block4)
{
	std::cout << " === CUDA Info === " << std::endl;
	cudaDeviceProp deviceProp; 
	cuassert(cudaGetDeviceProperties(&deviceProp, 0));
	int dvVers = 0; cuassert(cudaDriverGetVersion(&dvVers));
	int rtVers = 0; cuassert(cudaRuntimeGetVersion(&rtVers));
	std::printf("CUDA Driver Version:          %d.%d\n", dvVers/1000, dvVers % 100);
	std::printf("CUDA Runtime Version:         %d.%d\n", rtVers/1000, rtVers % 100);
	std::printf("Total GPU memory:             %u bytes\n", 
	            unsigned(deviceProp.totalGlobalMem));
	std::printf("Multiprocessors on device:    %u\n", 
	            unsigned(deviceProp.multiProcessorCount));
	std::printf("Max threads per block:        %u\n", 
	            unsigned(deviceProp.maxThreadsPerBlock));
	std::printf("Max warp size:                %u\n", 
	            unsigned(deviceProp.warpSize));
	std::printf("Selected grid size  (1):      (%3u, %3u, %3u)\n", 
	            grid1.x, grid1.y, grid1.z);
	std::printf("Selected block size (1):      (%3u, %3u, %3u) = %u\n", 
	            block1.x, block1.y, block1.z, block1.x*block1.y*block1.z );
	std::printf("Selected grid size  (4):      (%3u, %3u, %3u)\n", 
	            grid4.x, grid4.y, grid4.z);
	std::printf("Selected block size (4):      (%3u, %3u, %3u) = %u\n\n", 
	            block4.x, block4.y, block4.z, block4.x*block4.y*block4.z );
}

/**
 *	Copy a matrix from host to device (if [value_t = cuFloatComplex] a specialized
 *	version will be used)
 *	@param  d_ptr   device pointer
 *	@param  m       matrix to be copied from
 */
template< class value_t, class mat_t >
static void copy_from_host_to_device(value_t* & d_ptr, const mat_t& m)
{
	std::vector<value_t> tmp(m.size());
	for(std::size_t i = 0; i < m.size(); ++i)
		tmp[i] = value_t(m[i]);
	
	cuassert(cudaMemcpy(d_ptr, tmp.data(), sizeof(tmp[0]) * tmp.size(), 
	                    cudaMemcpyHostToDevice));
}

template< class mat_t >
static void copy_from_host_to_device(cuFloatComplex* & d_ptr, const mat_t& m)
{
	std::vector<cuFloatComplex> tmp(m.size());
	for(std::size_t i = 0; i < m.size(); ++i)
		tmp[i] = make_cuFloatComplex(m[i]);
		
	cuassert(cudaMemcpy(d_ptr, tmp.data(), sizeof(tmp[0]) * tmp.size(), 
	                    cudaMemcpyHostToDevice));
}

/**
 *	Copy a matrix from the device to host (if [value_t = cuFloatComplex] a specialized
 *	version will be used)
 *	@param  d_ptr   device pointer
 *	@param  m       matrix to be copied to (of type QLB::float_t)
 */
template< class value_t, class mat_t >
static void copy_from_device_to_host(value_t* d_ptr, mat_t& m)
{
	std::vector<value_t> tmp(m.size());

	cuassert(cudaMemcpy(tmp.data(), d_ptr, sizeof(tmp[0]) * tmp.size(), 
	                    cudaMemcpyDeviceToHost));

	for(std::size_t i = 0; i < m.size(); ++i)
		m[i] = value_t(tmp[i]);
}

template< class mat_t >
static void copy_from_device_to_host(cuFloatComplex* d_ptr, mat_t& m)
{
	std::vector<cuFloatComplex> tmp(m.size());

	cuassert(cudaMemcpy(tmp.data(), d_ptr, sizeof(tmp[0]) * tmp.size(), 
	                    cudaMemcpyDeviceToHost));

	for(std::size_t i = 0; i < m.size(); ++i)
		m[i] = make_stdComplex<QLB::float_t>(tmp[i]);
}

void QLB::init_device()
{
	// initialize constant matrices
	copy_from_host_to_device(d_X, X);
	copy_from_host_to_device(d_Y, Y);
	copy_from_host_to_device(d_Xinv, Xinv);
	copy_from_host_to_device(d_Yinv, Yinv);
	copy_from_host_to_device(d_alphaX, alphaX);
	copy_from_host_to_device(d_alphaY, alphaY);
	copy_from_host_to_device(d_beta, beta);
	
	// initialize simulation matrices
	copy_from_host_to_device(d_spinor_, spinor_);
	copy_from_host_to_device(d_spinoraux_, spinoraux_);
	copy_from_host_to_device(d_spinorrot_, spinorrot_);
	copy_from_host_to_device(d_V_, V_);
	
	// initialize simulation variables
	cuassert(cudaMemcpyToSymbol(d_L, &L_, sizeof(L_)));
	float dx = static_cast<float>(dx_);
	cuassert(cudaMemcpyToSymbol(d_dx, &dx, sizeof(dx)));
	float dt = static_cast<float>(dt_);
	cuassert(cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt)));
	float mass = static_cast<float>(mass_);
	cuassert(cudaMemcpyToSymbol(d_mass, &mass, sizeof(mass)));
	float g = static_cast<float>(g_);
	cuassert(cudaMemcpyToSymbol(d_g, &g, sizeof(g)));
	
	int current_scene = current_scene_;
	cuassert(cudaMemcpyToSymbol(d_current_scene, &current_scene, sizeof(current_scene)));
	float scaling = static_cast<float>(scaling_);
	cuassert(cudaMemcpyToSymbol(d_scaling, &scaling, sizeof(scaling)));
	cuassert(cudaMemcpyToSymbol(d_t, &t_, sizeof(t_)));

	if(opt_.verbose())
		print_version_information(grid1_, grid4_, block1_, block4_);
	
	cuassert(cudaDeviceSynchronize());
}

void QLB::get_device_arrays()
{
	copy_from_device_to_host(d_spinor_, spinor_);
	copy_from_device_to_host(d_spinorrot_, spinorrot_);
	copy_from_device_to_host(d_spinoraux_, spinoraux_);
}

void QLB::update_device_constants()
{
	float scaling = static_cast<float>(scaling_);
	cuassert(cudaMemcpyToSymbol(d_scaling, &scaling, sizeof(scaling)));

	int current_scene = current_scene_;
	cuassert(cudaMemcpyToSymbol(d_current_scene, &current_scene, sizeof(current_scene)));
}

// =============================== SIMULATION ==================================

#define at(i,j,k) 4*(d_L*(i) + (j)) + (k)

/** 
 *	Rotate the spinors and store the result in spinorrot and spinoraux
 *	@param spinor  device pointer spinors
 *	@param spinor  device pointer spinorrot
 *	@param spinor  device pointer spinoraux
 *	@param Rinv    inverse rotation matrix (4 x 4)
 */
__global__ void kernel_rotate(const cuFloatComplex* __restrict spinor, 
                              cuFloatComplex* spinorrot,
                              cuFloatComplex* spinoraux,
                              const cuFloatComplex* __restrict Rinv)
{
	__shared__ cuFloatComplex Rinv_loc[16];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	int lj = threadIdx.y;
	int lk = threadIdx.z;
	
	if(lj < 4 && lk < 4)
		Rinv_loc[lj*4 + lk] = Rinv[lj*4 + lk];
	
	__syncthreads();
			
	if(i < d_L && j < d_L)
	{
		const int ij  = at(i,j,0);
		const int ijk = ij + k;
		
		spinorrot[ijk] = Rinv_loc[4*k + 0] * spinor[ij + 0] + 
		                 Rinv_loc[4*k + 1] * spinor[ij + 1] +		
		                 Rinv_loc[4*k + 2] * spinor[ij + 2] +
		                 Rinv_loc[4*k + 3] * spinor[ij + 3];
				
		spinoraux[ijk] = spinorrot[ijk];
	}
}

/**
 *	Rotate spinorrot back and store the result in spinor
 *	@param spinor  device pointer spinor
 *	@param spinor  device pointer spinorrot
 *	@param R       rotation matrix (4 x 4)
 */
__global__ void kernel_rotate_back(cuFloatComplex* spinor, 
                                   const cuFloatComplex* __restrict spinorrot,
                                   const cuFloatComplex* __restrict R)
{
	__shared__ cuFloatComplex R_loc[16];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	int lj = threadIdx.y;
	int lk = threadIdx.z;
	
	if(lj < 4 && lk < 4)
		R_loc[lj*4 + lk] = R[lj*4 + lk];
	
	__syncthreads();	
	
	if(i < d_L && j < d_L)
	{
		const int ij  = at(i,j,0);
	
		spinor[ij+k] = R_loc[4*k + 0] * spinorrot[ij + 0] + 
		               R_loc[4*k + 1] * spinorrot[ij + 1] +		
		               R_loc[4*k + 2] * spinorrot[ij + 2] +
		               R_loc[4*k + 3] * spinorrot[ij + 3];
	}
} 

/** 
 *	Collide and stream with matrix Q_X
 *	@param spinorrot  device pointer spinorrot
 *	@param spinoraux  device pointer spinoraux
 *	@param V          device pointer potential
 */
__global__ void kernel_collide_Q_X(cuFloatComplex* spinorrot,
							       const cuFloatComplex* __restrict spinoraux,
                                   const float* __restrict V)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ia = (i + 1) % d_L;
		int ik = (i - 1 + d_L) % d_L;
		
		// Precomputed indices
		const int i_j  = at(i, j,0);
		const int ia_j = at(ia,j,0);
		const int ik_j = at(ik,j,0);
		
		const float m = 0.5f * d_mass* d_dt;
		const float g = 0.5f *  V[i*d_L +j] * d_dt;
		const float omega = m*m - g*g;

		const cuFloatComplex a_nom = make_cuFloatComplex(1.0f - 0.25f*omega, 0.0f);
		const cuFloatComplex a_den = make_cuFloatComplex(1.0f + 0.25f*omega, -1.0f*g);
		const cuFloatComplex b_nom = make_cuFloatComplex(m, 0.0f);

		const cuFloatComplex a      = a_nom / a_den;
		const cuFloatComplex b      = b_nom / a_den;
		const cuFloatComplex neg_bi = b * make_cuFloatComplex( 0.0f, -1.0f);
		const cuFloatComplex pos_bi = b * make_cuFloatComplex( 0.0f,  1.0f);

		// The matrix multiplication has been unrolled and 0 entries in Qhat 
		// have been skipped  
		// Qhat = X^(-1) * Q * X
		spinorrot[ia_j + 0] = a * spinoraux[i_j+0] + neg_bi * spinoraux[i_j+3];
		spinorrot[ia_j + 1] = a * spinoraux[i_j+1] + pos_bi * spinoraux[i_j+2]; 
		spinorrot[ik_j + 2] = pos_bi * spinoraux[i_j+1] + a * spinoraux[i_j+2];
		spinorrot[ik_j + 3] = neg_bi * spinoraux[i_j+0] + a * spinoraux[i_j+3];
	}
}

/** 
 *	Collide and stream with matrix Q_Y
 *	@param spinorrot  device pointer spinorrot
 *	@param spinoraux  device pointer spinoraux
 *	@param V          device pointer potential
 */
__global__ void kernel_collide_Q_Y(cuFloatComplex* spinorrot,
								   const cuFloatComplex* __restrict spinoraux,
                                   const float* __restrict V)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ja = (j + 1) % d_L;
		int jk = (j - 1 + d_L) % d_L;
		
		// Precomputed indices
		const int i_j  = at(i,j ,0);
		const int i_ja = at(i,ja,0);
		const int i_jk = at(i,jk,0);

		const float m = 0.5f * d_mass* d_dt;
		const float g = 0.5f *  V[i*d_L +j] * d_dt;
		const float omega = m*m - g*g;
		
		const cuFloatComplex a_nom = make_cuFloatComplex(1.0f - 0.25f*omega, 0.0f);
		const cuFloatComplex b_nom = make_cuFloatComplex(m, 0.0f);
		const cuFloatComplex a_den = make_cuFloatComplex(1.0f + 0.25f*omega, -1.0f*g);

		const cuFloatComplex a      = a_nom / a_den;
		const cuFloatComplex b      = b_nom / a_den;
		const cuFloatComplex neg_bi = b * make_cuFloatComplex( 0.0f, -1.0f);

		// The matrix multiplication has been unrolled and 0 entries in Qhat 
		// have been skipped  
		// Qhat = Y^(-1) * Q * Y
		spinorrot[i_ja + 0] = a * spinoraux[i_j+0] + neg_bi * spinoraux[i_j+3];
		spinorrot[i_ja + 1] = a * spinoraux[i_j+1] + neg_bi * spinoraux[i_j+2]; 
		spinorrot[i_jk + 2] = neg_bi * spinoraux[i_j+1] + a * spinoraux[i_j+2];
		spinorrot[i_jk + 3] = neg_bi * spinoraux[i_j+0] + a * spinoraux[i_j+3];
	}
}

/**
 *	Updat the potential array using the GP potential
 *	@param V         device pointer potential array
 *	@param spinor    device pointer spinor
 */
__global__ void kernel_set_potential(float* V, 
                                     const cuFloatComplex* __restrict spinor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	V[d_L*i + j] = d_g * ( 
			       cuCnormf(spinor[at(i,j,0)]) + cuCnormf(spinor[at(i,j,1)]) +
			       cuCnormf(spinor[at(i,j,2)]) + cuCnormf(spinor[at(i,j,3)]) );

} 

void QLB::evolution_GPU()
{
	// Update potential array if needed
	if(V_indx_ == 3) kernel_set_potential<<< grid1_, block1_ >>>(d_V_, d_spinor_);

	// Rotate with X^(-1)
	kernel_rotate<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_spinoraux_, d_Xinv);
	CUDA_CHECK_KERNEL
	
	// Collide & stream with Q_X 
	kernel_collide_Q_X<<< grid1_, block1_ >>>(d_spinorrot_, d_spinoraux_, d_V_);
	CUDA_CHECK_KERNEL
	
	// Rotate back with X 
	kernel_rotate_back<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_X);
	CUDA_CHECK_KERNEL
	
	// Rotate with Y^(-1) 
	kernel_rotate<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_spinoraux_, d_Yinv);
	CUDA_CHECK_KERNEL
	
	// Collide & stream with Q_Y 
	kernel_collide_Q_Y<<< grid1_, block1_ >>>(d_spinorrot_, d_spinoraux_, d_V_);
	CUDA_CHECK_KERNEL

	// Rotate back with Y
	kernel_rotate_back<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_Y);
	CUDA_CHECK_KERNEL
	
	cudaDeviceSynchronize();
	
	// Calculate the spreads
	if( (opt_.plot() & QLBopt::spread) >> 1 || (opt_.plot() & QLBopt::all) )
	{
		copy_from_device_to_host(d_spinor_, spinor_);
		cudaDeviceSynchronize();
		calculate_spread();
	}

	// Update time;
	t_ += 1;
}


// ================================= GRAPHICS ==================================

// Unrolled loop for the current
#define CURRENT_UNROLLED_LOOP(i,j) (CURRENT((i),(j),0)+CURRENT((i),(j),1)+\
                                    CURRENT((i),(j),2)+CURRENT((i),(j),3))

#define CURRENT(i,j,is) (CURRENT_1((i),(j),(is),0)+CURRENT_1((i),(j),(is),1)+\
                         CURRENT_1((i),(j),(is),2)+CURRENT_1((i),(j),(is),3))
                    
#define CURRENT_1(i,j,is,js) (cuCrealf(cuConjf(\
 d_ptr[at((i),(j),(is))])*alpha[(is)*4 + (js)]*d_ptr[at((i),(j),(js))]))


/** 
 *	Calculate the vertices (spinors,density or current) and copy them to the 
 *	vertex VBO.
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 *	@param alpha     pointer to the alpha matrix (unused if we don't calculate
 *	                 the current)
 *	@param alpha     pointer to the beta matrix (unused if we don't calculate
 *	                 the current)
 */
__global__ void kernel_calculate_vertex_scene(float3* vbo_ptr, 
                                              const cuFloatComplex* __restrict d_ptr,
                                              const cuFloatComplex* __restrict alpha,
                                              const cuFloatComplex* __restrict beta) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(i < d_L && j < d_L)
	{
		// Select the right scene (all warps always do the same)
		if(d_current_scene < 4)
		{
			int k = d_current_scene;
			vbo_ptr[d_L*i + j].y = d_scaling * cuCnormf( d_ptr[at(i,j,k)] );
		}
		else if(d_current_scene == 4)
		{
			vbo_ptr[d_L*i + j].y = d_scaling * ( 
			        cuCnormf( d_ptr[at(i,j,0)]) + cuCnormf( d_ptr[at(i,j,1)]) +
			        cuCnormf( d_ptr[at(i,j,2)]) + cuCnormf( d_ptr[at(i,j,3)]) );
		}
		else
		{
			vbo_ptr[d_L*i + j].y = d_scaling*( CURRENT_UNROLLED_LOOP(i,j) );
		}
	}
}

#define y(i,j)  3*((i)*L_ + (j)) + 1

void QLB::calculate_vertex_cuda()
{

#ifndef QLB_CUDA_GL_WORKAROUND
	vbo_vertex.map();

	float3* vbo_ptr = vbo_vertex.get_device_pointer();

	if(current_scene_ < 5)
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, NULL, NULL);
	else if(current_scene_ == 5)
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, d_alphaX, d_beta);
	else
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, d_alphaY, d_beta);
	CUDA_CHECK_KERNEL

	vbo_vertex.unmap();
#else

	if(current_scene_ < 5)
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(d_vertex_ptr_, d_spinor_, NULL, NULL);
	else if(current_scene_ == 5)
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(d_vertex_ptr_, d_spinor_, d_alphaX, d_beta);
	else
		kernel_calculate_vertex_scene<<< grid1_, block1_ >>>(d_vertex_ptr_, d_spinor_, d_alphaY, d_beta);
	CUDA_CHECK_KERNEL

	cuassert(cudaMemcpy(array_vertex_.data(), d_vertex_ptr_, 
		                sizeof(float) * array_vertex_.size(), 
	                    cudaMemcpyDeviceToHost));
			
	// Copy vertex array to vertex VBO
	vbo_vertex.bind();
	vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
		                     &array_vertex_[0]);
	vbo_vertex.unbind();
#endif
}

#undef y

/** 
 *	Calculate the vertices (by taking the abs of the potential V) and copy them 
 *	to the vertex VBO
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 */
__global__ void kernel_calculate_vertex_V(float3* vbo_ptr, float* d_ptr) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(i < d_L && j < d_L)
		vbo_ptr[d_L*i + j].y = d_scaling * fabsf( d_ptr[i*d_L +j] ) - 0.005f*d_L;
}


#define y(i,j)  3*((i)*L_ + (j)) + 1

void QLB::calculate_vertex_V_cuda()
{
#ifndef QLB_CUDA_GL_WORKAROUND
	vbo_vertex.map();
	
	float3* vbo_ptr = vbo_vertex.get_device_pointer();

	kernel_calculate_vertex_V<<< grid1_, block1_ >>>(vbo_ptr, d_V_);
	CUDA_CHECK_KERNEL
	
	vbo_vertex.unmap();
#else
	kernel_calculate_vertex_V<<< grid1_, block1_ >>>(d_vertex_ptr_, d_V_);
	CUDA_CHECK_KERNEL
	
	cuassert(cudaMemcpy(array_vertex_.data(), d_vertex_ptr_, 
		                sizeof(float) * array_vertex_.size(), 
	                    cudaMemcpyDeviceToHost));
			
	// Copy vertex array to vertex VBO
	vbo_vertex.bind();
	vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
		                     &array_vertex_[0]);
	vbo_vertex.unbind();
#endif
}

#undef y

/** 
 *  Calculate the normals of the spinors and copy them to the normal VBO
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 *	@param alpha     pointer to the alpha matrix (unused if we don't calculate
 *	                 the current)
 */
__global__ void kernel_calculate_normal_scene(float3* vbo_ptr, 
                                              const cuFloatComplex* __restrict d_ptr,
                                              const cuFloatComplex* __restrict alpha,
                                              const cuFloatComplex* __restrict beta)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ik = (i + 1) % d_L;
		int jk = (d_L - 1 + j) % d_L;
		
		float vertex_i_j, vertex_ik_j, vertex_i_jk;
		
		// Select the right scene (all warps always do the same)
		if(d_current_scene < 4)
		{
			int k = d_current_scene;
			vertex_i_j  = cuCnormf( d_ptr[at(i ,j ,k)] );
			vertex_ik_j = cuCnormf( d_ptr[at(ik,j ,k)] );
			vertex_i_jk = cuCnormf( d_ptr[at(i ,jk,k)] );
		}
		else if(d_current_scene == 4)
		{
			vertex_i_j  = cuCnormf(d_ptr[at(i,j,0)]) + cuCnormf(d_ptr[at(i,j,1)]) +
		                  cuCnormf(d_ptr[at(i,j,2)]) + cuCnormf(d_ptr[at(i,j,3)]);
		
			vertex_ik_j = cuCnormf(d_ptr[at(ik,j,0)]) + cuCnormf(d_ptr[at(ik,j,1)]) +
		                  cuCnormf(d_ptr[at(ik,j,2)]) + cuCnormf(d_ptr[at(ik,j,3)]);
		
			vertex_i_jk = cuCnormf(d_ptr[at(i,jk,0)]) + cuCnormf(d_ptr[at(i,jk,1)]) +
		                  cuCnormf(d_ptr[at(i,jk,2)]) + cuCnormf(d_ptr[at(i,jk,3)]);
		}
		else
		{
			vertex_i_j  = CURRENT_UNROLLED_LOOP(i ,j);
			vertex_ik_j = CURRENT_UNROLLED_LOOP(ik ,j);
			vertex_i_jk = CURRENT_UNROLLED_LOOP(i ,jk);
		}
		
		// x		
		float x2 =  d_scaling * vertex_i_j;
		
		// a
		float a1 =  d_dx;
		float a2 =  d_scaling * vertex_ik_j - x2;
	
		// b
		float b2 =  d_scaling * vertex_i_jk - x2;
		float b3 = -d_dx;
		
		// n = a x b
		float3 n;
		n.x =  a2*b3;
		n.y = -a1*b3;
		n.z =  a1*b2;
		
		// normalize
		float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
		
		vbo_ptr[d_L*i + j].x = n.x/norm;
		vbo_ptr[d_L*i + j].y = n.y/norm;
		vbo_ptr[d_L*i + j].z = n.z/norm;
	}
}


void QLB::calculate_normal_cuda()
{
#ifndef QLB_CUDA_GL_WORKAROUND
	vbo_normal.map();
	
	float3* vbo_ptr = vbo_normal.get_device_pointer();

	if(current_scene_ < 5)
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, NULL, NULL);
	else if(current_scene_ == 5)
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, d_alphaX, d_beta);
	else
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_, d_alphaY, d_beta);
	CUDA_CHECK_KERNEL

	vbo_normal.unmap();

#else
	if(current_scene_ < 5)
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(d_normal_ptr_, d_spinor_, NULL, NULL);
	else if(current_scene_ == 5)
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(d_normal_ptr_, d_spinor_, d_alphaX, d_beta);
	else
		kernel_calculate_normal_scene<<< grid1_, block1_ >>>(d_normal_ptr_, d_spinor_, d_alphaY, d_beta);
	CUDA_CHECK_KERNEL

	cuassert(cudaMemcpy(array_normal_.data(), d_normal_ptr_, 
		                sizeof(float) * array_normal_.size(), 
	                    cudaMemcpyDeviceToHost));

	vbo_normal.bind();
	vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float),
		                     &array_normal_[0]);
	vbo_normal.unbind();
#endif
}

/** 
 *  Calculate the normals of the potential V and copy them to the normal VBO
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 */
__global__ void kernel_calculate_normal_V(float3* vbo_ptr, float* d_ptr)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ik = (i + 1) % d_L;
		int jk = (d_L - 1 + j) % d_L;
		
		// x		
		float x2 =  d_scaling * fabsf( d_ptr[i*d_L +j] );
		
		// a
		float a1 =  d_dx;
		float a2 =  d_scaling * fabsf( d_ptr[ik*d_L +j] ) - x2;
	
		// b
		float b2 =  d_scaling * fabsf( d_ptr[i*d_L +jk] ) - x2;
		float b3 = -d_dx;
		
		// n = a x b
		float3 n;
		n.x =  a2*b3;
		n.y = -a1*b3;
		n.z =  a1*b2;
		
		// normalize
		float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
		
		vbo_ptr[d_L*i + j].x = n.x/norm;
		vbo_ptr[d_L*i + j].y = n.y/norm;
		vbo_ptr[d_L*i + j].z = n.z/norm;
	}
}

void QLB::calculate_normal_V_cuda()
{
#ifndef QLB_CUDA_GL_WORKAROUND
	vbo_normal.map();
	
	float3* vbo_ptr = vbo_normal.get_device_pointer();

	kernel_calculate_normal_V<<< grid1_, block1_ >>>(vbo_ptr, d_V_);
	CUDA_CHECK_KERNEL

	vbo_normal.unmap();
	
#else
	kernel_calculate_normal_V<<< grid1_, block1_ >>>(d_normal_ptr_, d_V_);
	CUDA_CHECK_KERNEL
	
	cuassert(cudaMemcpy(array_normal_.data(), d_normal_ptr_, 
		                sizeof(float) * array_normal_.size(), 
	                    cudaMemcpyDeviceToHost));
			
	vbo_normal.bind();
	vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float), 
		                     &array_normal_[0]);
	vbo_normal.unbind();
#endif
}
