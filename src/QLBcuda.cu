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
	
	int current_scene = current_scene_;
	cuassert(cudaMemcpyToSymbol(d_current_scene, &current_scene, sizeof(current_scene)));
	float scaling = static_cast<float>(scaling_);
	cuassert(cudaMemcpyToSymbol(d_scaling, &scaling, sizeof(scaling)));
	
	cudaDeviceProp deviceProp; 
	cuassert(cudaGetDeviceProperties(&deviceProp, 0));
	
	// Experiments showed that the minimal number of threads per block, which is
	// the warps size yield best performance in this case	

	block4_.x = 1;
	block4_.y = deviceProp.warpSize / 4;
	block4_.z = 4;
	
	grid4_  = dim3( L_ % block4_.x == 0 ? L_ / block4_.x : L_ / block4_.x + 1,
	                L_ % block4_.x == 0 ? L_ / block4_.y : L_ / block4_.y + 1, 1);

	block1_.x = 1;
	block1_.y = deviceProp.warpSize;
	block1_.z = 1;
	
	grid1_  = dim3( L_ % block1_.x == 0 ? L_ / block1_.x : L_ / block1_.x + 1,
	                L_ % block1_.x == 0 ? L_ / block1_.y : L_ / block1_.y + 1, 1);

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
 *	@param M       rotation matrix
 */
__global__ void kernel_rotate(cuFloatComplex* spinor, 
                              cuFloatComplex* spinorrot,
                              cuFloatComplex* spinoraux,
                              cuFloatComplex* M)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(i < d_L && j < d_L)
	{
		spinorrot[at(i,j,k)] = make_cuFloatComplex(0.0f, 0.0f);
		
		spinorrot[at(i,j,k)] = spinorrot[at(i,j,k)] + M[4*k + 0] * spinor[at(i,j,0)];
		spinorrot[at(i,j,k)] = spinorrot[at(i,j,k)] + M[4*k + 1] * spinor[at(i,j,1)];		
		spinorrot[at(i,j,k)] = spinorrot[at(i,j,k)] + M[4*k + 2] * spinor[at(i,j,2)];
		spinorrot[at(i,j,k)] = spinorrot[at(i,j,k)] + M[4*k + 3] * spinor[at(i,j,3)];
				
		spinoraux[at(i,j,k)] = spinorrot[at(i,j,k)];
	}

}

/** 
 *	Rotate spinorrot back and store the result in spinor
 *	@param spinor  device pointer spinor
 *	@param spinor  device pointer spinorrot
 *	@param Minv    inverse rotation matrix
 */
__global__ void kernel_rotate_back(cuFloatComplex* spinor, 
                                   cuFloatComplex* spinorrot,
                                   cuFloatComplex* Minv)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(i < d_L && j < d_L)
	{
		spinor[at(i,j,k)] = make_cuFloatComplex(0.0f, 0.0f);
		
		spinor[at(i,j,k)] = spinor[at(i,j,k)] + Minv[4*k + 0] * spinorrot[at(i,j,0)];
		spinor[at(i,j,k)] = spinor[at(i,j,k)] + Minv[4*k + 1] * spinorrot[at(i,j,1)];		
		spinor[at(i,j,k)] = spinor[at(i,j,k)] + Minv[4*k + 2] * spinorrot[at(i,j,2)];
		spinor[at(i,j,k)] = spinor[at(i,j,k)] + Minv[4*k + 3] * spinorrot[at(i,j,3)];
	}
} 

/** 
 *	Collide and stream with matrix Q_X
 *	@param spinor  device pointer spinorrot
 *	@param spinor  device pointer spinoraux
 *	@param V       device pointer potential
 */
__global__ void kernel_collide_X(cuFloatComplex* spinorrot,
								 cuFloatComplex* spinoraux,
                                 float* V)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ia = (i + 1) % d_L;
		int ik = (i - 1 + d_L) % d_L;

		spinorrot[at(ia,j,0)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(ia,j,1)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(ik,j,2)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(ik,j,3)] = make_cuFloatComplex(0.0f, 0.0f);

		float m = 0.5f * d_mass* d_dt;
		float g = 0.5f *  V[i*d_L +j] * d_dt;
		float omega = m*m - g*g;

		cuFloatComplex img = make_cuFloatComplex(0.0f, 1.0f);
		
		cuFloatComplex a_nom = make_cuFloatComplex(1.0f - 0.25f*omega, 0.0f);
		cuFloatComplex a_den = make_cuFloatComplex(1.0f + 0.25f*omega, -1.0f*g);
		cuFloatComplex b_nom = make_cuFloatComplex(m, 0.0f);

		cuFloatComplex a = a_nom / a_den;
		cuFloatComplex b = b_nom / a_den;

		cuFloatComplex Q_00 =  a;
		cuFloatComplex Q_03 =  b * img;
 
		cuFloatComplex Q_11 =  a;
		cuFloatComplex Q_12 =  make_cuFloatComplex( 0.0f,  2.0f) * b;
		cuFloatComplex Q_13 =  b * img;

		cuFloatComplex Q_20 =  make_cuFloatComplex( 0.0f, -0.5f) * b;
		cuFloatComplex Q_21 =  make_cuFloatComplex( 0.0f,  0.5f) * b;
		cuFloatComplex Q_22 =  a;

		cuFloatComplex Q_30 =  b * img;
		cuFloatComplex Q_33 =  a;

		spinorrot[at(ia,j,0)] = spinorrot[at(ia,j,0)] + Q_00 * spinoraux[at(i,j,0)];
		spinorrot[at(ik,j,2)] = spinorrot[at(ik,j,2)] + Q_20 * spinoraux[at(i,j,0)];
		spinorrot[at(ik,j,3)] = spinorrot[at(ik,j,3)] + Q_30 * spinoraux[at(i,j,0)];
		
		spinorrot[at(ia,j,1)] = spinorrot[at(ia,j,1)] + Q_11 * spinoraux[at(i,j,1)];
		spinorrot[at(ik,j,2)] = spinorrot[at(ik,j,2)] + Q_21 * spinoraux[at(i,j,1)];
		
		spinorrot[at(ia,j,1)] = spinorrot[at(ia,j,1)] + Q_12 * spinoraux[at(i,j,2)];
		spinorrot[at(ik,j,2)] = spinorrot[at(ik,j,2)] + Q_22 * spinoraux[at(i,j,2)];
		
		spinorrot[at(ia,j,0)] = spinorrot[at(ia,j,0)] + Q_03 * spinoraux[at(i,j,3)];
		spinorrot[at(ia,j,1)] = spinorrot[at(ia,j,1)] + Q_13 * spinoraux[at(i,j,3)];
		spinorrot[at(ik,j,3)] = spinorrot[at(ik,j,3)] + Q_33 * spinoraux[at(i,j,3)];
	}
}

/** 
 *	Collide and stream with matrix Q_Y
 *	@param spinor  device pointer spinorrot
 *	@param spinor  device pointer spinoraux
 *	@param V       device pointer potential
 */
__global__ void kernel_collide_Y(cuFloatComplex* spinorrot,
								 cuFloatComplex* spinoraux,
                                 float* V)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < d_L && j < d_L)
	{
		int ja = (j + 1) % d_L;
		int jk = (j - 1 + d_L) % d_L;

		spinorrot[at(i,ja,0)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(i,ja,1)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(i,jk,2)] = make_cuFloatComplex(0.0f, 0.0f);
		spinorrot[at(i,jk,3)] = make_cuFloatComplex(0.0f, 0.0f);

		float m = 0.5f * d_mass* d_dt;
		float g = 0.5f *  V[i*d_L +j] * d_dt;
		float omega = m*m - g*g;

		cuFloatComplex img = make_cuFloatComplex(0.0f, 1.0f);
		
		cuFloatComplex a_nom = make_cuFloatComplex(1.0f - 0.25f*omega, 0.0f);
		cuFloatComplex b_nom = make_cuFloatComplex(m, 0.0f);
		cuFloatComplex a_den = make_cuFloatComplex(1.0f + 0.25f*omega, -1.0f*g);

		cuFloatComplex a = a_nom / a_den;
		cuFloatComplex b = b_nom / a_den;

		cuFloatComplex Q_00 =  a;
		cuFloatComplex Q_03 =  b*img;

		cuFloatComplex Q_11 =  a;
		cuFloatComplex Q_12 =  make_cuFloatComplex( 0.0f,  2.0f) * b;
		cuFloatComplex Q_13 =  b * img;

		cuFloatComplex Q_20 =  make_cuFloatComplex( 0.0f, -0.5f) * b;
		cuFloatComplex Q_21 =  make_cuFloatComplex( 0.0f,  0.5f) * b;
		cuFloatComplex Q_22 =  a;

		cuFloatComplex Q_30 =  b*img;
		cuFloatComplex Q_33 =  a;

		spinorrot[at(i,ja,0)] = spinorrot[at(i,ja,0)] + Q_00 * spinoraux[at(i,j,0)];
		spinorrot[at(i,jk,2)] = spinorrot[at(i,jk,2)] + Q_20 * spinoraux[at(i,j,0)];
		spinorrot[at(i,jk,3)] = spinorrot[at(i,jk,3)] + Q_30 * spinoraux[at(i,j,0)];
		
		spinorrot[at(i,ja,1)] = spinorrot[at(i,ja,1)] + Q_11 * spinoraux[at(i,j,1)];
		spinorrot[at(i,jk,2)] = spinorrot[at(i,jk,2)] + Q_21 * spinoraux[at(i,j,1)];
		
		spinorrot[at(i,ja,1)] = spinorrot[at(i,ja,1)] + Q_12 * spinoraux[at(i,j,2)];
		spinorrot[at(i,jk,2)] = spinorrot[at(i,jk,2)] + Q_22 * spinoraux[at(i,j,2)];
		
		spinorrot[at(i,ja,0)] = spinorrot[at(i,ja,0)] + Q_03 * spinoraux[at(i,j,3)];
		spinorrot[at(i,ja,1)] = spinorrot[at(i,ja,1)] + Q_13 * spinoraux[at(i,j,3)];
		spinorrot[at(i,jk,3)] = spinorrot[at(i,jk,3)] + Q_33 * spinoraux[at(i,j,3)];
	}
}

void QLB::evolution_GPU()
{
	// Rotate with X
	kernel_rotate<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_spinoraux_, d_X);
	CUDA_CHECK_KERNEL
	
	// Collide & stream with Q_X 
	cudaDeviceSynchronize();
	kernel_collide_X<<< grid1_, block1_ >>>(d_spinorrot_, d_spinoraux_, d_V_);
	CUDA_CHECK_KERNEL
	
	// Rotate back with Xinv 
	cudaDeviceSynchronize();
	kernel_rotate_back<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_Xinv);
	CUDA_CHECK_KERNEL
	
	// Rotate with X 
	cudaDeviceSynchronize();
	kernel_rotate<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_spinoraux_, d_Y);
	CUDA_CHECK_KERNEL
	
	// Collide & stream with Q_Y 
	cudaDeviceSynchronize();
	kernel_collide_Y<<< grid1_, block1_ >>>(d_spinorrot_, d_spinoraux_, d_V_);
	CUDA_CHECK_KERNEL

	// Rotate back with Yinv
	cudaDeviceSynchronize();
	kernel_rotate_back<<< grid4_, block4_ >>>(d_spinor_, d_spinorrot_, d_Yinv);
	CUDA_CHECK_KERNEL
	
	cudaDeviceSynchronize();
	
	// Calculate the spreads and store them in a buffer as it would be too expensive
	// to always copy them to the CPU
	if( (opt_.plot() & QLBopt::spread) >> 1 || (opt_.plot() & QLBopt::all) )
		calculate_spread_cuda();
	
	// Update time;
	t_ += 1.0;
}


// ================================= GRAPHICS ==================================

/** 
 *	Calculate the vertices (by taking the norm of the spinors) and copy them to 
 *	the	vertex VBO
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 */
__global__ void kernel_calculate_vertex_spinor(float3* vbo_ptr, cuFloatComplex* d_ptr) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = d_current_scene;
	
	if(i < d_L && j < d_L)
		vbo_ptr[d_L*i + j].y = d_scaling * cuCnormf( d_ptr[at(i,j,k)] );
}

void QLB::calculate_vertex_spinor_cuda()
{
	vbo_vertex.map();
	
	float3* vbo_ptr = vbo_vertex.get_device_pointer();

	kernel_calculate_vertex_spinor<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_);
	CUDA_CHECK_KERNEL
	
	vbo_vertex.unmap();
}

/** 
 *	Calculate the vertices (by taking the norm of the potential V) and copy them 
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

void QLB::calculate_vertex_V_cuda()
{
	vbo_vertex.map();
	
	float3* vbo_ptr = vbo_vertex.get_device_pointer();

	kernel_calculate_vertex_V<<< grid1_, block1_ >>>(vbo_ptr, d_V_);
	CUDA_CHECK_KERNEL
	
	vbo_vertex.unmap();
}

/** 
 *  Calculate the normals of the spinors and copy them to the normal VBO
 *	@param vbo_ptr   pointer to the VBO
 *	@param d_ptr     pointer to the spinors
 */
__global__ void kernel_calculate_normal_spinor(float3* vbo_ptr, cuFloatComplex* d_ptr)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = d_current_scene;

	if(i < d_L && j < d_L)
	{
		int ik = (i + 1) % d_L;
		int jk = (d_L - 1 + j) % d_L;
		
		// x		
		float x2 =  d_scaling * cuCnormf( d_ptr[at(i,j,k)] );
		
		// a
		float a1 =  d_dx;
		float a2 =  d_scaling * cuCnormf( d_ptr[at(ik,j,k)] ) - x2;
	
		// b
		float b2 =  d_scaling * cuCnormf( d_ptr[at(i,jk,k)] ) - x2;
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


void QLB::calculate_normal_spinor_cuda()
{
	vbo_normal.map();
	
	float3* vbo_ptr = vbo_normal.get_device_pointer();

	kernel_calculate_normal_spinor<<< grid1_, block1_ >>>(vbo_ptr, d_spinor_);
	CUDA_CHECK_KERNEL

	vbo_normal.unmap();
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
	vbo_normal.map();
	
	float3* vbo_ptr = vbo_normal.get_device_pointer();

	kernel_calculate_normal_V<<< grid1_, block1_ >>>(vbo_ptr, d_V_);
	CUDA_CHECK_KERNEL

	vbo_normal.unmap();
}

// ============================== PRINTING =====================================

/** 
 *	Calculate spreads
 */
__global__ void kernel_calculate_spread(cuFloatComplex* spinor, 
                                        float* deltaX,
                                        float* deltaY)
{
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	int j = blockIdx.y*blockDim.y + threadIdx.y;
//	
//	if(i < d_L && j < d_L)
//	{
//		float deltax_nom = 0.0f; 
//		float deltax_den = 0.0f;
//		float deltay_nom = 0.0f; 
//		float deltay_den = 0.0f;

//		const float dV = dx_*dx_;
//	
//		float x = dx * ( i -0.5 *(d_L_-1));
//		float y = dx * ( j -0.5 *(d_L_-1));

//		}

//	// Update global variables
//	deltax_ = std::sqrt(deltax_nom/deltax_den);
//	deltay_ = std::sqrt(deltay_nom/deltay_den);
//	}
}


void QLB::calculate_spread_cuda()
{	
	copy_from_device_to_host(d_spinor_, spinor_);
	calculate_spread();
	d_deltaX_.push_back(static_cast<float>(deltax_));
	d_deltaY_.push_back(static_cast<float>(deltay_));
}

void QLB::write_spread_cuda()
{
	WARNING("this is highly inefficient");
//	std::size_t N = std::size_t(t_);

//	std::vector<float> deltaX(N);
//	std::vector<float> deltaY(N);

//	copy_from_device_to_host(d_deltaX_, deltaX);
//	copy_from_device_to_host(d_deltaY_, deltaY);
	
	fout.open("spread.dat");
	
	fout << std::left << std::setprecision(6);

	for(std::size_t i = 0; i < d_deltaX_.size(); ++i)
	{
		fout << std::setw(15) << i*dt_;
		fout << std::setw(15) << d_deltaX_[i];
		fout << std::setw(15) << d_deltaY_[i];
	
		if(V_indx_ == 0) // no potential
		{
			float_t deltax_t = std::sqrt( delta0_*delta0_ + i*dt_*i*dt_ /
			                            (4.0*mass_*mass_*delta0_*delta0_) );
			fout << std::setw(15) << deltax_t;
		}

		fout << std::endl;
	}
		
	fout.close();
}

