/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  This file contains all the OpenGL context and handles rendering
 */
 
#include "QLB.hpp"

#define x(i,j) 	3*((i)*L_ + (j))
#define y(i,j)  3*((i)*L_ + (j)) + 1
#define z(i,j)  3*((i)*L_ + (j)) + 2

#define tri(i,j) 6*((i)*(L_-1) + (j))

void QLB::init_GL(bool static_viewer)
{
	/* Initialize the arrays
	 * Note: the coordinate system in OpenGL is as follows
	 *       
	 *     y ^ 
	 *       |
	 *       |
	 *       0 -----> x
	 *      /
	 *     z  
	 *
	 * This is why we are storing the y-coordinates form the simulation 
	 * in the z array position
	 */
	if(!static_viewer)
	{   
		const float dx = float(dx_);
		for(unsigned i = 0; i < L_; ++i)
			for(unsigned j = 0; j < L_; ++j)
			{
				array_vertex_[x(i,j)] = dx*(i-0.5f*(L_-1));  // x
				array_vertex_[y(i,j)] = 0.0f;                 
				array_vertex_[z(i,j)] = dx*(j-0.5f*(L_-1));  // y
			}
	}
		
	/* Index array for GL_TRIANGLES. (used when drawing solid)
	 * The index array describes the order in which we are going to draw the 
	 * vertices. In total we have to draw 6*(L-1)*(L-1) vertices.
	 *
	 *    0 ------ 3 ------ 6           1. Triangle { 1, 0, 3}
	 *    |        |        |           2. Triangle { 4, 1, 3}
	 *    |        |        |           3. Triangle { 2, 1, 4}
	 *    1 ------ 4 ------ 7           4. Triangle { 5, 2, 4}
	 *    |        |        |           5. Triangle { 4, 3, 6}
	 *    |        |        |           6. Triangle { 7, 4, 6}
	 *    2 ------ 5 ------ 8           7. Triangle { 5, 4, 7}
	 *                                  8. Triangle { 8, 5, 7}
	 *
	 *   index = [ 1, 0, 3, 4, 1, 3, 2, 1, 4, 5, 2, 4, 4, 3, 6, 7, 4, 6
	 *             5, 4, 7, 8, 5, 7 ]
	 */
	
	for(unsigned i = 0; i < L_-1; ++i)
		for(unsigned j = 0; j < L_-1; ++j)
		{
			// We always store 2 triangles at once
			array_index_solid_[tri(i,j)    ] = i*L_ + j + 1;
			array_index_solid_[tri(i,j) + 1] = i*L_ + j;
			array_index_solid_[tri(i,j) + 2] = i*L_ + j + L_;
		
			array_index_solid_[tri(i,j) + 3] = i*L_ + j + L_ + 1;
			array_index_solid_[tri(i,j) + 4] = i*L_ + j + 1;
			array_index_solid_[tri(i,j) + 5] = i*L_ + j + L_;
		}
		
	/* Index array for GL_TRIANGLE_STRIP (used when drawing wire frame)
	 * The index array describes the order in which we are going to draw the 
	 * vertices. In this case we have to draw 2*L*(L-1) vertices.
	 *
	 *    0 ------ 3 ------ 6
	 *    |        |        |
	 *    |        |        |
	 *    1 ------ 4 ------ 7
	 *    |        |        |
	 *    |        |        |
	 *    2 ------ 5 ------ 8
	 *
	 *	 index = [ 0, 3, 1, 4, 2, 5, 8, 5, 7, 4, 6, 3 ]
	 */
	
	unsigned even = 0;
	unsigned odd  = L_;
	int inc = 1;
	
	for(unsigned i = 0; i < (L_-1); ++i)
	{
		if(i % 2 == 0)
		{
			even = i*L_;
			odd  = (i+1)*L_;
		}
		else
		{
			even += 2*L_-1;
			odd  -= 1;
		}
		
		for(unsigned j = 0; j < 2*L_; ++j)
		{
			if(j % 2 == 0)
			{
				array_index_wire_[2*L_*i + j] = even;
				even += inc;
			}
			else
			{ 
				array_index_wire_[2*L_*i + j] = odd;
				odd += inc;
			}
		}
		inc *= -1;
	}
	
	// Setup Vertex Buffer Objects
	vbo_vertex.init(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);  
	vbo_vertex.bind();
	vbo_vertex.malloc(array_vertex_.size()*sizeof(float));
	vbo_vertex.unbind();
	
	vbo_normal.init(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);  
	vbo_normal.bind();
	vbo_normal.malloc(array_normal_.size()*sizeof(float));
	vbo_normal.unbind();
		
	vbo_index_solid.init(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
	vbo_index_solid.bind();
	vbo_index_solid.BufferData(array_index_solid_.size()*sizeof(unsigned), 
	                           &array_index_solid_[0]);
	vbo_index_solid.unbind();
	
	vbo_index_wire.init(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
	vbo_index_wire.bind();
	vbo_index_wire.BufferData(array_index_wire_.size()*sizeof(unsigned), 
	                          &array_index_wire_[0]);
	vbo_index_wire.unbind();
	
	if(opt_.device() == 2)
	{
		vbo_vertex.bind();
		vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
	    	                     &array_vertex_[0]);
		vbo_vertex.unbind();
	}
	
	GL_is_initialzed_ = true;
	
	if(static_viewer)
		render_statically(true);
}

void QLB::calculate_vertex(int tid, int nthreads)
{
	const int L_per_thread = L_ / nthreads;
	const int lower = tid*L_per_thread;
	const int upper = (tid + 1) != nthreads ? (tid+1)*L_per_thread : L_;
	const int L = L_;

	for(int i = lower; i < upper; ++i)
		for(int j = 0; j < L; ++j)
			array_vertex_[y(i,j)] = float(scaling_*std::norm(spinor_(i,j,current_scene_)));
}

void QLB::scale_vertex(int change_scaling)
{
	const float scaling = change_scaling == 1 ? 2.f : 1.f/2.f;;
	for(unsigned i = 0; i < L_; ++i)
		for(unsigned j = 0; j < L_; ++j)
			array_vertex_[y(i,j)] = scaling * array_vertex_[y(i,j)];
}

void QLB::calculate_normal(int tid, int nthreads)
{
	/* Calculate normal by taking the cross product of ( a x b )
	 *
	 *   (i,jk)                                 b
	 *     ^                                    ^
	 *     |                       <====>   dx  |
	 *     |                                    |
	 *   (i,j) ------> (ik,j)                   x ------> a
	 *                                              dx
	 */

	const unsigned L_per_thread = L_ / nthreads;
	const unsigned lower = tid*L_per_thread;
	const unsigned upper = (tid + 1) != nthreads ? (tid+1)*L_per_thread : L_;

	float a1, a2, b2, b3, norm;
	const float dx = float(dx_);
	unsigned ik, jk;

	for(unsigned i = lower; i < upper; ++i)
	{
		ik = (i + 1) % L_;

		for(unsigned j = 0; j < L_; ++j)
		{
			jk = (L_ - 1 + j) % L_;

			// a
			a1 = dx;
			a2 = array_vertex_[y(ik,j)] - array_vertex_[y(i,j)];
		
			// b
			b2 = array_vertex_[y(i,jk)] - array_vertex_[y(i,j)];
			b3 = -dx;
		
			// n = a x b   (optimized as a3 == 0 and b1 == 0)
			array_normal_[x(i,j)] =   a2*b3;
			array_normal_[y(i,j)] = - a1*b3;
			array_normal_[z(i,j)] =   a1*b2;

			norm = std::sqrt(array_normal_[x(i,j)]*array_normal_[x(i,j)] + 
			                 array_normal_[y(i,j)]*array_normal_[y(i,j)] +
			                 array_normal_[z(i,j)]*array_normal_[z(i,j)]);
			// normalize
			array_normal_[x(i,j)] /= norm;
			array_normal_[y(i,j)] /= norm;
			array_normal_[z(i,j)] /= norm;
		}
	}
}

void QLB::render()
{
	if(!GL_is_initialzed_)
		FATAL_ERROR("QLB::OpenGL context is not initialized");
	
	// Setup VBO's
	if(opt_.device() != 2)
	{
		// Copy vertex array to vertex VBO
		vbo_vertex.bind();
		vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
			                     &array_vertex_[0]);
		vbo_vertex.unbind();
		
		// Copy normal array to normal VBO
		vbo_normal.bind();
		vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float),
			                     &array_normal_[0]);
		vbo_normal.unbind();
	}
#ifdef QLB_HAS_CUDA
	else
	{
		calculate_vertex_spinor_cuda();
		calculate_normal_spinor_cuda();
		cudaDeviceSynchronize();
	}
#endif
	
	// === Draw the scene ===
	std::size_t n_elements;
	if(current_render_ == SOLID)
	{
		n_elements = array_index_solid_.size();
		vbo_index_solid.bind();
	}
	else
	{ 
		n_elements = array_index_wire_.size();
		vbo_index_wire.bind();
	}	

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	vbo_normal.bind();
	glNormalPointer(GL_FLOAT, 0, 0);
	vbo_normal.unbind();
	
	vbo_vertex.bind();
	glVertexPointer(3, GL_FLOAT, 0, 0);
	vbo_vertex.unbind();

	glDrawElements(current_render_, (GLsizei) n_elements, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if(current_render_ == SOLID)
		vbo_index_solid.unbind();
	else 
		vbo_index_wire.unbind();
		
		
	// === Draw potential ===
	if(draw_potential_)
	{
		glEnable(GL_BLEND);
	
		if(opt_.device() != 2)
		{
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
					array_vertex_[y(i,j)] = float(scaling_*std::abs(V_(i,j))) - 0.005f*L_;
		
			calculate_normal(0, 1); 
		
			vbo_vertex.bind();
			vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
						             &array_vertex_[0]);
			vbo_vertex.unbind();

			vbo_normal.bind();
			vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float),
						             &array_normal_[0]);
			vbo_normal.unbind();
		}
#ifdef QLB_HAS_CUDA
		else
		{
			calculate_vertex_V_cuda();
			calculate_normal_V_cuda();
			cudaDeviceSynchronize();
		}
#endif

		glColor4d(1, 0, 0, 0.20);
		
		vbo_index_solid.bind();

		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_VERTEX_ARRAY);
	
		vbo_normal.bind();
		glNormalPointer(GL_FLOAT, 0, 0);
		vbo_normal.unbind();
	
		vbo_vertex.bind();
		glVertexPointer(3, GL_FLOAT, 0, 0);
		vbo_vertex.unbind();

		glDrawElements( SOLID, (GLsizei) array_index_solid_.size(), 
		                GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		vbo_index_solid.unbind();
		glDisable(GL_BLEND);
		glColor4d(1, 1, 1, 1);
	}
}

void QLB::render_statically(bool VBO_changed)
{
	if(!GL_is_initialzed_)
		FATAL_ERROR("QLB::OpenGL context is not initialized");
	
	// Copy vertex array to vertex VBO
	if(VBO_changed)
	{
		vbo_vertex.bind();
		vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float), 
			                     &array_vertex_[0]);
		vbo_vertex.unbind();
		
		// Copy normal array to normal VBO
		vbo_normal.bind();
		vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float),
			                     &array_normal_[0]);
		vbo_normal.unbind();
	}
	
	// Draw the scene
	std::size_t n_elements;
	if(current_render_ == SOLID)
	{
		n_elements = array_index_solid_.size();
		vbo_index_solid.bind();
	}
	else
	{ 
		n_elements = array_index_wire_.size();
		vbo_index_wire.bind();
	}	

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	vbo_normal.bind();
	glNormalPointer(GL_FLOAT, 0, 0);
	vbo_normal.unbind();
	
	vbo_vertex.bind();
	glVertexPointer(3, GL_FLOAT, 0, 0);
	vbo_vertex.unbind();

	glDrawElements(current_render_, (GLsizei) n_elements, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if(current_render_ == SOLID)
		vbo_index_solid.unbind();
	else 
		vbo_index_wire.unbind();
}

#undef x
#undef y
#undef z
#undef tri
