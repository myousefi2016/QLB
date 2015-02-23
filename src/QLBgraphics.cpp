/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	This file contains all the OpenGL context and handles rendering
 */
 
#include "QLB.hpp"

#ifndef QLB_USE_DUMMY

#define x(i,j) 	3*((i)*L_ + (j))
#define y(i,j)  3*((i)*L_ + (j)) + 1
#define z(i,j)  3*((i)*L_ + (j)) + 2

void QLB::init_GL()
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
	 *  This is why we are storing the y-coordinates form the simulation 
	 *  in the z array position
	 */  
	for(unsigned i = 0; i < L_; ++i)
		for(unsigned j = 0; j < L_; ++j)
		{
			array_vertex_[x(i,j)] = dx_*(i-0.5*(L_-1));  // x
			array_vertex_[y(i,j)] = 0.0;                 // y
			array_vertex_[z(i,j)] = dx_*(j-0.5*(L_-1));  // z
		}
		
	/* Index array for GL_TRIANGLE_STRIP. The index array describes the order
	 * in which we are going to draw the vertices. In total we have to draw
	 * 2*L*(L-1) vertices.
	 *
	 *    0 ------ 3 ------ 6
	 *    |        |        |
	 *    |        |        |
	 *    1 ------ 4 ------ 7
	 *    |        |        |
	 *    |        |        |
	 *    2 ------ 5 ------ 8
	 *
	 *	 array_index = [ 0, 3, 1, 4, 2, 5, 8, 5, 7, 4, 6, 3 ]
	 */
	int even = 0;
	int odd  = L_;
	
	for(unsigned i = 0; i < (L_-1); ++i)
	{
		if(i % 2 == 0)
		{
			even = i*L_;
			odd  = (i+1)*L_;
			for(unsigned j = 0; j < 2*L_; ++j)
			{
				if(j % 2 == 0)
//					std::cout << even++ << std::endl;
					array_index_[2*L_*i + j] = even++;
				else 
//					std::cout << odd++ << std::endl;
					array_index_[2*L_*i + j] = odd++;
			}
		}
		else
		{
			even += 2*L_-1;
			odd  -= 1;
			for(unsigned j = 0; j < 2*L_; ++j)
			{
				if(j % 2 == 0)
//					std::cout << even-- << std::endl;
					array_index_[2*L_*i + j] = even--;
				else 
//					std::cout << odd-- << std::endl;
					array_index_[2*L_*i + j] = odd--;
			}
			
		}
//		std::cout << std::endl;
	}


	// Setup shaders
//	ShaderLoader shader_vertex;
//	shader_vertex.load_from_file("shaders/vertex_shader.glsl", GL_VERTEX_SHADER);

//	ShaderLoader shader_fragment;
//	shader_fragment.load_from_file("shaders/fragment_shader.glsl",GL_FRAGMENT_SHADER);

//	shader_.add_shaders(shader_vertex, shader_fragment);
	
	GL_is_initialzed_ = true;
}

void QLB::prepare_arrays()
{
	// Copy data to vertex array
	switch( current_scene_ )
	{
		case spinor0:
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
					array_vertex_[y(i,j)] = scaling_*std::norm(spinor_(i,j,0));
			break;
		case spinor1:
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
					array_vertex_[y(i,j)] = scaling_*std::norm(spinor_(i,j,1));
			break;
		case spinor2:
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
					array_vertex_[y(i,j)] = scaling_*std::norm(spinor_(i,j,2));
			break;
		case spinor3:
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
					array_vertex_[y(i,j)] = scaling_*std::norm(spinor_(i,j,3));
			break;
		case potential:
			for(unsigned i = 0; i < L_; ++i)
				for(unsigned j = 0; j < L_; ++j)
				{
					if(V_indx_ == 0)
						array_vertex_[y(i,j)] = scaling_*std::abs(V_harmonic(i,j));
					else
						array_vertex_[y(i,j)] = std::abs(V_free(i,j)); 
				}
			break;
	}
	
	/* Calculate normal by taking the cross product of ( a x b )
	 * Note: we are using periodic boundary conditions
	 *
	 *   (i,jk)                                b
	 *     ^                                   ^
	 *     |                       <====>      |
	 *     |                                   |
	 *   (i,j) ------> (ik,j)                  x ------> a
	 */

	float_t a1, a2, a3, b1, b2, b3, norm;
	float_t signa = 1.0, signb = 1.0;
	int ik, jk;
	for(unsigned i = 0; i < L_; ++i)
	{
		signb = -1.0;
		if(i == L_-1) signa = -1.0;

		for(unsigned j = 0; j < L_; ++j)
		{
			jk = (L_ - 1 + j) % L_;
			ik = (i + 1) % L_;

			// a
			a1 = signa*array_vertex_[x(ik,j)] - signa*array_vertex_[x(i,j)];
			a2 = signa*array_vertex_[y(ik,j)] - signa*array_vertex_[y(i,j)];
			a3 = signa*array_vertex_[z(ik,j)] - signa*array_vertex_[z(i,j)];
			
			// b
			b1 = signb*array_vertex_[x(i,jk)] - signb*array_vertex_[x(i,j)];
			b2 = signb*array_vertex_[y(i,jk)] - signb*array_vertex_[y(i,j)];
			b3 = signb*array_vertex_[z(i,jk)] - signb*array_vertex_[z(i,j)];
			
			// n = a x b
			array_normal_[x(i,j)] = a2*b3 - a3*b2;
			array_normal_[y(i,j)] = a3*b1 - a1*b3;
			array_normal_[z(i,j)] = a1*b2 - a2*b1;

			norm = std::sqrt(array_normal_[x(i,j)]*array_normal_[x(i,j)] + 
			                 array_normal_[y(i,j)]*array_normal_[y(i,j)] +
			                 array_normal_[z(i,j)]*array_normal_[z(i,j)]);
			// normalize
			array_normal_[x(i,j)] /= norm;
			array_normal_[y(i,j)] /= norm;
			array_normal_[z(i,j)] /= norm;
			
			signb = 1.0;
		}
	}
}

void QLB::render()
{
	if(!GL_is_initialzed_)
		FATAL_ERROR("QLB::OpenGL context is not initialized");
	
	prepare_arrays();
	
	draw_coordinate_system();	

	//shader_.use_shader();
	
	glColor3d(1,1,1);
	glBegin(current_render_);
	for(unsigned i = 0; i < 2*L_*(L_-1); ++i)
	{
		glVertex3d(array_vertex_[3*array_index_[i]],
	               array_vertex_[3*array_index_[i]+1],
	               array_vertex_[3*array_index_[i]+2]);
		glNormal3d(array_normal_[3*array_index_[i]],
		           array_normal_[3*array_index_[i]+1],
		           array_normal_[3*array_index_[i]+2]);
	}	
	glEnd();	
}


void QLB::draw_coordinate_system() const
{
	// Draw origin
	glPushMatrix();
		glColor3d(1,0,0);
		glTranslated(0,0,0);		
		glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	// Draw x - red
	glPushMatrix();
		glColor3d(1,0,0);
		glTranslated(dx_*L_/2.0,0,0);		
		glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	glBegin(GL_LINES);
		glVertex3d(0,0,0);
		glVertex3d(dx_*L_/2.0,0,0);
	glEnd();
	
	// Draw y - green
	glPushMatrix();
		glColor3d(0,1,0);
		glTranslated(0,dx_*L_/2.0,0);		
		glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	glBegin(GL_LINES);
		glVertex3d(0,0,0);
		glVertex3d(0,dx_*L_/2.0,0);
	glEnd();
	
	// Draw z - blue
	glPushMatrix();
		glColor3d(0,0,1);
		glTranslated(0,0,dx_*L_/2.0);		
		glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	glBegin(GL_LINES);
		glVertex3d(0,0,0);
		glVertex3d(0,0,dx_*L_/2.0);
	glEnd();
	
	glColor3d(1,1,1);
}

#undef x
#undef y
#undef z

#endif /* QLB_USE_DUMMY */
