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
			array_vertex_[y(i,j)] = 0.0;                 
			array_vertex_[z(i,j)] = dx_*(j-0.5*(L_-1));  // y
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
	 * vertices. In this case l we have to draw 2*L*(L-1) vertices.
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
	vbo_vertex.malloc(array_vertex_.size()*sizeof(float_t));
	vbo_vertex.unbind();
	
	vbo_normal.init(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);  
	vbo_normal.bind();
	vbo_normal.malloc(array_normal_.size()*sizeof(float_t));
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
						array_vertex_[y(i,j)] = std::abs(V_free(i,j)); 
					else
						array_vertex_[y(i,j)] = scaling_*std::abs(V_harmonic(i,j));
				}
			break;
	}
	
	// Copy to vertex VBO
	vbo_vertex.bind();
	vbo_vertex.BufferSubData(0, array_vertex_.size()*sizeof(float_t), 
	                         &array_vertex_[0]);
	vbo_vertex.unbind();
	
	if( normal_per_face_ )
	{
		/* Calculate normal by taking the cross product of each triangle
		 * and store it at each vertex position of the triangle.
		 */
	 
	 	float_t v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, n_x, n_y, n_z, norm;
		unsigned ind_a, ind_b, ind_c;
	 
		for(unsigned i = 0; i < array_normal_.size(); ++i)
			array_normal_[i] = 0.0;
	
		const unsigned nvertices = 6*(L_-1)*(L_-1);
		for(unsigned i = 0; i < nvertices; i+=3)
		{
			ind_a = 3*array_index_solid_[i];
			ind_b = 3*array_index_solid_[i+1];
			ind_c = 3*array_index_solid_[i+2];
	
			// v1 = a - b 
			v1_x = array_vertex_[ind_a    ] - array_vertex_[ind_b    ];
			v1_y = array_vertex_[ind_a + 1] - array_vertex_[ind_b + 1];
			v1_z = array_vertex_[ind_a + 2] - array_vertex_[ind_b + 2];
		
			// v2 = c - b
			v2_x = array_vertex_[ind_c    ] - array_vertex_[ind_b    ];
			v2_y = array_vertex_[ind_c + 1] - array_vertex_[ind_b + 1];
			v2_z = array_vertex_[ind_c + 2] - array_vertex_[ind_b + 2];

			// n = v1 x v2
			n_x = v1_y * v2_z - v1_z * v2_y;
			n_y = v1_z * v2_x - v1_x * v2_z;
			n_z = v1_x * v2_y - v1_y * v2_x;

			// normalize
			norm = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
			n_x /= norm;
			n_y /= norm;
			n_z /= norm;
		
			// Add the contribution from this triangle to each adjacent vertex.
			array_normal_[ind_a    ] += n_x;
			array_normal_[ind_a + 1] += n_y;
			array_normal_[ind_a + 2] += n_z;
		
			array_normal_[ind_b    ] += n_x;
			array_normal_[ind_b + 1] += n_y;
			array_normal_[ind_b + 2] += n_z;

			array_normal_[ind_c    ] += n_x;
			array_normal_[ind_c + 1] += n_y;
			array_normal_[ind_c + 2] += n_z;
		}
	
		for(unsigned i = 0; i < L_; ++i)
			for(unsigned j = 0; j < L_; ++j)
			{
				n_x  = array_normal_[x(i,j)];
				n_y  = array_normal_[y(i,j)];
				n_z  = array_normal_[z(i,j)];
			
				norm = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
			
				array_normal_[x(i,j)] /= norm;
				array_normal_[y(i,j)] /= norm;
				array_normal_[z(i,j)] /= norm;
			}
	}
	else
	{
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
		unsigned ik, jk;
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
		
	// Copy to normal VBO
	vbo_normal.bind();
	vbo_normal.BufferSubData(0, array_normal_.size()*sizeof(float_t),
	                         &array_normal_[0]);
	vbo_normal.unbind();
}

void QLB::render()
{
	if(!GL_is_initialzed_)
		FATAL_ERROR("QLB::OpenGL context is not initialized");
	
	prepare_arrays();
	
//	draw_coordinate_system();	

//	shader_.use_shader();
	
//	glColor3d(1,1,1);
//	glBegin(current_render_);
//	const unsigned nvertices = 6*(L_-1)*(L_-1);
//	for(unsigned i = 0; i < nvertices; i++)
//	{
//		glNormal3d(array_normal_[3*array_index_solid_[i]    ],   // x
//	               array_normal_[3*array_index_solid_[i] + 1],   // y
//	               array_normal_[3*array_index_solid_[i] + 2]);  // z
//	               
//		glVertex3d(array_vertex_[3*array_index_solid_[i]    ],   // x
//	               array_vertex_[3*array_index_solid_[i] + 1],   // y
//	               array_vertex_[3*array_index_solid_[i] + 2]);  // z
//	}	
//	glEnd();

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
	glNormalPointer(QLB_FLOAT_T, 0, 0);
	vbo_normal.unbind();
	
	vbo_vertex.bind();
	glVertexPointer(3, QLB_FLOAT_T, 0, 0);
	vbo_vertex.unbind();

	glDrawElements(current_render_, (GLsizei) n_elements, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if(current_render_ == SOLID)
		vbo_index_solid.unbind();
	else 
		vbo_index_wire.unbind();
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
#undef tri

