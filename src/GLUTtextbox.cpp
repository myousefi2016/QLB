/**
 *	Quantum Lattice Boltzmann 
 *	(c) 2015 Fabian Thüring, ETH Zurich
 *
 *	Class to draw strings and text boxes in OpenGL, this class is part of
 *	the class UserInterface from GLUTui.hpp
 */

#include "GLUTtextbox.hpp"

TextBox::TextBox(float x, float y, float w, float h, std::size_t nrow, 
                 std::size_t ncol, int id, bool has_border, bool has_background,
	             bool width_aligned, bool height_aligned)
	:   x_(x), y_(y), w_(w), h_(h), text_(nrow*ncol), nrow_(nrow), ncol_(ncol), 
	    id_(id), has_border_(has_border), has_background_(has_background),
	    width_aligned_(width_aligned), height_aligned_(height_aligned), is_active_(true)
{}


void TextBox::init(float x, float y, float w, float h, std::size_t nrow, 
                   std::size_t ncol, int id, bool has_border, bool has_background,
                   bool width_aligned, bool height_aligned)
{
	x_ = x;
	y_ = y;
	
	w_ = w;
	h_ = h;

	text_.resize(nrow*ncol);
	
	nrow_ = nrow;
	ncol_ = ncol;
	
	id_ = id;
	has_border_ = has_border; 
	has_background_ = has_background;
	width_aligned_ = width_aligned;
	height_aligned_ = height_aligned;
	
	is_active_ = true;
}

void TextBox::add_text(std::size_t pos, std::string str)
{
	text_[pos] = str;
}
	
void TextBox::add_text(siterator_t begin, siterator_t end)
{
	siterator_t it_text = text_.begin();
	for(siterator_t it = begin; it != end; ++it)
		*it_text++ = *it;
}

void TextBox::draw(int width, int height) const
{
	// Save matrices from stack
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	
	// Disable some stuff
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );
//	glDisable( GL_CULL_FACE );
		
	glTranslatef(x_, y_, 0.0f);
	
	// === Background ===
	if(has_background_)
	{
		glColor3f(0.f, 0.f, 0.f);
		glBegin(GL_QUADS);
		glVertex2f(0.0f, 0.0f); glVertex2f(w_, 0.0f);
		glVertex2f(w_, h_);	    glVertex2f(0.0f, h_);
		glEnd();
	}
	
	
	const float x_offset = width_aligned_ ? (2.0f-w_)/4.0f : 0.0f;

	// === Border ===
	if(has_border_)
	{
		glBegin( GL_LINES );
		glColor3f(.5f, .5f, .5f);
		glVertex2f(x_offset      , 0.f); 
		glVertex2f(x_offset + w_ , 0.f);
		glVertex2f(x_offset      , 0.f); 
		glVertex2f(x_offset      , h_);

		glColor3f(0.9f, 0.9f, 0.9f);
		glVertex2f(x_offset + 0.f , h_); 
		glVertex2f(x_offset + w_  , h_);
		glVertex2f(x_offset + w_  , h_); 
		glVertex2f(x_offset + w_  , 0.f);

		glColor3f( 0.75f, 0.75f, 0.75f);
		glVertex2f(x_offset + 0.001f      , 0.001f); 
		glVertex2f(x_offset + w_ - 0.001f , 0.001f );
		glVertex2f(x_offset + 0.001f      , 0.001f); 
		glVertex2f(x_offset + 0.001f      , h_ - 0.001f );

		glColor3f(.9f, .9f, .9f );
		glVertex2f(x_offset + 0.001f      , h_ - 0.001f); 
		glVertex2f(x_offset + w_ - 0.001f , h_ - 0.001f);
		glVertex2f(x_offset + w_ - 0.001f , h_ - 0.001f); 
		glVertex2f(x_offset + w_ - 0.001f , 0.001f);
		glEnd();
	}
		
	// === Text ===
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	
	std::size_t num_char_in_row = 0;

	const float font_width   = TB_FONT_PIXEL_WIDTH  / float(width);
	
	float line_width  = 0.0f;
	float line_height = h_ / float(nrow_+1);
	const float line_width_offset  = 0.02f + x_offset;
	const float line_height_offset = 0.01f + h_ - 1.5f*line_height;;
	
	/* We are drawing column based. The following example show's how
	 * a 2x1 matrix would be drawn (i.e 2 rows and 1 column). c1 and c2 are
	 * the cursor position where we have to start drawing (set by glRasterPos2f)
	 *     
	 *                       ^
	 *                       | 3
	 *   1          2        v
	 * <----> <------------->
	 *                       ^
	 *                       | 4
	 *      c1.....TEXT..... v   
	 *                       ^
	 *                       | 4 
	 *      c2.....TEXT..... v
	 *      
	 *
	 * 1: line_width_offset
	 * 2: line_width
	 * 3: line_height_offset
	 * 4: line_height
	 */
	for(std::size_t j = 0; j < ncol_; ++j)
	{
		for(std::size_t i = 0; i < nrow_; ++i)
		{
			// x_ and y_ represent the bottom-left corner of the box
			glRasterPos2f(x_ + line_width_offset  + j*line_width, 
			              y_ + line_height_offset - i*line_height);

			for(std::size_t c = 0; c < text_[j*nrow_ + i].size(); ++c)
				glutBitmapCharacter(GLUT_BITMAP_8_BY_13, int(text_[j*nrow_+i][c]));
		}
		num_char_in_row  = text_[j*ncol_+nrow_-1].size();
		line_width = (num_char_in_row + (j+1)*TB_WHITESPACES)*font_width; 
	}
	
	glPopMatrix();
	
	glEnable( GL_DEPTH_TEST );
	glEnable( GL_LIGHTING );
	
	// Restore matrix stack
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();	
}
