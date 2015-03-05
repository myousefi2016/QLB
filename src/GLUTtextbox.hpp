/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Class to draw strings and text boxes in OpenGL, this class is used in
 *  UserInterface from GLUTui.hpp
 */

#ifndef GLUT_TEXTBOX_HPP
#define GLUT_TEXTBOX_HPP

// System includes
#include <iostream>
#include <string>

// Local includes
#include "GLerror.hpp"

#define TB_WHITESPACES         5
#define TB_FONT_PIXEL_WIDTH    16
#define TB_FONT_PIXEL_HEIGHT   26

/***********************
 *       TextBox       *
 ***********************/
class TextBox
{
public:
	typedef std::vector<std::string> svec_t;
	typedef std::vector<std::string>::iterator siterator_t;

	TextBox() {}

	/**
	 *	Constructor
	 *	@param  x               x-Coordinate in [-1, 1] of the bottom-left corner
 	 *	@param  y               y-Coordinate in [-1, 1] of the bottom-left corner
 	 *	@param  w               width of the box [0,2]
 	 *	@param  h               height of the box [0,2]
	 *	@param	nrow            number of rows
	 *	@param  ncol            number of columns
	 *	@param  has_border      boolean whether a border is drawn
	 *	@param 	has_background  boolean whether a background is drawn
	 *	@param  width_aligned   boolean whether the box is width aligned to the window
	 *	@param  width_relative  boolean whether the box is width is independant
	 *                          of the current window size (paramter w is ignored)
	 */
	TextBox(float x, float y, float w, float h, std::size_t nrow, std::size_t ncol, 
	        bool has_border, bool has_background, bool width_aligned, 
	        bool width_relative, float width_relative_scal);
	        
	/**
	 *	Initialize all variables (this is mainly used to avoid invocation of 
	 *	assignment operator when using TextBox in a container)
	 */
	void init(float x, float y, float w, float h, std::size_t nrow, 
	          std::size_t ncol, bool has_border, bool has_background,
	          bool width_aligned, bool width_relative, float width_relative_scal);
	
	/**
	 *	Add text to the string array
	 *	@param  pos         position in [0, nrow*ncol)
	 *	@param  str         string to be added
	 */        
	void add_text(std::size_t pos, std::string str);

	/**
	 *	Add text to the string array
	 *	@param  begin       iterator to the begin of a vector of size (nrow*ncol)
	 *	@param  end         iterator to the end of a vector size (nrow*ncol)
	 */        
	void add_text(siterator_t begin, siterator_t end);
	
	/**
	 *	Draw the content (OpenGL context required)
	 *	@param	width       width of the window
	 *	@param 	height      height of the window
	 */
	void draw(int width, int height) const;

	// === Getter ===
	bool is_active() const { return is_active_; }
	
	// === Setter ===
	void activate()   { is_active_ = true;  }
	void deactivate() { is_active_ = false; }
	
private:
	float x_,y_;
	float w_,h_;

	svec_t text_;
	std::size_t nrow_, ncol_;
	
	bool has_border_;
	bool has_background_;
	bool width_aligned_;
	bool width_relative_;
	float width_relative_scal_;
	
	bool is_active_;
};

#endif /* GLUTtextbox.hpp */
