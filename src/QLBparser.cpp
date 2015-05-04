/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Parse potential and initial conditions passed by the options '--potential=FILE' 
 *  and '--initial=FILE' and setup the QLB class.
 *	For the exact formatting of the files take a look at 'InputGenerator.py'
 */

#include "QLBparser.hpp"

QLBparser::QLBparser(std::string potential_file, std::string initial_file)
	:
		potential_(0),
		initial_(0),
		potential_is_present_(true),
		initial_is_present_(true),
		L_(0),
		dx_(0),
		mass_is_present_(false),
		mass_(0),
		delta0_is_present_(false),
		delta0_(0),
		initial_file_(""),
		potential_file_("")
		
{
	if(potential_file.empty())
		potential_is_present_ = false;
	else
	{
		pfin.open(potential_file);
	
		if(!pfin.is_open() && !pfin.good())
			FATAL_ERROR("Cannot open '"+potential_file+"'");
		
		potential_file_ = potential_file;
	}
		
	if(initial_file.empty())
		initial_is_present_ = false;
	else
	{
		ifin.open(initial_file);
	
		if(!ifin.is_open() && !ifin.good())
			FATAL_ERROR("Cannot open '"+initial_file+"'");
		
		initial_file_ = initial_file;
	}	
}

QLBparser::~QLBparser()
{
	if(ifin.is_open()) ifin.close();
	if(pfin.is_open()) pfin.close();
}

QLBparser::QLBparser(const QLBparser& parser)
{
	potential_is_present_ = parser.potential_is_present();
	initial_is_present_   = parser.initial_is_present();

	if(parser.is_valid())
	{
		L_ = parser.L();
		dx_ = parser.dx();

		mass_is_present_ = parser.mass_is_present();
		mass_ = parser.mass();
		
		delta0_is_present_ = parser.delta0_is_present();
		delta0_ = parser.delta0();
		
		potential_file_ = parser.potential_file();
		initial_file_ = parser.initial_file();
		
		potential_ = parser.potential_;
		initial_ = parser.initial_;

	}
}

/**
 *	Extract the value from a string
 *	@tparam  T    type of the extracted value
 *	@param   s    string
 */
template< typename T >
T extract_value(std::string s)
{
	std::istringstream value_str(s);
	T value;
	value_str >> value;
	return value;
}

/**
 *	Inform about conflicting arguments
 *	@param  arg    name of the argument
 *	@param  val1   parsed value of the first argument
 *	@param  val2   parsed value of the second argument
 */
template < typename T >
inline void conflict(std::string arg, T val1, T val2)
{
	std::cout << "FAILED" << std::endl;
	FATAL_ERROR("conflicting arguments in input files for '"+arg+"' : "+
	            std::to_string(val1)+" != "+std::to_string(val2));
}

void QLBparser::parse_input(const CmdArgParser* cmd)
{
	if(is_valid())
	{
		std::array<std::string, 10> keywords;
		keywords[0] = "$BEGIN_INPUT";
		keywords[1] = "$L";
		keywords[2] = "$DX";
		keywords[3] = "$MASS";
		keywords[4] = "$DELTA0";
		keywords[5] = "$BEGIN_POTENTIAL";
		keywords[6] = "$END_POTENTIAL";
		keywords[7] = "$BEGIN_INITIAL";
		keywords[8] = "$END_INITIAL";
		keywords[9] = "$END_INPUT";

		std::size_t pos;
		
		// Parse potential file
		if(potential_is_present_)
		{
			// Mandatory arguments
			bool begin_found = false;
			bool L_found = false;
			bool dx_found = false;
			bool end_found = false;
		
			// Optional arguments
			bool mass_found = false;
			bool delta0_found = false;
			bool potential_begin_found = false;
			bool potential_end_found = false;
		
			std::cout << "Parsing '" << potential_file_ << "' ... " << std::flush;
		
			std::string line;
			
			while(!std::getline(pfin, line).eof())
			{
				if(!begin_found)
				{
					begin_found = line.find(keywords[0]) != std::string::npos; 
				}	
				else if(!L_found)
				{
					pos = line.find(keywords[1]);
					L_found = pos != std::string::npos;
					if(L_found)
					{
						std::string value(line, pos+keywords[1].size()+1,
						                  line.size());
						L_ = extract_value<unsigned>(value);
					}
				}				
				else if(!dx_found)
				{
					pos = line.find(keywords[2]);
					dx_found = pos != std::string::npos;
					if(dx_found)
					{
						std::string value(line, pos+keywords[2].size()+1,
						                  line.size());
						dx_ = extract_value<float>(value);
					}
				}
				else
				{
					if(!mass_found)
					{
						pos = line.find(keywords[3]);
						mass_found = pos != std::string::npos;
						if(mass_found)
						{
							std::string value(line, pos+keywords[3].size()+1,
							                  line.size());
							mass_ = extract_value<float>(value);
							mass_is_present_ = true;
						}
					}
					
					if(!delta0_found)
					{
						pos = line.find(keywords[4]);
						delta0_found = pos != std::string::npos;
						if(delta0_found)
						{
							std::string value(line, pos+keywords[4].size()+1, 
							                  line.size());
							delta0_ = extract_value<float>(value);
							delta0_is_present_ = true;
						}
					}
					
					if(!potential_begin_found)
					{
						pos = line.find(keywords[5]);
						potential_begin_found = pos != std::string::npos;

						if(potential_begin_found)
						{
							potential_.resize(L_*L_);
							for(unsigned i = 0; i < L_; ++i)
								for(unsigned j = 0; j < L_; ++j)
									pfin >> potential_[i*L_ + j];				
						}
					}
					
					if(!potential_end_found)
					{
						pos = line.find(keywords[6]);
						potential_end_found = pos != std::string::npos;
					}
					
					if(!end_found)
					{
						end_found = line.find(keywords[9]) != std::string::npos; 
					}
				}
			}
			
			if(!begin_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$BEGIN_INPUT'"
				            " while parsing '"+potential_file_+"'");
			}
			else if(!L_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$L'"
				            " while parsing '"+potential_file_+"'");
			}
			else if(!dx_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$DX'"
				            " while parsing '"+potential_file_+"'");
			}
			else if(!potential_begin_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$POTENTIAL_BEGIN'"
				            " while parsing '"+potential_file_+"'");
			}
			else if(!potential_end_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$POTENTIAL_END'"
				            " while parsing '"+potential_file_+"'");
			}
			else if(!end_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$END_INPUT'"
				            " while parsing '"+potential_file_+"'");
			}
			
			pfin.close();
			std::cout << "Done" << std::endl;
		}
	
		// Parse inital condition file
		if(initial_is_present_)
		{
			// Mandatory arguments
			bool begin_found = false;
			bool L_found = false;
			bool dx_found = false;
			bool end_found = false;
		
			// Optional arguments
			bool mass_found = false;
			bool delta0_found = false;
			bool initial_begin_found = false;
			bool initial_end_found = false;		
			
			std::cout << "Parsing '" << initial_file_ << "' ... " << std::flush;
		
			std::string line;
			
			while(!std::getline(ifin, line).eof())
			{
				if(!begin_found)
				{
					begin_found = line.find(keywords[0]) != std::string::npos; 
				}	
				else if(!L_found)
				{
					pos = line.find(keywords[1]);
					L_found = pos != std::string::npos;
					if(L_found)
					{
						std::string value(line, pos+keywords[1].size()+1,
						                  line.size());
						unsigned L = extract_value<unsigned>(value);
						
						if(potential_is_present_ && L_ != L)
							conflict("L", L_, L);
						else
							L_ = L;
					}
				}				
				else if(!dx_found)
				{
					pos = line.find(keywords[2]);
					dx_found = pos != std::string::npos;
					if(dx_found)
					{
						std::string value(line, pos+keywords[2].size()+1,
						                  line.size());
						float dx = extract_value<float>(value);
						
						if(potential_is_present_ && dx_ != dx)
							conflict("dx", dx_, dx);
						else
							dx_ = dx;
					}
				}
				else
				{
					if(!mass_found)
					{
						pos = line.find(keywords[3]);
						mass_found = pos != std::string::npos;
						if(mass_found)
						{
							std::string value(line, pos+keywords[3].size()+1,
							                  line.size());
							float mass = extract_value<float>(value);
							             
							if(potential_is_present_ && mass_is_present_ &&
							   mass_ != mass)
								conflict("mass", mass_, mass);
							else
							{
								mass_is_present_ = true;
								mass_ = mass;
							}
						}
					}
					
					if(!delta0_found)
					{
						pos = line.find(keywords[4]);
						delta0_found = pos != std::string::npos;
						if(delta0_found)
						{
							std::string value(line, pos+keywords[4].size()+1, 
							                  line.size());
							float delta0 = extract_value<float>(value);
							
							if(potential_is_present_ && delta0_is_present_ &&
							   delta0_ != delta0)
								conflict("delta0", delta0_, delta0);
							else
							{
								delta0_is_present_ = true;
								delta0_ = delta0;
							}
							
						}
					}
					
					if(!initial_begin_found)
					{
						pos = line.find(keywords[7]);
						initial_begin_found = pos != std::string::npos;

						if(initial_begin_found)
						{
							initial_.resize(4*L_*L_);
							
							for(unsigned i = 0; i < L_; ++i)
								for(unsigned j = 0; j < L_; ++j)
									for(unsigned k = 0; k < 4; ++k)
										ifin >> initial_[4*L_*i + 4*j + k];
						}
					}
					
					if(!initial_end_found)
					{
						pos = line.find(keywords[8]);
						initial_end_found = pos != std::string::npos;
					}
				
					if(!end_found)
					{
						end_found = line.find(keywords[9]) != std::string::npos; 
					}
				}
			}
			
			
			if(!begin_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$BEGIN_INPUT'"
				            " while parsing '"+initial_file_+"'");
			}
			else if(!L_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$L'"
				            " while parsing '"+initial_file_+"'");
			}
			else if(!dx_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$DX'"
				            " while parsing '"+initial_file_+"'");
			}
			else if(!initial_begin_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$INITIAL_BEGIN'"
				            " while parsing '"+initial_file_+"'");
			}
			else if(!initial_end_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$INITIAL_END'"
				            " while parsing '"+initial_file_+"'");
			}
			else if(!end_found)
			{
				std::cout << "FAILED" << std::endl;
				FATAL_ERROR("invalid format : expected '$END_INPUT'"
				            " while parsing '"+initial_file_+"'");
			}
			
			ifin.close();
			std::cout << "Done" << std::endl;
		}
		
		// Check if we have conflicting arguments
		if(cmd->L() && (cmd->L_value() != L_))
			WARNING("ignoring option '--L="+std::to_string(cmd->L_value())+"'");		
		
		if(cmd->dx() && (cmd->dx_value() != dx_))
			WARNING("ignoring option '--dx="+std::to_string(cmd->L_value())+"'");
		
		if(cmd->delta0() && delta0_is_present_ && (cmd->delta0_value() != delta0_))		
			WARNING("ignoring option '--delta0="+std::to_string(cmd->delta0_value())+"'");
		
		if(cmd->mass() && mass_is_present_ && (cmd->mass_value() != mass_))		
			WARNING("ignoring option '--mass="+std::to_string(cmd->mass_value())+"'");
	}
}
