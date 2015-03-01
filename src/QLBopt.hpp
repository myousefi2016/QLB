/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  QLBopt is used to set various options of the QLB class.
 *  The options are:
 *   - plot    unsigned integer where the bits indicate which quantities are
 *             written to file after calling 'QLB::write_content_to_file()'
 *             [ 0]    :  all           <==>        1
 *             [ 1]    :  spread        <==>        2
 *             [ 2]    :  spinor1       <==>        4
 *             [ 3]    :  spinor2       <==>        8
 *             [ 4]    :  spinor3       <==>       16
 *             [ 5]    :  spinor4       <==>       32
 *             [ 6]    :  density       <==>       64
 *             [ 7]    :  currentX      <==>      128
 *             [ 8]    :  currentY      <==>      256
 *             [ 9]    :  veloX         <==>      512
 *             [10]    :  veloY         <==>     1024
 *  - verbose  enables verbose mode to get some additional information written
 *             to STDOUT during the simulation
 *  - stats    time each run of the simulation and allow usage of 'QLB::stats()'
 */

#ifndef QLB_OPT_HPP
#define QLB_OPT_HPP

class QLBopt
{
public:

	enum plot_t {     all = 1 << 0,    spread = 1 << 1,   spinor1 = 1 << 2,
	              spinor2 = 1 << 3,   spinor3 = 1 << 4,   spinor4 = 1 << 5,
	              density = 1 << 6,  currentX = 1 << 7,  currentY = 1 << 8,
	                veloX = 1 << 9,     veloY = 1 << 10  };
	
	// === Constructor ===
	QLBopt()
		:	plot_(0), verbose_(false), stats_(false)
	{}
	
	QLBopt(unsigned int plot, bool verb, bool stats)
		:	plot_(plot), verbose_(verb), stats_(stats)
	{}
	
	QLBopt(const QLBopt& opt)
		:	plot_(opt.plot()), verbose_(opt.verbose()), stats_(opt.stats())
	{}
	
	// === Getter ===
	inline unsigned int plot() const { return plot_; }
	inline bool verbose() const { return verbose_; }
	inline bool stats() const { return stats_; }

private:
	unsigned int plot_;
	bool verbose_;
	bool stats_;
};

#endif /* QLBopt.hpp */
