/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  QLBopt is used to set various options of the QLB class.
 *  The options are:
 *   - plot    unsigned integer where the bits indicate which quantities are
 *             written to file after calling 'QLB::write_content_to_file()'
 *             all           <==>        1
 *             spread        <==>        2
 *             spinor1       <==>        4
 *             spinor2       <==>        8
 *             spinor3       <==>       16
 *             spinor4       <==>       32
 *             density       <==>       64
 *             currentX      <==>      128
 *             currentY      <==>      256
 *             veloX         <==>      512
 *             veloY         <==>     1024
 *             e.g to write spinor1 and currentX to file pass:
 *			   unsigned int plot = QLBopt::spinor1 | QLBopt::currentX;
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
