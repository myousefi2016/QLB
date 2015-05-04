#!usr/bin python
# -*- coding: utf-8 -*-
#
#  Quantum Lattice Boltzmann 
#  (c) 2015 Fabian Thüring, ETH Zürich
# 
#  This library provides the functionality to generate initial conditions 
#  and potentials for the program QLB. 
#  For an implemenation in MATLAB see 'InputGenerator.m'
#
#  Examples
#  --------
#  >>> import InputGenerator as ig
#  >>> L = 128
#  >>> InputObj = ig.InputGenerator(L)
#  >>> Vfunc = lambda x,y : x*y
#  >>> InputObj.setPotential(Vfunc)
#  >>> InputObj.writePotential('myPotential.dat')
#
#  For further examples take a look at '<QLB-dir>/input/examples/'
#

import sys, os, shutil, ntpath
from time import gmtime, strftime

try:
	import numpy as np
except ImportError:
	print "InputGenerator: error: Module 'numpy' not found"
	sys.exit(1)

class InputGenerator:
	"""InputGenerator library

	This library provides the functionality to generate initial conditions and 
	potentials for the program QLB.

	Examples
	--------
	>>> import InputGenerator as ig
	>>> L = 128
	>>> InputObj = ig.InputGenerator(L)
	>>> Vfunc = lambda x,y : x*y
	>>> InputObj.setPotential(Vfunc)
	>>> InputObj.writePotential('potential-example-%i.dat' % L)

	For further examples take a look at '<QLB-dir>/input/examples/'
	"""
	L      = 128
	dx     = 1.5
	mass   = 0.1
	delta0 = 14.0
	x      = []
	y      = []
	potentialArray = []
	spinor0Array   = []
	spinor1Array   = []
	spinor2Array   = []
	spinor3Array   = []
	
	potentialFileName = ''
	initialFilename   = ''
	
	def __init__(self, L, dx=1.5625, mass=0.1, delta0=14.0):
		""" Initialize the library

		Initialize the library by calculating the underlyning grid.

			x = dx * (i - 0.5*(L - 1))
			y = dx * (j - 0.5*(L - 1))

		Parameters
		----------
		L : int
			The system will be of size L x L. The system size must hold (L >= 2)
			
		dx : scalar
			The spatial discretization (dx > 0). [default: 1.5625]
			
		mass : scalar
			The mass of each particle (mass > 0). [default: 0.1]
			
		delta0 : scalar
			The initial spread (delta0 > 0). [default: 14.0]

		"""
		if L <= 1:
			self.exitAfterError("system size 'L' must be >= 2")
		self.L = L

		if dx <= 0:
			self.exitAfterError("spatial discretization 'dx' must be > 0")
		self.dx = dx

		if mass <= 0:
			self.exitAfterError("particle mass 'mass' must be > 0")
		self.mass = mass
		
		if delta0 <= 0:
			self.exitAfterError("initial spread 'delta0' must be > 0")
		self.delta0 = delta0

		self.x = np.zeros(L)
		self.y = np.zeros(L)
		
		for i in xrange(0, self.L):
			self.x[i] = self.dx * (i - 0.5*(self.L - 1))
		
		for j in xrange(0, self.L):
			self.y[j] = self.dx * (j - 0.5*(self.L - 1))


	def setPotential(self, Vfunc, *args):
		""" Set the potential 

		Set the potential array by providing an evaluation function	'Vfunc(x,y)'
		where x and y are calculated the following:

			x = dx * (i - 0.5*(L - 1))
			y = dx * (j - 0.5*(L - 1))

		where i and j represent the indices in [0, L). This means the potential
		will be evaluated on the domain:

		[ -dx*0.5*(L-1), dx*0.5*(L-1)] x [ -dx*0.5*(L-1), dx*0.5*(L-1)]

		Parameters
		----------
		Vfunc : callable
			The 2D potential function which will be called as 'V(x,y)' or in case
			additional parameter are specified as 'V(x,y,args)'. 
					
		See Also
		--------
		setPotentialArray

		"""
		self.potentialArray = np.zeros((self.L, self.L))
		
		for i in xrange(0, self.L):
			for j in xrange(0, self.L):
				if args:
					self.potentialArray[i,j] = Vfunc(self.x[i], self.y[j], *args)
				else:
					self.potentialArray[i,j] = Vfunc(self.x[i], self.y[j])

	def setInitial(self, Ifunc, spinor, *args):
		""" Set the initial condition 

		Set the initial condition by providing an evaluation function 'Ifunc(x,y)'
		for the specified spinor where x and y are calculated the following:
	
			x = dx * (i - 0.5*(L - 1))
			y = dx * (j - 0.5*(L - 1))

		where i and j represent the indices in [0, L). This means the initial
		condition will be evaluated on the domain:

		[ -dx*0.5*(L-1), dx*0.5*(L-1)] x [ -dx*0.5*(L-1), dx*0.5*(L-1)]

		Parameters
		----------
		Ifunc : callable
			The 2D initial condition function which will be called as 'Ifunc(x,y)'
			or in case additional parameter are specified as 'Ifunc(x,y,args)'. 
		
		spinor : scalar
			Specify the spinor [0,3] for which the function will be evaluated.
					
		"""
		if spinor == 0:
			self.spinor0Array = np.zeros((self.L, self.L), dtype=np.complex64)
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					if args:
						self.spinor0Array[i,j] = Ifunc(self.x[i], self.y[j], *args)
					else:
						self.spinor0Array[i,j] = Ifunc(self.x[i], self.y[j])
		elif spinor == 1:
			self.spinor1Array = np.zeros((self.L, self.L), dtype=np.complex64)
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					if args:
						self.spinor1Array[i,j] = Ifunc(self.x[i], self.y[j], *args)
					else:
						self.spinor1Array[i,j] = Ifunc(self.x[i], self.y[j])
		elif spinor == 2:
			self.spinor2Array = np.zeros((self.L, self.L), dtype=np.complex64)
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					if args:
						self.spinor2Array[i,j] = Ifunc(self.x[i], self.y[j], *args)
					else:
						self.spinor2Array[i,j] = Ifunc(self.x[i], self.y[j])
		elif spinor == 3:
			self.spinor3Array = np.zeros((self.L, self.L), dtype=np.complex64)
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					if args:
						self.spinor3Array[i,j] = Ifunc(self.x[i], self.y[j], *args)
					else:
						self.spinor3Array[i,j] = Ifunc(self.x[i], self.y[j])
		else:
			exitAfterError("parameter 'spinor' must be in {0,1,2,3}")	


	def setPotentialArray(self, VArray):
		""" Set the potential array 

		Set the potential array by providing an (L x L)-dimensional array
		of the evaluated potential at the grid points:
		
			x(i) = dx * (i - 0.5*(L - 1))
			y(j) = dx * (j - 0.5*(L - 1))
			
		where i and j represent the indices in [0, L). This means:
		
			VArray(i,j) = V( x(i) , y(j) )

		Parameters
		----------
		Varray : ndarray
			Array of size (L x L) with the evaluated potential 
					
		See Also
		--------
		setPotential

		"""
		if VArray.shape != (self.L, self.L):
			self.exitAfterError("dimension mismatch in 'setPotentialArray'")
		self.potentialArray = VArray
	
	def writePotential(self, filename, title='', descrption=''):	
		""" Write the potential to a file 

		Write the potential to 'filename'. The file will have the following
		format:
		
		$BEGIN_INPUT
		$L=L
		$DX=dx
		$MASS=mass
		$DELTA0=delta0
		$BEGIN_POTENTIAL
		V[0, 0]   ...  V[0, L-1]
		   .               .
		   .               .
		   .               .
		V[L-1, 0] ...  V[L-1, L-1]
		$END_POTENTIAL
		$END_INPUT
		
		Everthing outside of the '$BEGIN_INPUT', '$END_INPUT' will be
		treated as comment and ignored by QLB.

		Parameters
		----------
		filename : string
			File name
			
		title : string (optional)
			Title of the file
			
		descrption : string (optional)
			Give a brief descrption of the potential
		
		See Also
		--------
		writeInitial
			
		"""
		if len(self.potentialArray) == 0:
			self.exitAfterError("potential has not been set or is empty")
		else:
			try:
				f = open(filename, 'w')
			except IOError:
				self.exitAfterError("Cannot open file '"+filename+"'")
			
			self.potentialFileName = filename
			
			if title:
				f.write('%s\n\n' % title)
			
			if descrption:
				for d in descrption:
					f.write('%s' % d)
				f.write('\n\n')
			
			programName = ntpath.basename(str(sys.argv[0]))
			f.write("Generated by "+programName+" at "+\
			        strftime("%a, %d %b %Y %H:%M:%S\n\n"))
			
			f.write('$BEGIN_INPUT\n')
			f.write('$L=%i\n' % self.L)
			f.write('$DX=%f\n' % self.dx)
			f.write('$MASS=%f\n' % self.mass)
			f.write('$DELTA0=%f\n' % self.delta0)
			f.write('$BEGIN_POTENTIAL\n')
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					f.write('%15.10f  ' % self.potentialArray[i,j])
				f.write('\n')
			f.write('$END_POTENTIAL\n')
			f.write('$END_INPUT\n')
			f.close()

	def writeInitial(self, filename, title='', descrption=''):	
		""" Write the initial condition to a file 

		Write the initial condition to 'filename'. The file will have the following
		format:
		
		$BEGIN_INPUT
		$L=L
		$DX=dx
		$MASS=mass
		$DELTA0=delta0
		$BEGIN_INITIAL
		spinor0[0, 0]   spinor1[0, 0]   spinor2[0, 0]   spinor3[0, 0]    ... 
		   .                     
		   .                     
		   .                     
		spinor0[L-1, 0] spinor1[L-1, 0] spinor2[L-1, 0] spinor3[L-1, 0]  ...
		$BEGIN_INITIAL
		$END_INPUT
		
		Everthing outside of the '$BEGIN_INPUT', '$END_INPUT' will be
		treated as comment and ignored by QLB.

		Parameters
		----------
		filename : string
			File name
			
		title : string (optional)
			Title of the file
			
		descrption : string (optional)
			Give a brief descrption of the potential
		
		See Also
		--------
		writePotential
			
		"""
		if len(self.spinor0Array) == 0 and len(self.spinor1Array) == 0 and \
		   len(self.spinor2Array) == 0 and len(self.spinor3Array) == 0 :
			self.exitAfterError("no initial condition has been set")
		else:
			try:
				f = open(filename, 'w')
			except IOError:
				self.exitAfterError("Cannot open file '"+filename+"'")
			
			self.initialFileName = filename
			
			if title:
				f.write('%s\n\n' % title)
			
			if descrption:
				for d in descrption:
					f.write('%s' % d)
				f.write('\n\n')
			
			programName = ntpath.basename(str(sys.argv[0]))
			f.write("Generated by "+programName+" at "+\
			        strftime("%a, %d %b %Y %H:%M:%S\n\n"))
			
			f.write('$BEGIN_INPUT\n')
			f.write('$L=%i\n' % self.L)
			f.write('$DX=%f\n' % self.dx)
			f.write('$MASS=%f\n' % self.mass)
			f.write('$DELTA0=%f\n' % self.delta0)
			f.write('$BEGIN_INITIAL\n')
			
			if len(self.spinor0Array) == 0:
				self.spinor0Array = np.zeros((self.L, self.L), dtype=np.complex64)
			if len(self.spinor1Array) == 0:
				self.spinor1Array = np.zeros((self.L, self.L), dtype=np.complex64)
			if len(self.spinor2Array) == 0:
				self.spinor2Array = np.zeros((self.L, self.L), dtype=np.complex64)
			if len(self.spinor3Array) == 0:
				self.spinor3Array = np.zeros((self.L, self.L), dtype=np.complex64)
			
			for i in xrange(0, self.L):
				for j in xrange(0, self.L):
					f.write('(%15.10f,%15.10f)  ' % (self.spinor0Array[i,j].real,\
					                                 self.spinor0Array[i,j].imag))
					f.write('(%15.10f,%15.10f)  ' % (self.spinor1Array[i,j].real,\
					                                 self.spinor1Array[i,j].imag))
					f.write('(%15.10f,%15.10f)  ' % (self.spinor2Array[i,j].real,\
					                                 self.spinor2Array[i,j].imag))
					f.write('(%15.10f,%15.10f)  ' % (self.spinor3Array[i,j].real,\
					                                 self.spinor3Array[i,j].imag))					                                 
				f.write('\n')					                                 
			f.write('$END_INITIAL\n')
			f.write('$END_INPUT\n')
			f.close()
		
	def moveToPotentialDB(self, path, override=False):
		""" Try to move the file to the potential-file DataBase 

		Parameters
		----------
		path : string 
			Path of the DataBase (absolut path)
			
		override : bool (optional)
			If True override any existing files without asking for permision
		
		See Also
		--------
		moveToInitialDB
		"""
		self.moveToDB(path, self.potentialFileName, override)
	
	def moveToInitialDB(self, path, override=False):
		""" Try to move the file to the initial-file DataBase 

		Parameters
		----------
		path : string 
			Path of the DataBase (absolut path)
			
		override : bool (optional)
			If True override any existing files without asking for permision
		
		See Also
		--------
		moveToPotentialDB
		"""
		self.moveToDB(path, self.initialFileName, override)	
		
	def moveToDB(self, path, filename, override=False):	
		""" Implementaion of the moveToDB* functions """
		if path.endswith('/') or path.endswith('\\'):
			dest = path+filename
		else:
			dest = path+'/'+filename

		if os.path.isfile(dest) and not override:
			print "InputGenerator: a file named '"+dest+"' already exists.\n",
			print "Do you want to replace it? [Y/n]",
			
			inputVar = 'n'
			try:
				inputVar = raw_input()
			except KeyboardInterrupt:
				print "\n"
				return

			if inputVar == '' or inputVar.lower() == 'y':
				shutil.copy2(filename, dest)
		else:
			shutil.copy2(filename, dest)

	def exitAfterError(self, errMsg):
		""" Print an error message and exit """
		print "InputGenerator: error: ", errMsg
		sys.exit(1)

