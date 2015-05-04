%  Quantum Lattice Boltzmann 
%  (c) 2015 Fabian Thüring, ETH Zürich
% 
%  This class provides the functionality to generate initial conditions 
%  and potentials for the program QLB.
%  For an implemenation in Python see 'InputGenerator.py'
%
%  Examples
%  --------
%  >>> L = 128
%  >>> ig = InputGenerator(L)
%  >>> Vfunc = @(x,y) x*y
%  >>> ig = setPotential(ig, Vfunc);
%  >>> ig = writePotential(ig, 'myPotential.dat'); 
%
%  For further examples take a look at '<QLB-dir>/input/examples/'

classdef InputGenerator
	%INPUTGENERATOR
	%  This class provides the functionality to generate initial conditions 
	%  and potentials for the program QLB. 
	%
	%  Examples
	%  --------
	%  >>> L = 128
	%  >>> ig = InputGenerator(L);
	%  >>> Vfunc = @(x,y) x*y;
	%  >>> ig = setPotential(ig, Vfunc);
	%  >>> ig = writePotential(ig, 'myPotential.dat'); 
	%
	%  For further examples take a look at '<QLB-dir>/input/examples/'
	%
	properties
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
			
		potentialFilename = ''
		initialFilename   = ''
	end
	
	methods
		function obj=InputGenerator(L, dx, mass, delta0)
			%INPUTGGENERATOR Initialize the library
			% 
			% 	Initialize the library by calculating the underlyning grid.
			% 
			% 		x = dx * (i - 0.5*(L - 1))
			% 		y = dx * (j - 0.5*(L - 1))
			% 
			% 	Parameters
			% 	----------
			% 	L : int
			% 		The system will be of size L x L. The system size must 
			%		hold (L >= 2)
			% 
			% 	dx : scalar
			% 		The spatial discretization (dx > 0). [default: 1.5625]
			% 
			% 	mass : scalar
			% 		The mass of each particle (mass > 0). [default: 0.1]
			% 
			% 	delta0 : scalar
			% 		The initial spread (delta0 > 0). [default: 14.0]
			% 
			if L <= 1 
				error('system size ''L'' must be >= 2')
			end
			obj.L = L;
			
			if nargin > 1
				if dx <= 0 
					error('spatial discretization ''dx'' must be > 0')
				end
				obj.dx = dx;
			end
			
			if nargin  > 2 
				if(mass <= 0)
					error('particle mass ''mass'' must be > 0')
				end
				obj.mass = mass;
			end
			
			if nargin  > 3 
				if(delta0 <= 0)
					error('initial spread ''delta0'' must be > 0')
				end
				obj.delta0 = delta0;	
			end
		
			obj.x = zeros(L,1);
			obj.y = zeros(L,1);
			
			obj.x(:,1) = dx .* ((1:L)' - 0.5 * (L - 1));
			obj.y(:,1) = dx .* ((1:L)' - 0.5 * (L - 1));
		end
		
		function obj=setPotential(obj, Vfunc, varargin)
			%SETPOTENTIAL Set the potential 
			% 
			% 	Set the potential array by providing an evaluation function	
			%	'Vfunc(x,y)' where x and y are calculated the following:
			% 
			% 		x = dx * (i - 0.5*(L - 1))
			% 		y = dx * (j - 0.5*(L - 1))
			% 
			% 	where i and j represent the indices in [0, L). This means 
			%	the potential will be evaluated on the domain:
			% 
			% 	[ -dx*0.5*(L-1), dx*0.5*(L-1)] x [ -dx*0.5*(L-1), dx*0.5*(L-1)]
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	Vfunc : callable
			% 		The 2D potential function which will be called as 'V(x,y)' 
			%		or in case additional parameter are specified as 
			%		'V(x,y,args)'. 
			% 
			% 	See Also
			% 	--------
			% 	setPotentialArray, setInitial
			%
			if ~isa(obj,'InputGenerator') 
				error('first argument must be of type ''InputGenerator''');
			end
			
			obj.potentialArray = zeros(obj.L,obj.L);
			
			for i=1:obj.L
				for j=1:obj.L
					if isempty(varargin)
						obj.potentialArray(i,j) = Vfunc(obj.x(i),obj.y(j)); 
					else
						obj.potentialArray(i,j) = Vfunc(obj.x(i),obj.y(j),...
							                            varargin);
					end
				end
			end
		end
		
		function obj=setPotentialArray(obj, VArray)
			%SETPOTENTIALARRAY Set the potential array 
			% 
			% 	Set the potential array by providing an (L x L)-dimensional 
			%	array of the evaluated potential at the grid points:
			% 
			% 		x(i) = dx * (i - 0.5*(L - 1))
			% 		y(j) = dx * (j - 0.5*(L - 1))
			% 
			% 	where i and j represent the indices in [0, L). This means:
			% 
			% 		VArray(i,j) = V( x(i) , y(j) )
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	Varray : ndarray
			% 		Array of size (L x L) with the evaluated potential 
			% 
			% 	See Also
			% 	--------
			% 	setPotential
			%
			if(~isa(obj,'InputGenerator'))
				error('first argument must be of type ''InputGenerator''');
			end
			
			sz = size(VArray);
			if sz(1) ~= obj.L && sz(2) ~= obj.L 
				error('dimension mismatch in ''setPotentialArray''');
			end

			obj.potentialArray = VArray;
		end
		
		function obj=setInitial(obj, Ifunc, spinor, varargin)
			%SETINITIAL Set the initial condition 
			% 
			% 	Set the initial condition by providing an evaluation 
			%	function 'Ifunc(x,y)' for the specified spinor where x and 
			%	y are calculated the following:
			% 
			% 		x = dx * (i - 0.5*(L - 1))
			% 		y = dx * (j - 0.5*(L - 1))
			% 
			% 	where i and j represent the indices in [0, L). This means 
			%	the initial condition will be evaluated on the domain:
			% 
			% 	[ -dx*0.5*(L-1), dx*0.5*(L-1)] x [ -dx*0.5*(L-1), dx*0.5*(L-1)]
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	Ifunc : callable
			% 		The 2D initial condition function which will be called as 
			%		'Ifunc(x,y)' or in case additional parameter are specified 
			%		as 'Ifunc(x,y,args)'. 
			% 
			% 	spinor : scalar
			% 		Specify the spinor [0,3] for which the function will be 
			%		evaluated.
			% 
			% 	See Also
			% 	--------
			% 	setPotentialArray, setInitial
			%
			if spinor == 0
				obj.spinor0Array = complex(zeros(obj.L, obj.L));
				for i=1:obj.L
					for j=1:obj.L
						if isempty(varargin)
							obj.spinor0Array(i,j) = Ifunc(obj.x(i),obj.y(j)); 
						else
							obj.spinor0Array(i,j) = Ifunc(obj.x(i),obj.y(j),...
															varargin);
						end
					end
				end
			elseif spinor == 1
				obj.spinor1Array = complex(zeros(obj.L, obj.L));
				for i=1:obj.L
					for j=1:obj.L
						if isempty(varargin)
							obj.spinor1Array(i,j) = Ifunc(obj.x(i),obj.y(j)); 
						else
							obj.spinor1Array(i,j) = Ifunc(obj.x(i),obj.y(j),...
															varargin);
						end
					end
				end
			elseif spinor == 2
				obj.spinor2Array = complex(zeros(obj.L, obj.L));
				for i=1:obj.L
					for j=1:obj.L
						if isempty(varargin)
							obj.spinor2Array(i,j) = Ifunc(obj.x(i),obj.y(j)); 
						else
							obj.spinor2Array(i,j) = Ifunc(obj.x(i),obj.y(j),...
															varargin);
						end
					end
				end
			elseif spinor == 3
				obj.spinor3Array = complex(zeros(obj.L, obj.L));
				for i=1:obj.L
					for j=1:obj.L
						if isempty(varargin)
							obj.spinor3Array(i,j) = Ifunc(obj.x(i),obj.y(j)); 
						else
							obj.spinor3Array(i,j) = Ifunc(obj.x(i),obj.y(j),...
															varargin);
						end
					end
				end
			else
				error('parameter ''spinor'' must be in {0,1,2,3}');
			end
		end

		function obj=writePotential(obj, filename, title, descrption)	
			%WRITEPOTENTIAL Write the potential to a file 
			% 
			% 	Write the potential to 'filename'. The file will have the 
			%	following format:
			% 
			% 	$BEGIN_INPUT
			% 	$L=L
			% 	$DX=dx
			% 	$MASS=mass
			% 	$DELTA0=delta0
			% 	$BEGIN_POTENTIAL
			% 	V[0, 0]   ...  V[0, L-1]
			% 	   .               .
			% 	   .               .
			% 	   .               .
			% 	V[L-1, 0] ...  V[L-1, L-1]
			% 	$END_POTENTIAL
			% 	$END_INPUT
			% 
			% 	Everthing outside of the '$BEGIN_INPUT', '$END_INPUT' will 
			%	be treated as comment and ignored by QLB.
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	filename : string
			% 		File name
			% 
			% 	title : string (optional)
			% 		Title of the file
			% 
			% 	descrption : string (optional)
			% 		Give a brief descrption of the potential
			%
			% 	See Also
			% 	--------
			% 	writeInitial
			% 
			if ~isa(obj,'InputGenerator') 
				error('first argument must be of type ''InputGenerator''');
			end
			
			if isempty(obj.potentialArray) 
				error('potential has not been set or is empty');
			end
			
			f = fopen(filename, 'w'); 
			if f < 0 
				error('Cannot open file ''%s''', filename);
			end

			obj.potentialFilename = filename;

			if nargin > 2 
				fprintf(f, '%s\n\n', title);
			end
			
			if nargin > 3 
				for i=1:length(descrption)
					fprintf(f, '%s', descrption(i));
				end
				fprintf(f, '\n\n');
			end
			
 			fprintf(f,'$BEGIN_INPUT\n');
 			fprintf(f,'$L=%i\n', obj.L);
 			fprintf(f,'$DX=%f\n', obj.dx);
 			fprintf(f,'$MASS=%f\n', obj.mass);
 			fprintf(f,'$DELTA0=%f\n', obj.delta0);
 			fprintf(f,'$BEGIN_POTENTIAL\n');
			p = waitbar(0, '1', 'Name', 'Writing to file ...',...
				       'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
 			for i=1:obj.L
				if getappdata(p,'canceling')
					delete(p);
					fclose(f);
					delete(obj.potentialFilename);
					error('User interrupt');
				end
				
 				for j=1:obj.L
 					fprintf(f,'%15.10f  ', obj.potentialArray(i,j));
				end
				fprintf(f,'\n');
				waitbar(i/obj.L, p, sprintf('%3.0f %%',100*i/obj.L));
			end
			delete(p);
 			fprintf(f,'$END_POTENTIAL\n');
 			fprintf(f,'$END_INPUT\n');
			
			fclose(f);
		end
		
		function obj=writeInitial(obj, filename, title, descrption)	
			%WRITEINITIAL Write the initial condition to a file 
			% 
			% 	Write the initial condition to 'filename'. The file will 
			%	have the following format:
			% 
			% 	$BEGIN_INPUT
			% 	$L=L
			% 	$DX=dx
			% 	$MASS=mass
			% 	$DELTA0=delta0
			% 	$BEGIN_INITIAL
			% 	spinor0[0, 0]   spinor1[0, 0]   spinor2[0, 0]   spinor3[0, 0]    ... 
			% 	   .                     
			% 	   .                     
			% 	   .                     
			% 	spinor0[L-1, 0] spinor1[L-1, 0] spinor2[L-1, 0] spinor3[L-1, 0]  ...
			% 	$BEGIN_INITIAL
			% 	$END_INPUT
			% 
			% 	Everthing outside of the '$BEGIN_INPUT', '$END_INPUT' will 
			%	be treated as comment and ignored by QLB.
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	filename : string
			% 		File name
			% 
			% 	title : string (optional)
			% 		Title of the file
			% 
			% 	descrption : string (optional)
			% 		Give a brief descrption of the potential
			%
			% 	See Also
			% 	--------
			% 	writePotential
			%
			if ~isa(obj,'InputGenerator') 
				error('first argument must be of type ''InputGenerator''');
			end
			
			if(	isempty(obj.spinor0Array) && isempty(obj.spinor1Array) && ...
				isempty(obj.spinor2Array) && isempty(obj.spinor3Array))
				error('no initial condition has been set');
			end
			
			f = fopen(filename, 'w'); 
			if(f < 0)
				error('Cannot open file ''%s''', filename);
			end

			obj.initialFilename = filename;

			if nargin > 2
				fprintf(f, '%s\n\n', title);
			end
			
			if nargin > 3 
				for i=1:length(descrption)
					fprintf(f, '%s', descrption(i));
				end
				fprintf(f, '\n\n');
			end
			
 			fprintf(f,'$BEGIN_INPUT\n');
 			fprintf(f,'$L=%i\n', obj.L);
 			fprintf(f,'$DX=%f\n', obj.dx);
 			fprintf(f,'$MASS=%f\n', obj.mass);
 			fprintf(f,'$DELTA0=%f\n', obj.delta0);
 			fprintf(f,'$BEGIN_INITIAL\n');
			
			if isempty(obj.spinor0Array)
				obj.spinor0Array = complex(zeros(obj.L, obj.L));
			end
			if isempty(obj.spinor1Array)
				obj.spinor1Array = complex(zeros(obj.L, obj.L));
			end
			if isempty(obj.spinor2Array)
				obj.spinor2Array = complex(zeros(obj.L, obj.L));
			end
			if isempty(obj.spinor3Array)
				obj.spinor3Array = complex(zeros(obj.L, obj.L));
			end
			
			p = waitbar(0, '1', 'Name', 'Writing to file ...',...
				       'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
 			for i=1:obj.L
				if getappdata(p,'canceling')
					delete(p);
					fclose(f);
					delete(obj.initialFilename);
					error('User interrupt');
				end
				
 				for j=1:obj.L
 					fprintf(f,'(%15.10f,%15.10f)  ', ...
						    real(obj.spinor0Array(i,j)),...
							imag(obj.spinor0Array(i,j)));
					fprintf(f,'(%15.10f,%15.10f)  ', ...
						    real(obj.spinor1Array(i,j)),...
							imag(obj.spinor1Array(i,j)));
 					fprintf(f,'(%15.10f,%15.10f)  ', ...
						    real(obj.spinor2Array(i,j)),...
							imag(obj.spinor2Array(i,j)));
 					fprintf(f,'(%15.10f,%15.10f)  ', ...
						    real(obj.spinor3Array(i,j)),...
							imag(obj.spinor3Array(i,j)));						
				end
				fprintf(f,'\n');
				waitbar(i/obj.L, p, sprintf('%3.0f %%',100*i/obj.L));
			end
			delete(p);
 			fprintf(f,'$END_INITIAL\n');
 			fprintf(f,'$END_INPUT\n');
			
			fclose(f);
		end
		
		function obj=moveToPotentialDB(obj, path, override)
			%MOVETOPOTENTIALDB Try to move the file to the potential-file DB
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	path : string 
			% 		Path of the DataBase (relative to the script location)
			% 
			% 	override : bool (optional)
			% 		If True override any existing files without asking for 
			%		permision
			%
			%	See Also
			%	--------
			%	moveToInitialDB
			%
			if nargin < 3, override=false; end
 			obj.moveToDB(path, obj.potentialFilename, override);
		end
		
		function obj=moveToInitialDB(obj, path, override)
			%MOVETOINITIALDB Try to move the file to the initial-file DB
			% 
			% 	Parameters
			% 	----------
			%	obj : InputGenerator
			%		InputGenerator class object
			%
			% 	path : string 
			% 		Path of the DataBase (absolut path)
			% 
			% 	override : bool (optional)
			% 		If True override any existing files without asking for 
			%		permision
			%
			%	See Also
			%	--------
			%	moveToPotentialDB
			%
			if nargin < 3, override=false; end
 			obj.moveToDB(path, obj.initialFilename, override);
		end
		
		function moveToDB(obj, path, filename, override)	
			%MOVETODB Implementaion of the moveToDB* functions
			%
			if strcmp(path(end), '/') || strcmp(path(end), '\')  
				dest = strcat(path, filename);
			else
				dest = sprintf('%s/%s', path, filename);
			end
			
			if exist(dest, 'file') == 2 && ~override
				fprintf('InputGenerator: a file named ''%s'' already exists.\n',...
					    dest)
				inputVar = input('Do you want to replace it? [Y/n] ','s');
				
				if strcmp(inputVar,'') || strcmpi(inputVar,'y')
					copyfile(filename, dest);
				end
			else
				copyfile(filename, dest);
			end
		end
	end
end

