# Quantum Lattice Boltzmann
# (c) 2015 Fabian Thüring, ETH Zürich
#
# [DESCRIPTION]
# This Makefile is used to compile the whole project on Linux/Mac OS X.
# 
#  ==== Linux ====
# The compilation relies on the following libraries:
#   -libGL
#   -libGLU
#   -libGLEW
#   -libglut
# All those libraries should be present in the repository of your distribution.
# The library 'libGLEW' can be built with this Makefile by using the command
# 'make libGLEW' this will build the most recent version directly from github.
#
# Note: Ubuntu 14.04/14.10 might suffer from a linker regression that is
#       exposed when linking against the system OpenGL library (Mesa) but
#       running the application against the NVIDIA one.   
#       The bug can be fixed by directly linking against the OpenGL library from 
#       NVIDIA by adding '-L/usr/lib/nvidia-346/' to the LDFLAGS.
#       See https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-319/+bug/1248642
#
#  ==== Mac OSX ====
# The compilation relies on 'libGLEW' which is not installed by default. You can
# build the library with this Makefile with 'make libGLEW' or install it
# on your own from http://glew.sourceforge.net/.
#
# For further assistance use: 
#   make help
#

# Compiler
CXX        = g++
CXX_NV     = g++
NVCC       = nvcc

# Use CUDA? 
CUDA      ?= true

# Location of the CUDA Toolkit (modify if necessary)
CUDA_DIR  = /usr/local/cuda

# Enable fast-math?
FAST_MATH ?= false

# ======================= FINDING LIBRARIES/HEADERS ============================

OS = $(shell uname -s 2>/dev/null)

GLEW_BUILD_DIR = glew
GLEW_GIT_URL   = https://github.com/nigels-com/glew.git

ifeq ($(OS), )	
 OS = unknown
endif

# === Find the CUDA libraries ===
CUDA_DIR    ?= /usr/local/cuda

NVCC        ?= $(CUDA_DIR)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_DIR)/include

ifeq ($(OS),Darwin)
 CUDA_LIB    = -L$(CUDA_DIR)/lib/ -lcudart
else
 CUDA_LIB    = -L$(CUDA_DIR)/lib64/ -lcudart
endif

# === Sources ===
EXE          = QLB
SRC_CU       = $(wildcard src/*.cu)
SRC_CXX      = $(wildcard src/*.cpp)
OBJECTS_CU   = $(patsubst src/%.cu, objects/%.o,$(SRC_CU))
OBJECTS_CXX  = $(patsubst src/%.cpp,objects/%.o,$(SRC_CXX))
OBJECTS      = $(OBJECTS_CXX)

BIN_PATH     = ./bin/$(OS)
EXE_BIN      = $(BIN_PATH)/$(EXE)

# === Compiler Flags ===
WARNINGS     = -Wall
DEFINES      =
DEBUG        =  
PROFILING    = 
INCLUDE      = -I./include/$(OS)
OPT          = -O2 -march=native
CUDAOPT      = -O2 #-ptxas=-v
CXXSTD       = -std=c++11
CUDAFLAGS    = $(DEBUG) $(INCLUDE) $(CXXSTD) $(DEFINES) $(CUDAOPT)
CXXFLAGS     = $(DEBUG) $(INCLUDE) $(CXXSTD) $(DEFINES) $(OPT) $(WARNINGS) $(PROFILING)
LDFLAGS      = -L./lib/$(OS) -lGL -lGLU -lglut -lGLEW -lpthread

# === Build adjustments ===

# Adjust to build on Mac OS X (link against OpenGL framework)
ifeq ($(OS),Darwin)
 LDFLAGS    = -framework GLUT -framework OpenGL -L./lib/$(OS) -lGLEW -lpthread
 WARNINGS  += -Wno-deprecated-declarations -Wno-deprecated-register
 DEFINES   += -DQLB_CUDA_GL_WORKAROUND
endif

ifeq ($(FAST_MATH),true)
 OPT       += -ffast-math
 CUDAOPT   += --use_fast_math
endif

# Clang specific flags
ifeq ($(CXX),clang++)
 WARNINGS  += -Wno-deprecated-register
endif

# Adjust the build to use CUDA
ifeq ($(CUDA),true)
 DEFINES   += -DQLB_HAS_CUDA
 OBJECTS   += $(OBJECTS_CU)
 LDFLAGS   += $(CUDA_LIB)
 INCLUDE   += $(CUDA_INCLUDE)
endif

# Adjust the build to not use the GPU PerformanceCounter
ifneq ($(GPUCOUNTER),false)
 ifeq ($(CUDA),true)
  DEFINES  += -DQLB_HAS_GPU_COUNTER
 endif
endif

# Detect multi display environment in Linux
ifeq ($(OS),Linux)
 ifeq ($(shell xrandr -q 2>/dev/null | grep ' connected' | wc -l),2)
  CXXFLAGS += -DQLB_MULTI_DISPLAY
 endif
endif

# =============================== COMPILATION ==================================

.PHONY: all
all: $(EXE_BIN)

$(OBJECTS) : | objects

objects :
	mkdir -p $@

# Compile c++
objects/%.o : src/%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<
	
# Compile CUDA
objects/%.o : src/%.cu
	$(NVCC) -ccbin $(CXX_NV) -c $(CUDAFLAGS) -o $@ $<

# Link
$(EXE) : $(OBJECTS)
	$(CXX) -o $(EXE) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS)

# Copy to bin
$(EXE_BIN) : $(EXE)
	mkdir -p $(BIN_PATH) 
	cp $< $@
	
# build libGLEW
libGLEW :
	$(info === Building OpenGL Extension Wrangler Library === )
	rm -rf $(GLEW_BUILD_DIR)
	git clone $(GLEW_GIT_URL) $(GLEW_BUILD_DIR)
	$(MAKE) -C $(GLEW_BUILD_DIR)/auto
	$(MAKE) -C $(GLEW_BUILD_DIR)
	mkdir -p lib/$(OS)/
	mkdir -p include/$(OS)/GL/
	cp $(GLEW_BUILD_DIR)/lib/libGLEW.a lib/$(OS)/
	cp $(GLEW_BUILD_DIR)/include/GL/* include/$(OS)/GL/
	
# === Cleaning ===
.PHONY: clean
clean:
	rm -f $(EXE) objects/*.o *.dat $(EXE_BIN) $(BIN_PATH)/*.dat 
	rm -f analysis.txt *.out
	
.PHONY: cleanall
cleanall : clean
	rm -rf $(GLEW_BUILD_DIR)
	rm -rf include/$(OS)/GL/
	rm -rf lib/$(OS)/
	
# === Help ===
.PHONY: help
help :
	@echo "Usage: make Target [Options...]"
	@echo ""
	@echo " Target:"
	@echo "    all      - compiles the program [default]"
	@echo "    libGLEW  - build the OpenGL Extension Wrangler Library (libGLEW)"
	@echo "    clean    - removes all execuatbles and object files"
	@echo "    cleanall - like clean but removes libGLEW aswell"
	@echo "    help     - prints this help"
	@echo ""
	@echo " Options:"
	@echo "    CUDA={true|false}"
	@echo "       This flag will enable or disable CUDA support [default : true]"
	@echo ""
	@echo "    GPUCOUNTER={true|false}"
	@echo "       This flag will enable or disable the GPU performance counter"
	@echo "       (Only affects Linux and CUDA builds) [default : true]"
	@echo ""
	@echo "    FAST_MATH={true|false}"
	@echo "       This flag will enable or disable fast-math operations which break"
	@echo "       ANSI or IEEE rules for speed (potentially dangerous) [default : false]"
	@echo ""
	@echo " Example:"
	@echo "   make CUDA=false"
	@echo ""
