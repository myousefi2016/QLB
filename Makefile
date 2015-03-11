# Quantum Lattice Boltzmann
# (c) 2015 Fabian Thüring, ETH Zürich
#
# [DESCRIPTION]
# This Makefile is used to compile the whole project on Linux/Mac OS X.
# 
# === Linux ===
# The compilation relies on the following libraries:
#   -libGL
#   -libGLU
#   -libGLEW
#   -libglut
# All those libraries should be present in the repository of your distribution.
# The library 'libGLEW' can be built with this Makefile by using the command
# 'make libGLEW' this will build the most recent version directly from github.
#
# === Mac OSX ===
# The compilation relies on 'libGLEW' which is not installed by default. You can
# build the library with this Makefile by issuing 'make libGLEW' or install it
# on your own from http://glew.sourceforge.net/.
#
# For further assistance use: 
#	make help

CXX     = clang++
CXX_NV  = g++
NVCC    = nvcc
 
CUDA_DIR = /usr/local/cuda-6.5

# CUDA is not supported yet
NO_CUDA ?= false

# ======================= FINDING LIBRARIES/HEADERS ============================

OS = $(shell uname -s 2>/dev/null)

GLEW_BUILD_DIR = glew
GLEW_GIT_URL   = https://github.com/nigels-com/glew.git

ifeq ($(OS), )
 OS = unknown
endif

# === Find the CUDA libraries ===
CUDA_DIR   ?= /usr/local/cuda
NVCC       ?= $(CUDA_DIR)/bin/nvcc
CUDA_LIB    = -L$(CUDA_DIR)/lib64/ -lcudart

# === Sources ===
EXE         = QLB
SRC_CU      = $(wildcard src/*.cu)
SRC_CPP     = $(wildcard src/*.cpp)
OBJECTS_CU  = $(patsubst src/%.cu, objects/%.o,$(SRC_CU))
OBJECTS_CPP = $(patsubst src/%.cpp,objects/%.o,$(SRC_CPP))
OBJECTS     = $(OBJECTS_CPP)

BIN_PATH    = ./bin/$(OS)
EXE_BIN     = $(BIN_PATH)/$(EXE)

# === Compiler Flags ===
WARNINGS    = -Wall
DEFINES     = 
DEBUG       =
PROFILING   = 
INCLUDE     = -I./inc/$(OS)
OPT         = -O2 -march=native
CUDAOPT     = -O2
CXXSTD      = -std=c++11
CUDAFLAGS   = $(DEBUG) $(INCLUDE) $(CXXSTD) $(CUDAOPT)
CXXFLAGS    = $(DEBUG) $(INCLUDE) $(CXXSTD) $(OPT) $(WARNINGS) $(DEFINES) $(PROFILING)
LDFLAGS     = -L./lib/$(OS) -lGL -lGLU -lglut -lGLEW -lpthread

# === Build adjustments ===

# Adjust to build on Mac OS X
ifeq ($(OS)),Darwin)
 LDFLAGS   = -framework GLUT -framework OpenGL -L./lib/$(OS) -lGLEW -lpthread
 WARNINGS += -Wno-deprecated-declarations -Wno-deprecated-register
endif

# Compiler specific warnings
ifeq ($(CXX),clang++)
 WARNINGS += -Wno-deprecated-register
endif

# Adjust the build to use CUDA
ifdef NO_CUDA
 DEFINES    += -DQLB_NO_CUDA
else
 OBJECTS    += $(OBJECTS_CU)
 LDFLAGS    += $(CUDA_LIB)
endif

# Detect multi display environment in Linux
ifeq ($(OS),Linux)
 ifeq ($(shell xrandr -q 2>/dev/null | grep ' connected' | wc -l),2)
  CXXFLAGS  += -DQLB_MULTI_DISPLAY
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
	cd $(GLEW_BUILD_DIR)/auto && $(MAKE)
	cd $(GLEW_BUILD_DIR) && $(MAKE)
	mkdir -p lib/$(OS)/
	mkdir -p inc/$(OS)/GL/
	cp $(GLEW_BUILD_DIR)/lib/libGLEW.a lib/$(OS)/
	cp $(GLEW_BUILD_DIR)/include/GL/* inc/$(OS)/GL/
	
# === Cleaning ===
.PHONY: clean
clean:
	rm -f $(EXE) objects/*.o *.dat $(EXE_BIN) $(BIN_PATH)/*.dat 
	rm -f analysis.txt *.out
	
.PHONY: cleanall
cleanall : clean
	rm -rf $(GLEW_BUILD_DIR)
	rm -rf inc/$(OS)/GL/
	rm -rf lib/$(OS)/
	
# === Help ===
.PHONY: help
help :
	@echo "Usage: make Target [Options...]"
	@echo ""
	@echo " Target:"
	@echo "    all      - compiles the program (default)"
	@echo "    libGLEW  - build the OpenGL Extension Wrangler Library (libGLEW)"
	@echo "    clean    - removes all execuatbles and object files"
	@echo "    cleanall - like clean but removes libGLEW aswell"
	@echo "    help     - prints this help"
	@echo ""
	@echo " Options:"
	@echo "    NO_CUDA=true"
	@echo "       This flag will disable compilation against CUDA"
	@echo ""
	@echo " Example:"
	@echo "   make NO_CUDA=true"
	@echo ""
