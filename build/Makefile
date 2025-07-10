# Once compiled run with
# for mpi:
# mpirun -np 1 ./staggamma --grid 4.4.4.4 
# for scalar:
# ./staggamma --grid 4.4.4.4 

gridinstall=$(HOME)/Dropbox/Physics/projects/gridProjects/Grid/install-scalar
config=$(gridinstall)/bin/grid-config

INCLUDES := -I../src/cpp # Manually specify include dirs
#
#  flags from the grid utility program
#
CXXFLASGS= $(shell $(config) --cxxflags)   -I$(gridinstall)/include
LDFLAGS=$(shell $(config) --ldflags)
LIBS= $(shell $(config) --libs) -L$(gridinstall)/lib  -lGrid
COMPILER=$(shell $(config) --cxx)

original_stag_gamma: ../src/_original_Test_stag_gamma.cc
	$(COMPILER) -O0 $(INCLUDES) -o $@ $(CXXFLASGS)  ../src/_original_Test_stag_gamma.cc  $(LDFLAGS) $(LIBS) 

stag_gamma: ../src/Test_stag_gamma.cc
	$(COMPILER) -O0 $(INCLUDES) -o $@ $(CXXFLASGS)  ../src/Test_stag_gamma.cc  $(LDFLAGS) $(LIBS) 
