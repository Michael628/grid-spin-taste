# Once compiled run with
# for mpi:
# mpirun -np 1 ./staggamma --grid 4.4.4.4 
# for scalar:
# ./staggamma --grid 4.4.4.4 

gridinstall=$(HOME)/work/Grid/install-mpi
config=$(gridinstall)/bin/grid-config

INCLUDES := -I./src/cpp # Manually specify include dirs
#
#  flags from the grid utility program
#
CXXFLASGS= $(shell $(config) --cxxflags)   -I$(gridinstall)/include
LDFLAGS=$(shell $(config) --ldflags)
LIBS= $(shell $(config) --libs) -L$(gridinstall)/lib  -lGrid
COMPILER=$(shell $(config) --cxx)

stag_a2a: src/Test_meson_field.cc
	$(COMPILER) -g $(INCLUDES) -o $@ $(CXXFLASGS)  src/Test_meson_field.cc  $(LDFLAGS) $(LIBS) 
