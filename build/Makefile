# Once compiled run with
# for mpi:
# mpirun -np 1 ./staggamma --grid 4.4.4.4 
# for scalar:
# ./staggamma --grid 4.4.4.4 

gridinstall=$(HOME)/HadronsMILC/Grid/install-perlmutter
config=$(gridinstall)/bin/grid-config

INCLUDES := -I../src/cpp # Manually specify include dirs
#
#  flags from the grid utility program
#
CXXFLASGS= $(shell $(config) --cxxflags)   -I$(gridinstall)/include
LDFLAGS=$(shell $(config) --ldflags)
LIBS= $(shell $(config) --libs) -L$(gridinstall)/lib  -lGrid
COMPILER=$(shell $(config) --cxx)

meson_field.o: ../src/Test_meson_field.cc
	$(COMPILER) -O3 $(INCLUDES) -c -o $@ -g -lineinfo $(CXXFLASGS)  ../src/Test_meson_field.cc  $(LDFLAGS) $(LIBS) 

Test_meson_field: 
	nvcc -O3 $(INCLUDES) -o Test_meson_field meson_field.o $(CXXFLASGS)  $(LDFLAGS) $(LIBS) 

my_meson_field: ../src/Test_my_meson_field.cc
	$(COMPILER) -O3 $(INCLUDES) -o $@ -g -lineinfo -lnvToolsExt $(CXXFLASGS)  ../src/Test_my_meson_field.cc  $(LDFLAGS) $(LIBS) 

meson_field: ../src/Test_meson_field.cc
	$(COMPILER) -O3 $(INCLUDES) -o $@ -g -lineinfo -lnvToolsExt $(CXXFLASGS)  ../src/Test_meson_field.cc  $(LDFLAGS) $(LIBS) 

stag_gamma: ../src/Test_stag_gamma.cc
	$(COMPILER) -O0 $(INCLUDES) -o $@ $(CXXFLASGS)  ../src/Test_stag_gamma.cc  $(LDFLAGS) $(LIBS) 

stag_a2a: ../src/Test_meson_field.cc
	$(COMPILER) -O3 $(INCLUDES) -o $@ $(CXXFLASGS)  ../src/Test_meson_field.cc  $(LDFLAGS) $(LIBS) 
