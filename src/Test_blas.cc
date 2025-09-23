/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: tests/core/Test_meson_field.cc

Copyright (C) 2015-2018

Author: Felix Erben <felix.erben@ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/

// clang-format off
#include <Grid/Grid.h>
#include <ProdStagA2Autils.h>
#include <DevStagA2Autils.h>
#include <StagGamma.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
// clang-format on

using namespace Grid;

typedef typename NaiveStaggeredFermionD::ComplexField ComplexField;
typedef typename NaiveStaggeredFermionD::FermionField FermionField;

GRID_SERIALIZABLE_ENUM(MFType, undef, prod, 0, dev, 1);

// clang-format off
class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, 
                                  int, blockSize, 
                                  MFType, mfType, 
                                  bool, symmetric, 
                                  std::string, gammas, 
                                  std::string, writeFile);
};
// clang-format on

int main(int argc, char *argv[]) {
  // initialization
  Grid_init(&argc, &argv);
  MemoryManager::Print();
  std::cout << GridLogMessage << "Grid initialized" << std::endl;

  // Lattice and rng setup
  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplex::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  int Nt = GridDefaultLatt()[Tp];
  Lattice<iScalar<vInteger>> t(&grid);
  LatticeCoordinate(t, Tp);

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  std::string paramFile = argv[1];
  XmlReader reader(paramFile, false, "grid");

  GlobalPar inputParams;
  read(reader, "parameters", inputParams);

  const int devA2Ablocking = DEV_A2A_BLOCKING;

  using VecStag = iVector<iScalar<iScalar<Complex>>, devA2Ablocking>;
  using vVecStag = iVector<iScalar<iScalar<vComplex>>, devA2Ablocking>;
  typedef Lattice<vVecStag> LatticeVecStag;

  LatticeVecStag spinMat(&grid);
  std::vector<VecStag> sliced;
  random(pRNG, spinMat);

  // Gamma matrices used in the contraction
  std::vector<StagGamma::SpinTastePair> gammas =
      StagGamma::ParseSpinTaste(inputParams.gammas);

  int Ngamma = gammas.size();
  std::vector<ComplexField> momGamma(Ngamma, &grid);
  StagGamma spinTaste;

  for (int mu = 0; mu < Ngamma; mu++) {
    momGamma[mu] = 1.0;

    spinTaste.setSpinTaste(gammas[mu]);
    spinTaste.applyPhase(momGamma[mu], momGamma[mu]);
  }
  MomentumProject<LatticeVecStag, ComplexField> MP;
  MP.Allocate(Ngamma, &grid);
  MP.ImportMomenta(momGamma);

  double start, stop, usec;
  start = usecond();
  nvtxRangePushA("Project");
  MP.Project(spinMat, sliced);
  nvtxRangePop();
  stop = usecond();
  usec = stop - start;
  int vol = 1;
  for (int i = 0; i < Nd - 1; i++) {
    vol *= grid.LocalDimensions()[i];
  };
  int Lt = grid.LocalDimensions()[Nd - 1];
  double Fm = (6.0 * Lt * Ngamma * devA2Ablocking) * vol;
  double Fs = (2.0 * Lt * devA2Ablocking) * (vol - 1);

  double flops = (Fm + Fs) / (usec / pow(10.0, 6)) / pow(2.0, 30);

  std::cout << GridLogMessage << "Fm = " << Fm << std::endl;
  std::cout << GridLogMessage << "Fs = " << Fs << std::endl;
  std::cout << GridLogMessage << "block size: " << devA2Ablocking << std::endl;
  std::cout << GridLogMessage << "M(rho,phi) created, execution time " << usec
            << " (us), " << flops << " Gflops" << std::endl;

  Grid_finalize();

  return EXIT_SUCCESS;
}
