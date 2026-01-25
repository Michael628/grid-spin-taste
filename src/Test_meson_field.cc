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
#include <DevStagA2AutilsBuggy.h>
#include <DevStagA2AutilsMat.h>
#include <DevStagA2AutilsFlatMat.h>
#include <DevStagA2AutilsVector.h>
#include <StagGamma.h>
// #include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
// clang-format on

using namespace Grid;

typedef typename NaiveStaggeredFermionD::ComplexField ComplexField;
typedef typename NaiveStaggeredFermionD::FermionField FermionField;

GRID_SERIALIZABLE_ENUM(MFType, undef, prod, 0, buggy, 1, mat, 2, flatmat, 3,
                       vector, 4);
GRID_SERIALIZABLE_ENUM(SourceType, undef, random, 0, point, 1);

// clang-format off
class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, 
                                  int, blockSize, 
                                  MFType, mfType, 
                                  bool, symmetric,
                                  SourceType, sourceType,
                                  std::vector<int>, phiSource,  // [x,y,z,t] for lhs point source
                                  std::vector<int>, rhoSource,  // [x,y,z,t] for rhs point source
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

  std::cout << GridLogMessage << "Allocating vectors" << std::endl;
  int blockSize = inputParams.blockSize;
  std::vector<FermionField> phi(blockSize, &grid);
  std::vector<FermionField> rho(0, &grid);

  // Check if sourceType is set, default to random if not specified
  SourceType sourceType = SourceType::random;
  if (inputParams.sourceType != SourceType::undef) {
    sourceType = inputParams.sourceType;
  }

  if (sourceType == SourceType::point) {
    // Default to origin if not specified
    Coordinate phiLoc = (inputParams.phiSource.size() == 4)
                            ? Coordinate(inputParams.phiSource)
                            : Coordinate(4, 0);
    Coordinate rhoLoc = (inputParams.rhoSource.size() == 4)
                            ? Coordinate(inputParams.rhoSource)
                            : Coordinate(4, 0);

    std::cout << GridLogMessage << "Initialising point source fields"
              << std::endl;
    std::cout << GridLogMessage << "  phi source at: (" << phiLoc[0] << ","
              << phiLoc[1] << "," << phiLoc[2] << "," << phiLoc[3] << ")"
              << std::endl;

    // Create point source for phi vectors
    for (unsigned int i = 0; i < blockSize; ++i) {
      phi[i] = Zero();

      // Create point source: set value to 1.0 at specified location, zero
      // elsewhere We set it for all color/spin components
      typename FermionField::scalar_object one_src;
      one_src = 1.0;
      pokeSite(one_src, phi[i], phiLoc);
    }

    if (!inputParams.symmetric) {
      std::cout << GridLogMessage << "  rho source at: (" << rhoLoc[0] << ","
                << rhoLoc[1] << "," << rhoLoc[2] << "," << rhoLoc[3] << ")"
                << std::endl;
      rho.resize(blockSize, &grid);
      for (unsigned int i = 0; i < blockSize; ++i) {
        rho[i] = Zero();
        typename FermionField::scalar_object one_src;
        one_src = 1.0;
        pokeSite(one_src, rho[i], rhoLoc);
      }
    }
  } else {
    std::cout << GridLogMessage << "Initialising random meson fields"
              << std::endl;

    for (unsigned int i = 0; i < blockSize; ++i) {
      random(pRNG, phi[i]);
    }

    if (!inputParams.symmetric) {
      rho.resize(blockSize, &grid);
      for (unsigned int i = 0; i < blockSize; ++i) {
        random(pRNG, rho[i]);
      }
    }
  }

  // Gamma matrices used in the contraction
  std::vector<StagGamma::SpinTastePair> spinTastes =
      StagGamma::ParseSpinTaste(inputParams.gammas);

  // momentum phases e^{ipx}
  std::vector<std::vector<double>> momenta = {{0., 0., 0.}};
  // std::vector<std::vector<double>> momenta = {
  //     {0., 0., 0.}, {1., 0., 0.}, {-1., 0., 0.}, {0, 1., 0.},  {0, -1., 0.},
  //     {0, 0, 1.},   {0, 0, -1.},  {1., 1., 0.},  {1., 1., 1.}, {2., 0., 0.}};
  std::cout << GridLogMessage << "Meson fields will be created for "
            << spinTastes.size() << " Gamma matrices and " << momenta.size()
            << " momenta." << std::endl;

  std::cout << GridLogMessage << "Computing complex phases" << std::endl;
  std::vector<ComplexField> phases(momenta.size(), &grid);
  ComplexField coor(&grid);
  Complex Ci(0.0, 1.0);
  for (unsigned int j = 0; j < momenta.size(); ++j) {
    phases[j] = Zero();
    for (unsigned int mu = 0; mu < momenta[j].size(); mu++) {
      LatticeCoordinate(coor, mu);
      phases[j] = phases[j] + momenta[j][mu] / GridDefaultLatt()[mu] * coor;
    }
    phases[j] = exp((Real)(2 * M_PI) * Ci * phases[j]);
  }
  std::cout << GridLogMessage << "Computing complex phases done." << std::endl;

  Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> Mpp(
      momenta.size(), spinTastes.size(), Nt, blockSize, blockSize);

  // timer
  double start, stop;

  /////////////////////////////////////////////////////////////////////////
  // execute meson field routine
  /////////////////////////////////////////////////////////////////////////
  auto worker = A2AWorkerLocal<StaggeredImplR>(&grid, {}, spinTastes, Tp);

  std::cout << GridLogMessage << "Run is "
            << (inputParams.symmetric ? "symmetric" : "not symmetric")
            << std::endl;

  start = usecond();
  nvtxRangePushA("Grid utils");
  // cudaProfilerStart();
  std::vector<FermionField> &rho_ref = phi;
  if (!inputParams.symmetric) {
    rho_ref = rho;
  }
  std::cout << GridLogMessage << "Found: " << inputParams.mfType << std::endl;
  switch (inputParams.mfType) {
  case MFType::prod:
    std::cout << GridLogMessage << "Running Production MesonField" << std::endl;
    worker.StagMesonField(Mpp, &rho_ref[0], nullptr, &phi[0], nullptr);
    break;
  case MFType::buggy:
    std::cout << GridLogMessage
              << "Running Broken (Momentum Project) MesonField" << std::endl;
    DevA2AutilsBuggy<StaggeredImplR>::MesonField(Mpp, rho_ref, phi, spinTastes,
                                                 phases, Tp);
    break;
  case MFType::mat:
    std::cout << GridLogMessage << "Running MatObj MesonField" << std::endl;
    DevA2AutilsMat<StaggeredImplR>::MesonField(Mpp, rho_ref, phi, spinTastes,
                                               phases, Tp);
    break;
  case MFType::flatmat:
    std::cout << GridLogMessage << "Running Flattened loop MatObj MesonField"
              << std::endl;
    DevA2AutilsFlatMat<StaggeredImplR>::MesonField(Mpp, rho_ref, phi,
                                                   spinTastes, phases, Tp);
    break;
  case MFType::vector:
    std::cout << GridLogMessage << "Running VecObj MesonField" << std::endl;
    DevA2AutilsVector<StaggeredImplR>::MesonField(Mpp, rho_ref, phi, spinTastes,
                                                  phases, Tp);
    break;
  default:
    std::cout << GridLogMessage << "Unknown MFType" << std::endl;
    break;
  }
  // cudaProfilerStop();
  nvtxRangePop();
  stop = usecond();
  std::cout << GridLogMessage << "M(rho,phi) created, execution time "
            << stop - start << " us" << std::endl;
  std::string FileName = inputParams.writeFile;
#ifdef HAVE_HDF5
  using Default_Reader = Grid::Hdf5Reader;
  using Default_Writer = Grid::Hdf5Writer;
  FileName.append(".h5");
#else
  using Default_Reader = Grid::BinaryReader;
  using Default_Writer = Grid::BinaryWriter;
  FileName.append(".bin");
#endif
  {
    Default_Writer w(FileName);
    write(w, "MesonField", Mpp);
  }
  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
