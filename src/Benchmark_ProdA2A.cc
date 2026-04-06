/*************************************************************************************
    Benchmark for production A2A meson field kernel (ProdStagA2Autils).

    Tests full-volume local contraction performance, sweeping:
      - local volume (via --grid or internal sweep)
      - number of left/right vectors (equal)
      - number of gamma phases

    Usage:
      ./benchmark_proda2a --grid L.L.L.L
    or without --grid to sweep hardcoded lattice sizes.
*************************************************************************************/

#include <Grid/Grid.h>
#include <ProdStagA2Autils.h>

using namespace Grid;

typedef ImprovedStaggeredFermionD FImpl;
typedef typename FImpl::ComplexField ComplexField;
typedef typename FImpl::FermionField FermionField;
typedef typename FImpl::SiteSpinor vobj;
typedef typename vobj::scalar_type scalar_type;

struct BenchParams {
  int nvec;
  int ngamma;
};

void benchmarkGrid(GridCartesian &grid, const std::vector<BenchParams> &params,
                   int Nloop, int Nwarmup) {

  int nd = grid.Nd();
  int nsimd = grid.Nsimd();
  Coordinate ldims = grid.LocalDimensions();
  int Nt = grid.GlobalDimensions()[nd - 1];
  int osites = grid.oSites();
  int rNt = grid._rdimensions[nd - 1];
  uint64_t rNxyz = osites / rNt;
  int Nc = 3; // staggered colour

  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;
  std::cout << GridLogMessage << "Local volume " << ldims[0] << "x" << ldims[1]
            << "x" << ldims[2] << "x" << ldims[3] << "  oSites=" << osites
            << " Nsimd=" << nsimd << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;

  std::cout << GridLogMessage << "Contract: local, full volume" << std::endl;
  std::cout << GridLogMessage << std::setw(6) << "nvec" << std::setw(8)
            << "ngamma" << std::setw(10) << "oSites" << std::setw(12)
            << "kernel(ms)" << std::setw(12) << "gsum(ms)" << std::setw(12)
            << "total(ms)" << std::setw(12) << "GFLOP/s" << std::setw(10)
            << "GB/s" << std::endl;

  // RNG for field initialization
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  // Pre-allocate max number of vectors we'll need
  int maxVec = 0;
  int maxGamma = 0;
  for (const auto &p : params) {
    if (p.nvec > maxVec)
      maxVec = p.nvec;
    if (p.ngamma > maxGamma)
      maxGamma = p.ngamma;
  }

  // Allocate fermion fields
  std::vector<FermionField> lhs(maxVec, &grid);
  std::vector<FermionField> rhs(maxVec, &grid);
  for (int i = 0; i < maxVec; i++) {
    random(pRNG, lhs[i]);
    random(pRNG, rhs[i]);
  }

  // Allocate phase fields (random complex, unit magnitude would be
  // more realistic but doesn't affect timing)
  std::vector<ComplexField> phases(maxGamma, &grid);
  for (int mu = 0; mu < maxGamma; mu++) {
    phases[mu] = 1.0;
  }

  for (const auto &p : params) {
    int nvec = p.nvec;
    int ngamma = p.ngamma;

    // Create task with phases (full volume, no checkerboarding)
    std::vector<ComplexField> phaseSubset(phases.begin(),
                                          phases.begin() + ngamma);
    A2ATaskLocal<FImpl> task(&grid, Tdir, phaseSubset);

    // Set left and right views
    task.setLeft(lhs.data(), nvec);
    task.setRight(rhs.data(), nvec);

    // Output: flat scalar_type array of size ngamma * Nt * nvec * nvec
    // Must be device-allocated — execute()'s simdSum kernel writes from GPU
    int resultSize = ngamma * Nt * nvec * nvec;
    scalar_type *result_p =
        (scalar_type *)acceleratorAllocDevice(resultSize * sizeof(scalar_type));

    // Warmup
    for (int w = 0; w < Nwarmup; w++) {
      accelerator_for(i, resultSize, 1, { result_p[i] = scalar_type(0.0); });
      task.execute(result_p);
    }

    // Benchmark
    std::vector<double> t_kernel(Nloop), t_total(Nloop);
    for (int n = 0; n < Nloop; n++) {
      accelerator_for(i, resultSize, 1, { result_p[i] = scalar_type(0.0); });

      double t0 = usecond();
      task.execute(result_p);
      double t1 = usecond();

      // GlobalSum (as done in A2AWorkerBase::StagMesonField)
      grid.GlobalSumVector(result_p, resultSize);
      double t3 = usecond();

      t_kernel[n] = t1 - t0;
      t_total[n] = t3 - t0;
    }

    // Median
    std::vector<double> t_kernel_s(t_kernel), t_total_s(t_total);
    std::sort(t_kernel_s.begin(), t_kernel_s.end());
    std::sort(t_total_s.begin(), t_total_s.end());
    double tk = t_kernel_s[Nloop / 2];
    double tt = t_total_s[Nloop / 2];
    double tgsum = tt - tk;

    // execute() internally calls vectorSum + simdSum.
    // We report kernel = vectorSum + simdSum together since we can't
    // separate them without modifying the class.
    // For simd breakdown, use GRID_TRACE profiling.

    // FLOPs per site per (i,j) pair:
    //   inner product: 3 colours x (6 mult + 2 add) = 24 flops (complex)
    //   but as real: Nc * 8 = 24
    //   per gamma: phase multiply (8) + accumulate (2) = 10
    //   total per (i,j,site): 24 + ngamma * 10
    // The kernel runs over oSites with Nsimd lanes, so effective sites = oSites * Nsimd
    // But Grid's simd means the actual flops are on reduced sites with vector ops
    // Use: effective_sites = product of local dimensions (full volume)
    int localVol = 1;
    for (int d = 0; d < nd; d++)
      localVol *= ldims[d];
    double flops_per_site = (double)Nc * 8.0 + ngamma * 10.0;
    double total_flops = flops_per_site * (double)nvec * nvec * localVol;
    double gflops = total_flops / (tk * 1e3);

    // HBM traffic estimate (minimum):
    //   per (i,j) pair: read left_i (Nc * sizeof(vComplexD) per osite)
    //                    + read right_j (same)
    //                    + read ngamma phases (sizeof(cobj) per osite)
    //   left_i is reused across j (ideal), right_j reused across i
    //   Minimum: nvec * oSites * Nc * sizeof(vComplexD) * 2
    //            + ngamma * oSites * sizeof(vComplexD)
    //            + write shm: ngamma * nvec * nvec * rNt * sizeof(vComplexD)
    // But actual traffic depends on cache behavior. Report theoretical minimum.
    double bytes_vecs = 2.0 * nvec * osites * Nc * sizeof(vComplexD);
    double bytes_phases = (double)ngamma * osites * sizeof(vComplexD);
    double bytes_shm =
        (double)ngamma * nvec * nvec * rNt * sizeof(vComplexD);
    double hbm_min = bytes_vecs + bytes_phases + bytes_shm;
    // Actual: each (i,j) pair re-reads right_j, so realistic estimate:
    double hbm_actual =
        (double)nvec * osites * Nc * sizeof(vComplexD) +         // left (cached across j)
        (double)nvec * nvec * osites * Nc * sizeof(vComplexD) +  // right (re-read per i)
        bytes_phases * nvec * nvec +                              // phases (re-read per i,j)
        bytes_shm;                                                // shm write
    double gbps = hbm_actual / (tk * 1e-6) / 1.0e9;

    std::cout << GridLogMessage << std::setw(6) << nvec << std::setw(8)
              << ngamma << std::setw(10) << osites << std::fixed
              << std::setprecision(2) << std::setw(12) << tk / 1000.0
              << std::setw(12) << tgsum / 1000.0 << std::setw(12)
              << tt / 1000.0 << std::setprecision(1) << std::setw(12)
              << gflops << std::setw(10) << gbps << std::endl;

    acceleratorFreeDevice(result_p);
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::cout << GridLogMessage
            << "========================================" << std::endl;
  std::cout << GridLogMessage
            << "= Benchmark_ProdA2A (local, full vol)  =" << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  Coordinate mpi_layout = GridDefaultMpi();

  int Nloop = 10;
  int Nwarmup = 2;

  // Parameter sweep: (nvec, ngamma)
  std::vector<int> nvecs = {8, 16, 32, 64, 128};
  std::vector<int> ngammas = {1, 4, 8, 16};

  std::vector<BenchParams> params;
  for (int nv : nvecs)
    for (int ng : ngammas)
      params.push_back({nv, ng});

  std::cout << GridLogMessage << "Nloop=" << Nloop << " Nwarmup=" << Nwarmup
            << std::endl;

  Coordinate cmdLatt = GridDefaultLatt();
  bool hasGridArg = (cmdLatt[0] != 0);

  if (hasGridArg) {
    Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
    GridCartesian grid(cmdLatt, simd_layout, mpi_layout);
    benchmarkGrid(grid, params, Nloop, Nwarmup);
  } else {
    std::vector<int> lats = {8, 12, 16, 24};
    for (int lat : lats) {
      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1],
                            lat * mpi_layout[2], lat * mpi_layout[3]});
      Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
      GridCartesian grid(latt_size, simd_layout, mpi_layout);
      benchmarkGrid(grid, params, Nloop, Nwarmup);
    }
  }

  MemoryManager::Print();
  Grid_finalize();
  return 0;
}
