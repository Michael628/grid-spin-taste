/*************************************************************************************
    Benchmark for SpatialTraceCoalesced batched GEMM performance.

    Sweeps nleft, nright, and local volume to characterize GEMM scaling.
    Reports GFLOP/s and HBM bandwidth utilization.

    Usage:
      ./benchmark_batchtrace --grid L.L.L.L
    or without --grid to sweep hardcoded lattice sizes.
*************************************************************************************/

#include <Grid/Grid.h>
#include <SpatialTraceCoalesced.h>

using namespace Grid;

typedef iScalar<iScalar<iScalar<vComplexD>>> vSinglet;
typedef Lattice<vSinglet> ScalarField;
typedef iSinglet<Complex> ResultObj;
typedef SpatialTraceCoalesced<ScalarField, ScalarField, ResultObj> BatchTrace;

struct BenchParams {
  int nleft;
  int nright;
};

void benchmarkGrid(GridCartesian &grid, const std::vector<BenchParams> &params,
                   int Nloop, int Nwarmup, double maxMemGB) {

  int nd = grid.Nd();
  int nsimd = grid.Nsimd();
  int rNt = grid._rdimensions[nd - 1];
  uint64_t rNxyz = grid.oSites() / rNt;
  Coordinate ldims = grid.LocalDimensions();

  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;
  std::cout << GridLogMessage << "Local volume " << ldims[0] << "x" << ldims[1]
            << "x" << ldims[2] << "x" << ldims[3] << "  rNt=" << rNt
            << " rNxyz=" << rNxyz << " Nsimd=" << nsimd << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;

  std::cout << GridLogMessage << std::setw(6) << "nleft" << std::setw(8)
            << "nright" << std::setw(8) << "M" << std::setw(8) << "N"
            << std::setw(8) << "K" << std::setw(6) << "batch" << std::setw(10)
            << "mem(GB)" << std::setw(12) << "total(ms)" << std::setw(12)
            << "GFLOP/s" << std::setw(10) << "GB/s" << std::endl;

  for (const auto &p : params) {
    int nleft = p.nleft;
    int nright = p.nright;

    uint64_t leftWB = nsimd;
    uint64_t rightWB = nsimd;
    uint64_t resultWB = (uint64_t)nsimd * nsimd;

    double mem_L = (double)rNt * rNxyz * nleft * leftWB * sizeof(ComplexD);
    double mem_R = (double)rNt * rNxyz * nright * rightWB * sizeof(ComplexD);
    uint64_t nresults = (uint64_t)nleft * nright;
    double mem_T = (double)rNt * nresults * resultWB * sizeof(ComplexD);
    double mem_GB = (mem_L + mem_R + mem_T) / 1.0e9;

    uint64_t M = nleft * leftWB;
    uint64_t N = nright * rightWB;
    uint64_t K = rNxyz;
    int batches = rNt;

    if (mem_GB > maxMemGB) {
      std::cout << GridLogMessage << std::setw(6) << nleft << std::setw(8)
                << nright << std::setw(8) << M << std::setw(8) << N
                << std::setw(8) << K << std::setw(6) << batches
                << "  SKIP (" << std::fixed << std::setprecision(1) << mem_GB
                << " GB)" << std::endl;
      continue;
    }

    BatchTrace ST;
    ST.Allocate(nleft, nright, &grid);

    // Fill BLAS buffers with constant data on device
    {
      auto *L_p = &ST.BLAS_L[0];
      auto *R_p = &ST.BLAS_R[0];
      uint64_t nL = ST.BLAS_L.size();
      uint64_t nR = ST.BLAS_R.size();
      accelerator_for(i, nL, 1, { L_p[i] = ComplexD(1.0, 0.0); });
      accelerator_for(i, nR, 1, { R_p[i] = ComplexD(1.0, 0.0); });
    }

    std::vector<ResultObj> trace_result;

    // Warmup
    for (int w = 0; w < Nwarmup; w++) {
      ST.Trace(trace_result);
    }

    // Benchmark: collect per-iteration times
    std::vector<double> times(Nloop);
    for (int n = 0; n < Nloop; n++) {
      double t0 = usecond();
      ST.Trace(trace_result);
      double t1 = usecond();
      times[n] = t1 - t0;
    }

    // Report median to filter outliers
    std::sort(times.begin(), times.end());
    double t_us = times[Nloop / 2];

    // GEMM FLOPs: complex C = A * B^T
    // Each complex multiply-accumulate = 8 real flops
    double flops = 8.0 * M * N * K * batches;
    double gflops = flops / (t_us * 1e3);

    // Minimum HBM traffic: read L + read R + write T
    double hbm_bytes = mem_L + mem_R + mem_T;
    double gbps = hbm_bytes / (t_us * 1e-6) / 1.0e9;

    std::cout << GridLogMessage << std::setw(6) << nleft << std::setw(8)
              << nright << std::setw(8) << M << std::setw(8) << N
              << std::setw(8) << K << std::setw(6) << batches << std::fixed
              << std::setprecision(2) << std::setw(10) << mem_GB
              << std::setprecision(2) << std::setw(12) << t_us / 1000.0
              << std::setprecision(1) << std::setw(12) << gflops
              << std::setw(10) << gbps << std::endl;

    ST.Deallocate();
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::cout << GridLogMessage
            << "========================================" << std::endl;
  std::cout << GridLogMessage
            << "= Benchmark_BatchTrace                 =" << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  Coordinate mpi_layout = GridDefaultMpi();

  int Nloop = 10;
  int Nwarmup = 2;
  double maxMemGB = 32.0;

  // Parameter sweep
  std::vector<int> nlefts = {4, 8, 16, 32, 64};
  std::vector<int> nrights = {64, 256, 1024, 4096, 16384};

  std::vector<BenchParams> params;
  for (int nl : nlefts)
    for (int nr : nrights)
      params.push_back({nl, nr});

  std::cout << GridLogMessage << "Nloop=" << Nloop << " Nwarmup=" << Nwarmup
            << " maxMem=" << maxMemGB << " GB" << std::endl;

  // If --grid was provided, use that single volume
  Coordinate cmdLatt = GridDefaultLatt();
  bool hasGridArg = (cmdLatt[0] != 0);

  if (hasGridArg) {
    Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
    GridCartesian grid(cmdLatt, simd_layout, mpi_layout);
    benchmarkGrid(grid, params, Nloop, Nwarmup, maxMemGB);
  } else {
    std::vector<int> lats = {8, 12, 16, 24};
    for (int lat : lats) {
      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1],
                            lat * mpi_layout[2], lat * mpi_layout[3]});
      Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
      GridCartesian grid(latt_size, simd_layout, mpi_layout);
      benchmarkGrid(grid, params, Nloop, Nwarmup, maxMemGB);
    }
  }

  MemoryManager::Print();
  Grid_finalize();
  return 0;
}
