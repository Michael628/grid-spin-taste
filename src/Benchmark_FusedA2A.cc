/*************************************************************************************
    Benchmark for fused GEMM A2A meson field kernel (FusedStagA2Autils).

    Tests full-volume local contraction performance, sweeping:
      - local volume (via --grid or internal sweep)
      - number of left/right vectors (equal)
      - number of gamma phases

    Usage:
      ./benchmark_fuseda2a --grid L.L.L.L
    or without --grid to sweep hardcoded lattice sizes.
*************************************************************************************/

#include <Grid/Grid.h>
#include <FusedStagA2Autils.h>

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
  int Nc = 3;

  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;
  std::cout << GridLogMessage << "Local volume " << ldims[0] << "x" << ldims[1]
            << "x" << ldims[2] << "x" << ldims[3] << "  oSites=" << osites
            << " Nsimd=" << nsimd << std::endl;
  std::cout << GridLogMessage
            << "============================================" << std::endl;

  std::cout << GridLogMessage << "Contract: fused GEMM, full volume"
            << std::endl;
  std::cout << GridLogMessage << std::setw(6) << "nvec" << std::setw(8)
            << "ngamma" << std::setw(8) << "block" << std::setw(10) << "oSites"
            << std::setw(12) << "kernel(ms)" << std::setw(12) << "gsum(ms)"
            << std::setw(12) << "total(ms)" << std::setw(12) << "GFLOP/s"
            << std::setw(10) << "GB/s" << std::setw(12) << "BLAS(MB)"
            << std::endl;

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

  // Allocate momentum phases (just zero momentum = unit phase)
  std::vector<ComplexField> mom(1, &grid);
  mom[0] = 1.0;

  // All 16 spin-taste pairs (TXYZ convention)
  // The specific gamma doesn't affect timing, only the count matters
  using SA = StagGamma::StagAlgebra;
  std::vector<StagGamma::SpinTastePair> allGammas = {
      {SA::G1, SA::G1},   {SA::GZ, SA::GZ},   {SA::GY, SA::GY},
      {SA::GYZ, SA::GYZ}, {SA::GX, SA::GX},   {SA::GZX, SA::GZX},
      {SA::GXY, SA::GXY}, {SA::G5T, SA::G5T}, {SA::GT, SA::GT},
      {SA::GZT, SA::GZT}, {SA::GYT, SA::GYT}, {SA::G5X, SA::G5X},
      {SA::GXT, SA::GXT}, {SA::G5Y, SA::G5Y}, {SA::G5Z, SA::G5Z},
      {SA::G5, SA::G5}};

  for (const auto &p : params) {
    int nvec = p.nvec;
    int ngamma = p.ngamma;

    // Use block = nvec (single block) for simplicity in benchmarking
    int block = nvec;

    // Subset of gammas
    std::vector<StagGamma::SpinTastePair> gammas(allGammas.begin(),
                                                  allGammas.begin() + ngamma);

    // Output tensor: (mom, gamma, t, lhs, rhs)
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> Mpp(
        1, ngamma, Nt, nvec, nvec);

    // Warmup
    for (int w = 0; w < Nwarmup; w++) {
      FusedA2Autils<FImpl>::MesonFieldLocal(Mpp, lhs, rhs, gammas, mom, Tdir,
                                             block);
    }

    // Benchmark
    std::vector<double> t_kernel(Nloop), t_total(Nloop);
    for (int n = 0; n < Nloop; n++) {
      double t0 = usecond();
      FusedA2Autils<FImpl>::MesonFieldLocal(Mpp, lhs, rhs, gammas, mom, Tdir,
                                             block);
      double t1 = usecond();

      // GlobalSum (as would be done in production)
      int resultSize = 1 * ngamma * Nt * nvec * nvec;
      grid.GlobalSumVector((scalar_type *)Mpp.data(), resultSize);
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

    // GEMM FLOPs:
    //   GEMM C = L * R^T:  M = Nphase*block*Nsimd, N = block*Nsimd, K = rNxyz*Nc
    //   FLOPs per GEMM = 2*M*N*K (complex: 8*M*N*K real flops)
    //   Batches = rNt
    int Nphase = 1 * ngamma; // Nmom * Ngamma
    int64_t M_gemm = (int64_t)Nphase * block;
    int64_t N_gemm = (int64_t)block;
    int64_t K_gemm = (int64_t)rNxyz * Nc;
    // Complex GEMM: 8 real flops per (i,j,k) element
    double gemm_flops = 8.0 * M_gemm * nsimd * N_gemm * nsimd * K_gemm * rNt;
    // Fill kernel FLOPs: conjugate + multiply per element
    // L fill: Nphase * block * osites * Nc * ~6 flops (conj + mult)
    // R fill: block * osites * Nc * ~2 flops (copy)
    double fill_flops = (double)Nphase * block * osites * Nc * 6.0 +
                        (double)block * osites * Nc * 2.0;
    double total_flops = gemm_flops + fill_flops;
    double gflops = total_flops / (tk * 1e3);

    // HBM traffic:
    //   Read left vectors: nvec * osites * Nc * sizeof(vComplexD) [once per io block]
    //   Read right vectors: nvec * osites * Nc * sizeof(vComplexD) [once per jo block]
    //   Read phases: Nphase * osites * sizeof(vComplexD) [once per io block]
    //   Write BLAS_L: Nphase * block * rNxyz * Nc * rNt * sizeof(vComplexD)
    //   Write BLAS_R: block * rNxyz * Nc * rNt * sizeof(vComplexD)
    //   GEMM read/write: dominated by above
    double bytes_read = ((double)nvec * 2 + Nphase) * osites * Nc *
                            sizeof(vComplexD) +
                        (double)Nphase * osites * sizeof(vComplexD);
    double bytes_blas_l =
        (double)Nphase * block * rNxyz * Nc * rNt * nsimd * sizeof(scalar_type);
    double bytes_blas_r =
        (double)block * rNxyz * Nc * rNt * nsimd * sizeof(scalar_type);
    double hbm = bytes_read + bytes_blas_l + bytes_blas_r;
    double gbps = hbm / (tk * 1e-6) / 1.0e9;

    // BLAS buffer memory (outside Grid's memory pool)
    // SpatialTraceCoalesced allocates with K_factor=Nc:
    //   rNxyz_k = rNxyz * Nc
    //   BLAS_L: rNt * rNxyz_k * Nphase * block * nsimd scalars
    //   BLAS_R: rNt * rNxyz_k * block * nsimd scalars
    //   BLAS_T: rNt * Nphase * block * block * nsimd * nsimd scalars
    int64_t rNxyz_k = rNxyz * Nc;
    double blas_l_bytes = (double)rNt * rNxyz_k * Nphase * block * nsimd *
                          sizeof(scalar_type);
    double blas_r_bytes =
        (double)rNt * rNxyz_k * block * nsimd * sizeof(scalar_type);
    double blas_t_bytes = (double)rNt * Nphase * block * block * nsimd *
                          nsimd * sizeof(scalar_type);
    double blas_total_mb = (blas_l_bytes + blas_r_bytes + blas_t_bytes) / 1e6;

    std::cout << GridLogMessage << std::setw(6) << nvec << std::setw(8)
              << ngamma << std::setw(8) << block << std::setw(10) << osites
              << std::fixed << std::setprecision(2) << std::setw(12)
              << tk / 1000.0 << std::setw(12) << tgsum / 1000.0
              << std::setw(12) << tt / 1000.0 << std::setprecision(1)
              << std::setw(12) << gflops << std::setw(10) << gbps
              << std::setprecision(1) << std::setw(12) << blas_total_mb
              << std::endl;
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::cout << GridLogMessage
            << "========================================" << std::endl;
  std::cout << GridLogMessage
            << "= Benchmark_FusedA2A (local, full vol) =" << std::endl;
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
