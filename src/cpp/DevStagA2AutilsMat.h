#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <A2AView.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <SpatialTrace.h>
#include <StagGamma.h>
#include <nvtx3/nvToolsExt.h>
#include <typeinfo>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

template <typename... Ts> struct print_types;

template <typename FImpl> class DevA2AutilsMat {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSpinMatrix<vector_type> SpinMatrix_v;
  typedef iSpinMatrix<scalar_type> SpinMatrix_s;
  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;

  typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;

  // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  template <typename TensorType>
  static void MesonField(TensorType &mat, std::vector<FermionField> &lhs_wi,
                         std::vector<FermionField> &rhs_vj,
                         std::vector<StagGamma::SpinTastePair> gammas,
                         const std::vector<ComplexField> &mom, int orthogdim);
};

#ifndef DEV_A2A_BLOCKING
#define DEV_A2A_BLOCKING 128
#endif

extern const int devA2Ablocking;

template <typename vtype>
using iMatStag = iMatrix<iScalar<iScalar<vtype>>, devA2Ablocking>;
typedef iMatStag<Complex> MatStag;
typedef iMatStag<vComplex> vMatStag;
typedef Lattice<vMatStag> LatticeMatStag;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void DevA2AutilsMat<FImpl>::MesonField(
    TensorType &mat, std::vector<FermionField> &lhs_wi,
    std::vector<FermionField> &rhs_vj,
    std::vector<StagGamma::SpinTastePair> gammas,
    const std::vector<ComplexField> &mom, int orthogdim) {

  const int block = devA2Ablocking;
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  int Lblock = mat.dimension(3);
  int Rblock = mat.dimension(4);

  GridBase *grid = lhs_wi[0].Grid();

  //  const int    Nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();
  int Nmom = mom.size();

  // Allocate Nmom*Ngamma complex vectors
  std::vector<ComplexField> momGamma(Nmom * Ngamma, grid);
  StagGamma spinTaste;

  // Instantiate momenta and gammas
  for (int m = 0; m < Nmom; m++) {
    for (int mu = 0; mu < Ngamma; mu++) {
      int mmu = m * Ngamma + mu;
      momGamma[mmu] = mom[m];

      spinTaste.setSpinTaste(gammas[mu]);
      spinTaste.applyPhase(momGamma[mmu], momGamma[mmu]);
    }
  }

  std::cout << GridLogMessage << "A2A Meson Field" << std::endl;
  SpatialTrace<ComplexField, ComplexField, MatStag> ST;

  MemoryManager::Print();

  // Get grid dimensions
  int nd = grid->_ndimension;
  Coordinate ldims = grid->LocalDimensions();
  auto nt = ldims[grid->Nd() - 1];
  auto nxyz = grid->lSites() / nt;

  // Allocate BLAS buffers
  // BLAS_L.resize(nt * nxyz * nleft * leftWords);
  // BLAS_R.resize(nt * nxyz * nright * rightWords);
  // BLAS_T.resize(nt * nresults * resultWords);
  ST.Allocate(Nmom * Ngamma, block * block, grid);

  auto blas_g = ST.getBlasLeftPointer();
  auto blas_ip = ST.getBlasRightPointer();
  auto imap_p = ST.getImapPointer();
  auto omap_p = ST.getOmapPointer();

  // Initialize BLAS_L
  for (int mmu = 0; mmu < Nmom * Ngamma; mmu++) {
    autoView(momG_v, momGamma[mmu], AcceleratorRead);

    int64_t Nt = nt;
    int64_t Nxyz = nxyz;
    int64_t Nleft = Nmom * Ngamma;

    accelerator_for(ls, grid->lSites(), 1, {
      auto ss = omap_p[ls];
      auto lane = imap_p[ls];

      int64_t l_t = ls / Nxyz;
      int64_t l_xyz = ls % Nxyz;
      auto data = extractLane(lane, momG_v[ss]);

      uint64_t idx = mmu + l_xyz * Nleft + l_t * Nxyz * Nleft;
      // uint64_t idx = l_xyz + mmu * Nxyz + l_t * Nxyz * Nleft;
      blas_g[idx] = data;
    });
  }

  A2AFieldView<vobj> lhs_view, rhs_view;

  for (int io = 0; io < Lblock; io += block) {
    int nlcache = MIN(Lblock - io, block);

    std::cout << GridLogMessage << "Computing inner products for block " << io
              << " of " << Lblock << std::endl;

    lhs_view.openViews(&lhs_wi[io], nlcache);
    auto lhs_v = lhs_view.getView();

    for (int jo = 0; jo < Rblock; jo += block) {
      int nrcache = MIN(Rblock - jo, block);

      std::cout << GridLogMessage << "Computing inner products for block " << jo
                << " of " << Rblock << std::endl;

      rhs_view.openViews(&rhs_vj[jo], nrcache);
      auto rhs_v = rhs_view.getView();

      nvtxRangePushA("local Inner");

      int64_t Nt = nt;
      int64_t Nxyz = nxyz;
      int64_t Niprod = block * block; // Maximum size allocated

      // take local inner product
      // and Initialize BLAS_R
      accelerator_for2d(ls, grid->lSites(), ii, nlcache, 1, {
        // Map from blas layout to grid lattice layout
        auto ss = omap_p[ls];
        auto lane = imap_p[ls];

        int64_t l_t = ls / Nxyz;
        int64_t l_xyz = ls % Nxyz;

        auto left = lhs_v[ii][ss];

        for (int jj = 0; jj < nrcache; jj++) {
          auto right = rhs_v[jj][ss];
          Scalar_v vv;

          vv = innerProduct(left, right);
          auto data = extractLane(lane, vv);

          int64_t word_idx = ii * block + jj;
          // uint64_t idx = word_idx + l_xyz * Niprod + l_t * Nxyz * Niprod;
          uint64_t idx = l_xyz + word_idx * Nxyz + l_t * Nxyz * Niprod;
          blas_ip[idx] = data;
        }
      });

      nvtxRangePop();

      rhs_view.closeViews();

      nvtxRangePushA("SpatialTrace");

      std::vector<MatStag> trace_result;
      ST.Trace(trace_result);

      nvtxRangePop();

      nvtxRangePushA("Extract results");

      thread_for2d(mmom, Nmom * Ngamma, t, Nt, {
        int m = mmom / Ngamma;
        int mu = mmom % Ngamma;
        int idx = mmom + Nmom * Ngamma * t;

        for (int i = io; i < MIN(Lblock, io + block); i++) {
          int ii = i % block;
          for (int j = jo; j < MIN(Rblock, jo + block); j++) {
            int jj = j % block;

            auto tmp = peekIndex<LorentzIndex>(trace_result[idx], ii, jj);
            mat((long)m, mu, (long)t, i, j) = tmp()();
          }
        }
      });

      nvtxRangePop();

    } // jo

    lhs_view.closeViews();

  } // io
}

NAMESPACE_END(Grid);
