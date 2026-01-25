#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <A2AView.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <SpatialTraceVector.h>
#include <StagGamma.h>
#include <nvtx3/nvToolsExt.h>
#include <typeinfo>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

template <typename... Ts> struct print_types;

template <typename FImpl> class DevA2AutilsVector {
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
using iVecStag = iVector<iScalar<iScalar<vtype>>, devA2Ablocking>;
typedef iVecStag<Complex> VecStag;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void DevA2AutilsVector<FImpl>::MesonField(
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
  SpatialTraceVector<ComplexField, ComplexField, VecStag> ST;

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
  ST.Allocate(Nmom * Ngamma, block, grid);

  auto blas_g = ST.getBlasLeftPointer();
  auto blas_ip = ST.getBlasRightPointer();
  auto tMap_p = ST.getTmapPointer();
  auto xyzMap_p = ST.getXYZmapPointer();

  uint64_t osites = grid->oSites();

  // Initialize BLAS_L
  for (int mmu = 0; mmu < Nmom * Ngamma; mmu++) {
    autoView(momG_v, momGamma[mmu], AcceleratorRead);

    int64_t Nleft = Nmom * Ngamma;

    accelerator_for(os, osites, Nsimd, {
      auto lane = acceleratorSIMTlane(Nsimd);
      auto lane_idx = lane * osites;

      auto data = extractLane(lane, momG_v[os]);

      uint64_t idx = mmu + xyzMap_p[lane_idx + os] * Nleft +
                     tMap_p[lane_idx + os] * nxyz * Nleft;
      blas_g[idx] = data;
    });
  }

  A2AFieldView<vobj> rhs_view;

  for (int jo = 0; jo < Rblock; jo += block) {
    int nrcache = MIN(Rblock - jo, block);

    std::cout << GridLogMessage << "Computing inner products for block " << jo
              << " of " << Rblock << std::endl;

    rhs_view.openViews(&rhs_vj[jo], nrcache);
    auto rhs_v = rhs_view.getView();

    for (int i = 0; i < Lblock; i++) {
      autoView(lhs_v, lhs_wi[i], AcceleratorRead);

      if (i == 0) {
        nvtxRangePushA("inner_profile");
      }
      nvtxRangePushA("local Inner");

      // take local inner product
      // and Initialize BLAS_R
      // Parallelize over all (ii, jj, os) combinations
      uint64_t total_work = (uint64_t)nrcache * osites;

      accelerator_for(work_idx, total_work, Nsimd, {
        uint32_t os = work_idx % osites;
        uint32_t jj = work_idx / osites;

        // Map from blas layout to grid lattice layout
        auto lane = acceleratorSIMTlane(Nsimd);
        auto lane_idx = lane * osites + os;

        Scalar_v vv;

        vv = innerProduct(coalescedRead(lhs_v[os]),
                          coalescedRead(rhs_v[jj][os]));
        auto data = extractLane(lane, vv);

        // HOISTED: Compute invariant terms and index
        uint64_t word_offset = jj * nxyz;
        uint64_t t_stride = nxyz * block;
        uint64_t xyz = xyzMap_p[lane_idx];
        uint64_t t = tMap_p[lane_idx];
        uint64_t idx = xyz + word_offset + t * t_stride;

        blas_ip[idx] = data;
      });

      nvtxRangePop();

      if (i == 0) {
        nvtxRangePop();
      }
      nvtxRangePushA("SpatialTrace");

      std::vector<VecStag> trace_result;
      ST.Trace(trace_result);

      nvtxRangePop();

      nvtxRangePushA("Extract results");

      thread_for2d(mmom, Nmom * Ngamma, t, nt, {
        int m = mmom / Ngamma;
        int mu = mmom % Ngamma;
        int idx = mmom + Nmom * Ngamma * t;

        for (int j = jo; j < MIN(Rblock, jo + block); j++) {
          int jj = j % block;

          auto tmp = peekIndex<LorentzIndex>(trace_result[idx], jj);
          mat((long)m, mu, (long)t, i, j) = tmp()();
        }
      });
      nvtxRangePop();
    } // i

    rhs_view.closeViews();
  } // jo
}

NAMESPACE_END(Grid);
