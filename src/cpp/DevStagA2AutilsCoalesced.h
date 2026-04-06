#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>
#include <SpatialTraceCoalesced.h>
#include <StagGamma.h>
#include <a2a/A2AView.h>
#include <typeinfo>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

template <typename... Ts> struct print_types;

template <typename FImpl> class DevA2AutilsCoalesced {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename ComplexField::vector_object cobj;
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
  typedef decltype(coalescedRead(Scalar_v())) calcScalar;

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

const int devA2Ablocking = DEV_A2A_BLOCKING;

template <typename vtype>
using iMatStag = iMatrix<iScalar<iScalar<vtype>>, devA2Ablocking>;
typedef iMatStag<Complex> MatStag;
typedef iMatStag<vComplex> vMatStag;
typedef Lattice<vMatStag> LatticeMatStag;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void DevA2AutilsCoalesced<FImpl>::MesonField(
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
  SpatialTraceCoalesced<ComplexField, ComplexField, MatStag> ST;

  MemoryManager::Print();

  // Allocate BLAS buffers
  // BLAS_L.resize(nt * nxyz * nleft * leftWords);
  // BLAS_R.resize(nt * nxyz * nright * rightWords);
  // BLAS_T.resize(nt * nresults * resultWords);
  ST.Allocate(Nmom * Ngamma, block * block, grid);

  auto blas_g = ST.getBlasLeftPointer();
  auto blas_ip = ST.getBlasRightPointer();
  int osites = grid->oSites();

  // Initialize BLAS_L
  int64_t Nleft = Nmom * Ngamma;
  for (int mmu = 0; mmu < Nleft; mmu++) {
    autoView(momG_v, momGamma[mmu], AcceleratorRead);

    accelerator_for(os, osites, Nsimd, {
      calcScalar data = coalescedRead(momG_v[os]);

      uint64_t idx = mmu + os * Nleft;
      coalescedWrite(((cobj *)blas_g)[idx], data);
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

      // Get grid dimensions
      int nd = grid->_ndimension;
      auto rNt = grid->_rdimensions[nd - 1];
      auto rNxyz = osites / rNt;

      {
        GRID_TRACE("localInner");

        int64_t Niprod = block * block; // Maximum size allocated

        // take local inner product
        // and Initialize BLAS_R
        accelerator_for2d(os, grid->oSites(), ii, nlcache, Nsimd, {
          int64_t r_t = os / rNxyz;
          int64_t r_xyz = os % rNxyz;

          auto left = coalescedRead(lhs_v[ii][os]);

          // TODO: Consider caching nrcache results before copying to blas_ip
          for (int jj = 0; jj < nrcache; jj++) {
            auto right = coalescedRead(rhs_v[jj][os]);
            calcScalar data = innerProduct(left, right);

            int64_t word_idx = ii * block + jj;
            // TEST: new layout: (t,word_idx,xyz)
            //  Old layout doesn't work with coalesced writes:
            //  uint64_t idx = r_xyz + word_idx * rNxyz + r_t * rNxyz * Niprod;
            //  Combine with transpose in SpatialTraceCoalesced
            uint64_t idx = word_idx + os * Niprod;
            coalescedWrite(((cobj *)blas_ip)[idx], data);
          }

          // for (int mu = 0; mu < nGamma; mu++) {
          //   int shmem_idx = rt + shmem_base + mu * gammaStride;
          //   coalescedWrite(shm_p[shmem_idx], sum[mu]);
          // }
        });
      }

      rhs_view.closeViews();

      std::vector<MatStag> trace_result;
      {
        GRID_TRACE("SpatialTrace");

        ST.Trace(trace_result);
      }

      {
        GRID_TRACE("ExtractResults");

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
      }

    } // jo

    lhs_view.closeViews();

  } // io
}

NAMESPACE_END(Grid);
