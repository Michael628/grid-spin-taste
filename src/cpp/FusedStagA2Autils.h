#pragma once

#include <Grid/Grid_Eigen_Tensor.h>
#include <SpatialTraceCoalesced.h>
#include <StagGamma.h>
#include <a2a/A2AView.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// FusedA2Autils: GEMM-based meson field contraction for staggered fermions.
//
// Instead of a two-stage approach (inner product kernel → spatial trace GEMM),
// this fuses inner product + phase multiplication + spatial sum into a single
// batched GEMM by interleaving colour components into the K dimension:
//
//   result[γ,i,j,t] = Σ_{xyz,c} conj(phase[γ,xyz] · left_i^c[xyz])
//                                     · right_j^c[xyz]
//
// BLAS_L stores conj(phase · left) with layout M×K column-major:
//   M = Ngamma × block × Nsimd,  K = rNxyz × Nc
//   m = (γ·block + i)·Nsimd + lane,  k = xyz·Nc + c
//
// BLAS_R stores right vectors with layout N×K column-major:
//   N = block × Nsimd,  K = rNxyz × Nc
//   n = j·Nsimd + lane,  k = xyz·Nc + c
//
// GEMM: C = L × R^T  (OP_N, OP_T)
// Conjugation is baked into L during fill. This works because staggered
// spin-taste phases are real (±1). For complex momentum phases, factor
// them onto the right side: conj(spin_taste · left) in L, e^{ip·x} · right in
// R.
///////////////////////////////////////////////////////////////////////////////

template <typename FImpl> class FusedA2Autils {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename ComplexField::vector_object cobj;
  typedef typename FImpl::FermionField FermionField;

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;
  typedef decltype(coalescedRead(Scalar_v())) calcScalar;
  typedef decltype(coalescedRead(vobj())) calcSpinor;

  static constexpr int Nc = 3;

  template <typename TensorType>
  static void MesonFieldLocal(TensorType &mat,
                              std::vector<FermionField> &lhs_wi,
                              std::vector<FermionField> &rhs_vj,
                              std::vector<StagGamma::SpinTastePair> gammas,
                              const std::vector<ComplexField> &mom,
                              int orthogdim, int block);
};

template <class FImpl>
template <typename TensorType>
void FusedA2Autils<FImpl>::MesonFieldLocal(
    TensorType &mat, std::vector<FermionField> &lhs_wi,
    std::vector<FermionField> &rhs_vj,
    std::vector<StagGamma::SpinTastePair> gammas,
    const std::vector<ComplexField> &mom, int orthogdim, int block) {

  int Lblock = mat.dimension(3);
  int Rblock = mat.dimension(4);

  GridBase *grid = lhs_wi[0].Grid();
  const int Nsimd = grid->Nsimd();

  int Nt = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();
  int Nmom = mom.size();

  int nd = grid->_ndimension;
  auto rNt = grid->_rdimensions[nd - 1];
  auto rNxyz = grid->oSites() / rNt;
  int osites = grid->oSites();

  // Compute momGamma phase fields
  std::vector<ComplexField> momGamma(Nmom * Ngamma, grid);
  StagGamma spinTaste;

  for (int m = 0; m < Nmom; m++) {
    for (int mu = 0; mu < Ngamma; mu++) {
      int mmu = m * Ngamma + mu;
      momGamma[mmu] = mom[m];
      spinTaste.setSpinTaste(gammas[mu]);
      spinTaste.applyPhase(momGamma[mmu], momGamma[mmu]);
    }
  }

  int Nphase = Nmom * Ngamma;

  std::cout << GridLogDebug << "Fused A2A Meson Field: Ngamma=" << Ngamma
            << " Nmom=" << Nmom << " block=" << block << " Nc=" << Nc
            << std::endl;

  // nleft = Nphase * block (gammas × left vectors per block)
  // nright = block (right vectors per block)
  // K = rNxyz * Nc (spatial sites × colour)
  typedef iSinglet<Complex> ResultObj;
  SpatialTraceCoalesced<ComplexField, ComplexField, ResultObj> ST;

  if (GridLogPerformance.isActive()) {
    MemoryManager::Print();
  }

  ST.Allocate(Nphase * block, block, grid, Nc);

  auto *blas_L = ST.getBlasLeftPointer();
  auto *blas_R = ST.getBlasRightPointer();

  int64_t Nleft = Nphase * block;
  int64_t Nright = block;
  int64_t K = rNxyz * Nc;

  A2AFieldView<vobj> lhs_view, rhs_view;

  // Open phase views
  A2AFieldView<cobj> phase_view;
  phase_view.openViews(momGamma.data(), Nphase);
  auto phase_v = phase_view.getView();

  for (int io = 0; io < Lblock; io += block) {
    int nlcache = MIN(Lblock - io, block);

    std::cout << GridLogDebug << "Left block " << io << " of " << Lblock
              << std::endl;

    lhs_view.openViews(&lhs_wi[io], nlcache);
    auto lhs_v = lhs_view.getView();

    // Fill BLAS_L: conj(phase[mmu] * left_i^c) for each (mmu, i, c)
    // Layout: M×K column-major via vector* pointer
    //   vector_idx = m_vec + Nleft * (os * Nc + c)
    //   where m_vec = mmu * block + ii
    {
      GRID_TRACE("FillBlasL");

      for (int mmu = 0; mmu < Nphase; mmu++) {

        accelerator_for2d(os, osites, ii, nlcache, Nsimd, {
          calcSpinor left = coalescedRead(lhs_v[ii][os]);
          calcScalar phase = coalescedRead(phase_v[mmu][os]);
          int64_t m_vec = mmu * block + ii;

          for (int c = 0; c < Nc; c++) {
            // Store conj(phase * left^c) = conj(phase) * conj(left^c)
            // For real staggered phases: conj(phase) = phase
            calcScalar val;
            val()()() = conjugate(phase()()()) * conjugate(left()()(c));

            int64_t idx = m_vec + Nleft * (os * Nc + c);
            coalescedWrite(((cobj *)blas_L)[idx], val);
          }
        });
      }
    }

    for (int jo = 0; jo < Rblock; jo += block) {
      int nrcache = MIN(Rblock - jo, block);

      std::cout << GridLogDebug << "  Right block " << jo << " of " << Rblock
                << std::endl;

      rhs_view.openViews(&rhs_vj[jo], nrcache);
      auto rhs_v = rhs_view.getView();

      // Fill BLAS_R: right_j^c for each (j, c)
      // Layout: N×K column-major via vector* pointer
      //   vector_idx = jj + Nright * (os * Nc + c)
      {
        GRID_TRACE("FillBlasR");

        accelerator_for2d(os, osites, jj, nrcache, Nsimd, {
          calcSpinor right = coalescedRead(rhs_v[jj][os]);

          for (int c = 0; c < Nc; c++) {
            calcScalar val;
            val()()() = right()()(c);

            int64_t idx = jj + Nright * (os * Nc + c);
            coalescedWrite(((cobj *)blas_R)[idx], val);
          }
        });
      }

      rhs_view.closeViews();

      std::vector<ResultObj> trace_result;
      {
        GRID_TRACE("SpatialTrace");
        ST.Trace(trace_result);
      }

      // Extract results into output tensor
      // trace_result layout: gt * nresults entries
      // nresults = Nleft * Nright = Nphase * block * block
      // ExportTrace produces row-major: result[t * nresults + i * Nright + j]
      // where i = mmu * block + ii, j = jj
      {
        GRID_TRACE("ExtractResults");

        int nresults = Nleft * Nright;

        thread_for2d(mmu, Nphase, t, Nt, {
          int m = mmu / Ngamma;
          int mu = mmu % Ngamma;

          for (int ii = 0; ii < nlcache; ii++) {
            for (int jj = 0; jj < nrcache; jj++) {
              int result_idx = (mmu * block + ii) * Nright + jj;
              auto tmp = trace_result[t * nresults + result_idx];
              mat((long)m, mu, (long)t, io + ii, jo + jj) = tmp()();
            }
          }
        });
      }

    } // jo

    lhs_view.closeViews();

  } // io

  phase_view.closeViews();
}

NAMESPACE_END(Grid);
