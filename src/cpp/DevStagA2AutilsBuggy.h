#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <A2AView.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <MomentumProject.h>
#include <StagGamma.h>
#include <nvtx3/nvToolsExt.h>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

///////////////////////////////////////////////////////////////////
// Meson
//  Interested in
//
//       sum_x,y Trace[ G S(x,tx,y,ty) G S(y,ty,x,tx) ]
//
//  Conventional meson field:
//
//     = sum_x,y Trace[ sum_j G |v_j(y,ty)> <w_j(x,tx)|  G sum_i |v_i(x,tx)
//     ><w_i(y,ty)| ] = sum_ij sum_x,y < w_j(x,tx)| G |v_i(x,tx) > <w_i(y,ty)
//     (x)|G| v_j(y,ty) > = sum_ij PI_ji(tx) PI_ij(ty)
//
//  G5-Hermiticity
//
//       sum_x,y Trace[ G S(x,tx,y,ty) G S(y,ty,x,tx) ]
//     = sum_x,y Trace[ G S(x,tx,y,ty) G g5 S^dag(x,tx,y,ty) g5 ]
//     = sum_x,y Trace[ g5 G sum_j |v_j(y,ty)> <w_j(x,tx)|  G g5 sum_i
//     (|v_j(y,ty)> <w_i(x,tx)|)^dag ]      --  (*)
//
//  NB:  Dag applies to internal indices spin,colour,complex
//
//     = sum_ij sum_x,y Trace[ g5 G |v_j(y,ty)> <w_j(x,tx)|  G g5  |w_i(x,tx)>
//     <v_i(y,ty)| ] = sum_ij sum_x,y <v_i(y,ty)|g5 G |v_j(y,ty)> <w_j(x,tx)|  G
//     g5 |w_i(x,tx)> = sum_ij  PionVV(ty) PionWW(tx)
//
//  (*) is only correct estimator if w_i and w_j come from distinct noise sets
//  to preserve the kronecker
//      expectation value. Otherwise biased.
////////////////////////////////////////////////////////////////////

template <typename FImpl> class DevA2AutilsBuggy {
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

const int devA2Ablocking = DEV_A2A_BLOCKING;

template <typename vtype>
using iVecStag = iVector<iScalar<iScalar<vtype>>, devA2Ablocking>;
typedef iVecStag<Complex> VecStag;
typedef iVecStag<vComplex> vVecStag;
typedef Lattice<vVecStag> LatticeVecStag;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void DevA2AutilsBuggy<FImpl>::MesonField(
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

  LatticeVecStag spinMat(grid);
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

  std::cout << GridLogMessage << "A2A Meson Field" << std::endl;
  MyMomentumProject<LatticeVecStag, ComplexField> MP;

  MemoryManager::Print();

  std::cout << GridLogMessage << "Allocating BLAS arrays" << std::endl;
  std::cout << GridLogMessage
            << "scalar size=" << sizeof(LatticeVecStag::scalar_type)
            << std::endl;
  auto words = sizeof(LatticeVecStag::scalar_object) /
               sizeof(LatticeVecStag::scalar_type);
  std::cout << GridLogMessage << "words=" << words << std::endl;
  auto ldims = grid->LocalDimensions();
  auto nt = ldims[grid->Nd() - 1];
  std::cout << GridLogMessage << "nt=" << nt << std::endl;
  std::cout << GridLogMessage << "Nmom=" << Nmom << std::endl;
  auto nxyz = grid->lSites() / nt;
  std::cout << GridLogMessage << "nxyz=" << nxyz << std::endl;

  std::cout << GridLogMessage << "BLAS_V size=" << (nxyz * nt * words)
            << std::endl;
  std::cout << GridLogMessage << "BLAS_M size=" << (Nmom * nxyz) << std::endl;
  std::cout << GridLogMessage << "BLAS_P size=" << (Nmom * nt * words)
            << std::endl;

  MP.Allocate(Nmom * Ngamma, grid);
  std::cout << GridLogMessage << "Importing momenta" << std::endl;
  MP.ImportMomenta(momGamma);
  std::cout << GridLogMessage << "Momentum project momenta imported"
            << std::endl;

  A2AFieldView<vobj> rhs_view;
  std::vector<VecStag> sliced;

  for (int i = 0; i < Lblock; i++) {
    autoView(spinMat_v, spinMat, AcceleratorWrite);
    autoView(lhs_v, lhs_wi[i], AcceleratorRead);
    for (int jo = 0; jo < Rblock; jo += block) {

      rhs_view.openViews(&rhs_vj[jo], MIN(Rblock - jo, block));
      auto rhs_v = rhs_view.getView();

      nvtxRangePushA("local Inner");
      accelerator_for(ss, grid->oSites(), (size_t)Nsimd, {
        auto left = lhs_v(ss);
        auto vv = spinMat_v(ss);
        for (int j = 0; j < MIN(Rblock - jo, block); j++) {
          auto right = rhs_v[j](ss);
          vv(j)()() = innerProduct(left, right)()()();
        }
        coalescedWrite(spinMat_v[ss], vv);
      });
      nvtxRangePop();

      rhs_view.closeViews();

      assert(orthogdim == Nd - 1);
      nvtxRangePushA("spatial trace");
      MP.Project(spinMat, sliced);
      nvtxRangePop();

      nvtxRangePushA("Extract results");
      thread_for2d(mmom, Nmom * Ngamma, t, Nt, {
        int m = mmom / Ngamma;
        int mu = mmom % Ngamma;
        int idx = t + mmom * Nt;
        for (int j = jo; j < MIN(Rblock, jo + block); j++) {
          int jj = j % block;
          auto tmp = peekIndex<LorentzIndex>(sliced[idx], jj);
          mat((long)m, mu, (long)t, i, j) = tmp()();
        }
      });
      nvtxRangePop();
    } // jo
  }
}

NAMESPACE_END(Grid);
