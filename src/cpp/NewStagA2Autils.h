#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>
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

template <typename FImpl> class NewA2Autils {
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
  static void MesonField(TensorType &mat, const FermionField *lhs_wi,
                         const FermionField *rhs_vj,
                         std::vector<StagGamma::SpinTastePair> gammas,
                         const std::vector<ComplexField> &mom, int orthogdim);
  template <typename TensorType>
  static void MesonField(TensorType &mat, const FermionField *lhs_wi,
                         const FermionField *rhs_vj,
                         std::vector<StagGamma::SpinTastePair> gammas,
                         const std::vector<ComplexField> &mom, int orthogdim,
                         double *timer) {
    MesonField(mat, lhs_wi, rhs_vj, gammas, mom, orthogdim);
  }
};

const int newA2Ablocking = 128;

template <typename vtype>
using iVecStag = iVector<iScalar<iScalar<vtype>>, newA2Ablocking>;
typedef iVecStag<Complex> VecStag;
typedef iVecStag<vComplex> vVecStag;
typedef Lattice<vVecStag> LatticeVecStag;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void NewA2Autils<FImpl>::MesonField(
    TensorType &mat, const FermionField *lhs_wi, const FermionField *rhs_vj,
    std::vector<StagGamma::SpinTastePair> gammas,
    const std::vector<ComplexField> &mom, int orthogdim) {

  const int block = newA2Ablocking;
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
  MomentumProject<LatticeVecStag, ComplexField> MP;
  MP.Allocate(Nmom * Ngamma, grid);
  MP.ImportMomenta(momGamma);
  std::cout << GridLogMessage << "Momentum project momenta imported"
            << std::endl;

  double t_view, t_gamma, t_kernel, t_momproj;
  t_view = 0;
  t_gamma = 0;
  t_kernel = 0;
  t_momproj = 0;

  std::vector<VecStag> sliced;
  for (int i = 0; i < Lblock; i++) {
    t_view -= usecond();
    autoView(spinMat_v, spinMat, AcceleratorWrite);
    autoView(lhs_v, lhs_wi[i], AcceleratorRead);
    t_view += usecond();
    for (int jo = 0; jo < Rblock; jo += block) {
      nvtxRangePushA("local Inner");
      for (int j = jo; j < MIN(Rblock, jo + block); j++) {
        int jj = j % block;
        t_view -= usecond();
        autoView(rhs_v, rhs_vj[j], AcceleratorRead); // Create a vector of views
        t_view += usecond();
        //////////////////////////////////////////
        // Should write a SpinOuterColorTrace
        //////////////////////////////////////////

        t_kernel -= usecond();
        accelerator_for(ss, grid->oSites(), (size_t)Nsimd, {
          auto left = conjugate(lhs_v(ss));
          auto right = rhs_v(ss);
          auto vv = spinMat_v(ss);
          vv(jj)()() = left()()(0) * right()()(0) + left()()(1) * right()()(1) +
                       left()()(2) * right()()(2);
          coalescedWrite(spinMat_v[ss], vv);
        });
        t_kernel += usecond();

      } // j within block
      nvtxRangePop();
      // After getting the sitewise product do the mom phase loop

      assert(orthogdim == Nd - 1);
      t_momproj -= usecond();
      MP.Project(spinMat, sliced);
      t_momproj += usecond();

      t_gamma -= usecond();
      thread_for2d(mmom, Nmom * Ngamma, t, Nt, {
        int m = mmom / Ngamma;
        int mu = mmom % Ngamma;
        //      for(int m=0;m<Nmom;m++)
        //	for(int t=0;t<Nt;t++)
        int idx = t + mmom * Nt;
        for (int j = jo; j < MIN(Rblock, jo + block); j++) {
          int jj = j % block;
          auto tmp = peekIndex<LorentzIndex>(sliced[idx], jj);
          mat((long)m, mu, (long)t, i, j) = tmp()();
        }
      });
      t_gamma += usecond();
    } // jo
  }
  std::cout << GridLogMessage << " A2A::MesonField t_view    " << t_view / 1e6
            << "s" << std::endl;
  std::cout << GridLogMessage << " A2A::MesonField t_momproj "
            << t_momproj / 1e6 << "s" << std::endl;
  std::cout << GridLogMessage << " A2A::MesonField t_kernel  " << t_kernel / 1e6
            << "s" << std::endl;
  std::cout << GridLogMessage << " A2A::MesonField t_gamma   " << t_gamma / 1e6
            << "s" << std::endl;
}

NAMESPACE_END(Grid);
