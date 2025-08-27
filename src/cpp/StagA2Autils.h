#pragma once
// #include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>
#include <StagGamma.h>
#include <nvtx3/nvToolsExt.h>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

template <typename FImpl> class StagA2Autils {
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

  // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  template <typename TensorType>
  static void MesonField(TensorType &mat, const FermionField *lhs_wi,
                         const FermionField *rhs_vj,
                         std::vector<StagGamma::SpinTastePair> gammas,
                         const std::vector<ComplexField> &mom, int orthogdim,
                         double *t_kernel = nullptr, double *t_gsum = nullptr);
};

const int stagA2Ablocking = 128;

template <typename vtype>
using iVecStag = iVector<iScalar<iScalar<vtype>>, stagA2Ablocking>;
typedef iVecStag<Complex> VecStag;
typedef iVecStag<vComplex> vVecStag;
typedef Lattice<vVecStag> LatticeVecStag;

template <class FImpl>
template <typename TensorType>
void StagA2Autils<FImpl>::MesonField(
    TensorType &mat, const FermionField *lhs_wi, const FermionField *rhs_vj,
    std::vector<StagGamma::SpinTastePair> gammas,
    const std::vector<ComplexField> &mom, int orthogdim, double *t_kernel,
    double *t_gsum) {

  const int block = stagA2Ablocking;
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  int Lblock = mat.dimension(3);
  int Rblock = mat.dimension(4);

  //  assert(Lblock % block==0);
  //  assert(Rblock % block==0);

  GridBase *grid = lhs_wi[0].Grid();

  //  const int    Nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();
  int Nmom = mom.size();

  LatticeVecStag SpinMat(grid);
  LatticeVecStag MomSpinMat(grid);
  StagGamma spinTaste;

  std::vector<VecStag> sliced;
  for (int i = 0; i < Lblock; i++) {
    autoView(SpinMat_v, SpinMat, AcceleratorWrite);
    autoView(lhs_v, lhs_wi[i], AcceleratorRead);
    for (int jo = 0; jo < Rblock; jo += block) {
      nvtxRangePushA("local Inner");
      for (int j = jo; j < MIN(Rblock, jo + block); j++) {
        int jj = j % block;
        autoView(rhs_v, rhs_vj[j], AcceleratorRead);

        accelerator_for(ss, grid->oSites(), (size_t)Nsimd, {
          auto left = conjugate(lhs_v(ss));
          auto right = rhs_v(ss);
          auto vv = SpinMat_v(ss);
          vv(jj)()() = left()()(0) * right()()(0) + left()()(1) * right()()(1) +
                       left()()(2) * right()()(2);
          coalescedWrite(SpinMat_v[ss], vv);
        });

      } // j within block
      nvtxRangePop();
      // After getting the sitewise product do the mom phase loop
      nvtxRangePushA("sliceSum");
      for (int m = 0; m < Nmom; m++) {

        for (int mu = 0; mu < Ngamma; mu++) {

          MomSpinMat = SpinMat * mom[m];

          spinTaste.setSpinTaste(gammas[mu]);
          spinTaste.applyPhase(MomSpinMat, MomSpinMat);

          sliceSum(MomSpinMat, sliced, orthogdim);

          nvtxRangePushA("write Eigen");
          for (int t = 0; t < sliced.size(); t++) {
            for (int j = jo; j < MIN(Rblock, jo + block); j++) {
              int jj = j % block;
              auto tmp = peekIndex<LorentzIndex>(sliced[t], jj);
              mat(m, mu, t, i, j) = tmp()();
            }
          }
          nvtxRangePop();
        }
      }
      nvtxRangePop();
    } // jo
  }
}

NAMESPACE_END(Grid);
