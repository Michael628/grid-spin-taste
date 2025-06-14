#include <Eigenpack.h>
#include <Grid/Eigen/unsupported/CXX11/Tensor>
#include <Grid/Grid.h>
#include <StagGamma.h>
// #include "cpp/Contract.h"

using namespace Grid;

template <typename T>
using A2AMatrixSet = Eigen::TensorMap<Eigen::Tensor<T, 5, Eigen::RowMajor>>;

template <typename T>
using A2AMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

template <typename T>
using A2AMatrixTr = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;

#if 1
typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;
typedef typename ImprovedStaggeredFermionF::FermionField FermionFieldF;
typedef typename ImprovedStaggeredFermionD::ComplexField ComplexField;
typedef typename ImprovedStaggeredFermionD::Impl_t STAGIMPL;
typedef ImprovedStaggeredFermionD FMat;
typedef typename ImprovedStaggeredFermionD::SiteSpinor vobj;
#else
typename NaiveStaggeredFermionR::ImplParams params;
typedef typename NaiveStaggeredFermionD::FermionField FermionFieldD;
typedef typename NaiveStaggeredFermionF::FermionField FermionFieldF;
typedef typename NaiveStaggeredFermionD::ComplexField ComplexField;
typedef typename NaiveStaggeredFermionD::Impl_t STAGIMPL;
typedef NaiveStaggeredFermionD FMat;
typedef typename NaiveStaggeredFermionD::SiteSpinor vobj;
#endif

class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, std::string, gammas, bool,
                                  shiftParityOdd);
};

void swapEvecCheckerFn(FermionFieldD &out,
                       const std::vector<FermionFieldD> &lowModes,
                       const Vector<ComplexD> eval, FMat &action, int index) {
  FermionFieldD temp(lowModes[0].Grid());

  ComplexD eval_D = ComplexD(0.0, eval[index].imag());
  int cb = lowModes[index].Checkerboard();
  int cbNeg = (cb == Even) ? Odd : Even;

  temp.Checkerboard() = cbNeg;
  action.Meooe(lowModes[index], temp);
  out.Checkerboard() = cbNeg;
  out = (1.0 / eval_D) * temp;
}

void execute(A2AMatrixSet<ComplexD> &mBlock,
             const std::vector<FermionFieldD> &left,
             const std::vector<FermionFieldD> &right,
             std::vector<FermionFieldD> *evecs, const Vector<ComplexD> &evals,
             FMat &action, std::vector<ComplexField> &mom,
             std::vector<StagGamma::SpinTastePair> &stagGammas,
             LatticeGaugeFieldD &U, bool shiftParityOdd) {
  std::vector<StagGamma::SpinTastePair> &gamma = stagGammas;
  // std::vector<Gamma::Algebra> gamma =
  // {Gamma::Algebra::Gamma5,Gamma::Algebra::GammaX,Gamma::Algebra::GammaY,Gamma::Algebra::GammaZ};

  std::vector<FermionFieldD> lowBufi_;
  std::vector<FermionFieldD> lowBufj_;
  int next_ = mom.size();
  int nstr_ = gamma.size();

  GridBase *grid_ = evecs->at(0).Grid();
  int Nt = grid_->_fdimensions[Tdir];

  int blockSize_ = 2 * evecs->size(), cacheBlockSize_ = 2 * evecs->size();

  commVector<ComplexD> mCache(mom.size() * gamma.size() * Nt * cacheBlockSize_ *
                                  cacheBlockSize_,
                              ComplexD(0.0));

  bool checkerboarded_low = true;
  int Ncb = checkerboarded_low
                ? 2
                : 1; // Ncb == 2 if the low modes are checkerboarded

  RealD norm = 1.0;

  int N_low = 0;
  if (evecs != nullptr) {
    norm = 1.0 / ::sqrt(norm2(evecs->at(0))); // Calculate norm of eigenvectors
    N_low =
        evecs->size(); // N_low is the number of evecs for M + evecs for Mdag
  }

  int N_i = left.size() + N_low;  // Total number of bra vectors to contract
  int N_j = right.size() + N_low; // Total number of ket vectors to contract

  double flops, bytes, t_kernel, t_gsum;
  double nodes = grid_->NodeCount();

  if (checkerboarded_low) {
    norm = norm / ::sqrt(2); // Reduce checkerboarded norm to 1/sqrt(2)

    if (blockSize_ % 2 != 0 || cacheBlockSize_ % 2 != 0) {
      assert(0);
    }

    lowBufi_.resize(cacheBlockSize_,
                    evecs->at(0).Grid()); // storage for caching checkerboards
    lowBufj_.resize(cacheBlockSize_, evecs->at(0).Grid());
  }

  int NBlock_i = N_i / blockSize_ +
                 (((N_i % blockSize_) != 0)
                      ? 1
                      : 0); // Round up on the number of blocks to compute
  int NBlock_j = N_j / blockSize_ + (((N_j % blockSize_) != 0) ? 1 : 0);

  bool low_i, low_ii, low_j, low_jj;
  int i, j, evec_i, evec_j, N_ii, N_jj;

  i = 0, evec_i = 0;
  while (i < N_i) { // While we still have bra vectors to contract

    low_i = i < N_low;

    if (low_i) {
      N_ii = MIN(N_low - i, blockSize_);

      if (checkerboarded_low) {
        for (int idxi = evec_i; idxi < (MIN(N_low, i + N_ii)); idxi++) {
          lowBufi_[idxi - evec_i] = evecs->at(
              idxi); // Cache original evecs to avoid excessive Meooe ops.
          swapEvecCheckerFn(evecs->at(idxi), *evecs, evals, action,
                            idxi); // Swap original evec checkerboard to
                                   // complementary checkerboard.
        }
      }
    } else {
      N_ii = MIN(N_i - i, blockSize_);
    }

    j = 0, evec_j = 0;
    while (j < N_j) {

      low_j = j < N_low;

      if (low_j) {
        N_jj = MIN(N_low - j, blockSize_);

        if (checkerboarded_low && i != j) { // Only cache and swap kets if it
                                            // hasn't already been done for bras
          for (int idxj = evec_j; idxj < (MIN(N_low, j + N_jj)); idxj++) {
            lowBufj_[idxj - evec_j] = evecs->at(idxj);
            swapEvecCheckerFn(evecs->at(idxj), *evecs, evals, action,
                              idxj); // Swap original evec checkerboard to
                                     // complementary checkerboard.
          }
        }
      } else {
        N_jj = MIN(N_j - j, blockSize_);
      }

      // Get the W and V vectors for this block^2 set of terms
      flops = 0.0;
      bytes = 0.0;

      double t;
      int ii, jj, evec_ii, evec_jj, N_iii, N_jjj;

      ComplexD *mCache_p = mCache.data();
      accelerator_for(r, mCache.size(), 1,
                      { mCache_p[r] = ComplexD(0.0, 0.0); });

      ii = 0, evec_ii = 0;
      while (ii < N_ii) {

        low_ii = (i + ii) < N_low;

        const FermionFieldD *l_temp_e, *l_temp_o;
        // If there are still low modes to process
        if (low_ii) {
          // Pick the min of how many low modes are left vs. cacheBlockSize_
          N_iii = MIN(N_low - (i + ii), cacheBlockSize_);
          if (checkerboarded_low) {
            if (lowBufi_[0].Checkerboard() == Even) {
              l_temp_e = &lowBufi_[0];
              l_temp_o = &evecs->at(evec_i + evec_ii);
            } else {
              l_temp_o = &lowBufi_[0];
              l_temp_e = &evecs->at(evec_i + evec_ii);
            }
          } else {
            l_temp_e = &evecs->at(evec_i + evec_ii);
            l_temp_o = nullptr;
          }
        } else {
          // Pick the min of how many high modes are left vs. cacheBlockSize_
          N_iii = MIN(N_ii - ii, cacheBlockSize_);
          l_temp_e = &left[i + ii - N_low];
          l_temp_o = nullptr;
        }

        jj = 0, evec_jj = 0;
        while (jj < N_jj) {

          low_jj = (j + jj) < N_low;

          const FermionFieldD *r_temp_e, *r_temp_o;
          // If there are still low modes to process
          if (low_jj) {
            // Pick the min of how many low modes are left vs. cacheBlockSize_
            N_jjj = MIN(N_low - (j + jj), cacheBlockSize_);
            if (i == j) {
              r_temp_e = l_temp_e;
              if (checkerboarded_low)
                r_temp_o = l_temp_o;
              else
                r_temp_o = nullptr;

            } else if (checkerboarded_low) {
              if (lowBufj_[0].Checkerboard() == Even) {
                r_temp_e = &lowBufj_[0];
                r_temp_o = &evecs->at(evec_j + evec_jj);
              } else {
                r_temp_o = &lowBufj_[0];
                r_temp_e = &evecs->at(evec_j + evec_jj);
              }
            } else {
              r_temp_e = &evecs->at(evec_j + evec_jj);
              r_temp_o = nullptr;
            }
          } else {
            // Pick the min of how many high modes are left vs. cacheBlockSize_
            N_jjj = MIN(N_jj - jj, cacheBlockSize_);
            r_temp_e = &right[j + jj - N_low];
            r_temp_o = nullptr;
          }

          A2AMatrixSet<ComplexD> mCacheBlock(mCache.data(), next_, nstr_, Nt,
                                             2 * N_iii, 2 * N_jjj);

          std::cout << GridLogMessage << "Before kernel" << std::endl;
          // util.StagMesonFieldNoGlobalSum(mCacheBlock,l_temp_e,l_temp_o,r_temp_e,r_temp_o,gamma,mom,Tdir,&U);
          A2AutilsMILC<STAGIMPL>::StagMesonFieldNoGlobalSum(
              mCacheBlock, l_temp_e, l_temp_o, r_temp_e, r_temp_o, gamma, mom,
              Tdir, &U);
          // MyA2Autils<STAGIMPL>::StagMesonFieldNoGlobalSum(mCacheBlock,l_temp_e,l_temp_o,r_temp_e,r_temp_o,gamma,mom,Tdir,&U);
          // A2AutilsMILC<STAGIMPL>::StagMesonFieldLocalMILCOld(mCacheBlock,l_temp,r_temp,gamma,mom,Tdir);
          std::cout << GridLogMessage << "After kernel" << std::endl;

          jj += N_jjj;
          if (low_jj)
            evec_jj += N_jjj;
        }

        ii += N_iii;
        if (low_ii)
          evec_ii += N_iii;
      }

      t_gsum = -usecond();
      grid_->GlobalSumVector(mCache.data(), mCache.size());
      t_gsum += usecond();

      ComplexD *result_p = mBlock.data();
      ComplexD *cache_p = mCache.data();
      ComplexD *evals_p = (ComplexD *)&evals[0];
      accelerator_for(jj, N_jj, 1, {
        for (int e = 0; e < next_; e++) {
          for (int s = 0; s < nstr_; s++) {
            for (int t = 0; t < Nt; t++) {
              for (int ii = 0; ii < N_ii; ii++) {
                ComplexD coeff = 1.0;
                int idx = 2 * jj + 4 * N_low * ii +
                          4 * N_low * N_low * (t + Nt * (s + nstr_ * e));
                // int idx = jj + N_jj*ii + N_jj*N_ii*t + N_jj*N_ii*Nt*(s +
                // nstr_*e); If the ket vectors (corresponding to the solves)
                // are low modes, multiply by the eigenvals
                if ((i + ii) < N_low || (j + jj) < N_low) {
                  coeff = norm; // Normalize low modes appropriately
                  if ((i + ii) < N_low && (j + jj) < N_low) {
                    coeff *= coeff;
                  }

                  if ((j + jj) < N_low) {
                    coeff = coeff / evals_p[evec_j + jj]; // Minv evals
                  }
                }
                result_p[idx] = coeff * cache_p[idx];
                result_p[idx + 2 * N_low] = coeff * cache_p[idx + 2 * N_low];
                result_p[idx + 1] = conjugate(coeff) * cache_p[idx + 1];
                result_p[idx + 2 * N_low + 1] =
                    conjugate(coeff) * cache_p[idx + 2 * N_low + 1];
              }
            }
          }
        }
      });
      // IO
      double ioTime;
      unsigned int myRank = grid_->ThisRank(), nRank = grid_->RankCount();

      if (checkerboarded_low && low_j && i != j) {
        for (int idxj = evec_j; idxj < (MIN(N_low, j + N_jj)); idxj++)
          evecs->at(idxj) = lowBufj_[idxj - evec_j];
      }
      j += N_jj;
      evec_j += (N_jj);
    } // End while (j < N_j) Loop

    if (checkerboarded_low && low_i) {
      for (int idxi = evec_i; idxi < (MIN(N_low, i + N_ii)); idxi++)
        evecs->at(idxi) = lowBufi_[idxi - evec_i];
    }
    i += N_ii;
    evec_i += (N_ii);
  }
}

class MesonFile : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFile, std::string, gammaName,
                                  std::vector<ComplexD>, data);
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 1;

  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()),
      GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian *UGrid_f = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()),
      GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid_f =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);

  FermionFieldD src1(UGrid);
  FermionFieldD src2(UGrid);
  FermionFieldD result1(UGrid);
  FermionFieldD result2(UGrid);
  FermionFieldD temp(UGrid);

  LatticeGaugeFieldD U(UGrid);
  LatticeGaugeFieldD U_fat(UGrid);
  LatticeGaugeFieldD U_long(UGrid);

  LatticeGaugeFieldF Uf(UGrid_f);
  LatticeGaugeFieldF U_fatf(UGrid_f);
  LatticeGaugeFieldF U_longf(UGrid_f);
  LatticeColourMatrixD Umu(UGrid);

  RealD mass = 2 * 0.1, c1 = 2 * 1.0, c2 = 2 * 1.0, u0 = 1.0;

  // std::vector<StagGamma::SpinTastePair> gammas =
  // {{StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::GX},
  //                                                 {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::GY},
  //                                                 {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::GZ},
  //                                                 {StagGamma::StagAlgebra::GT,StagGamma::StagAlgebra::GT},
  //                                                 {StagGamma::StagAlgebra::GXY,StagGamma::StagAlgebra::GXY},
  //                                                 {StagGamma::StagAlgebra::GZX,StagGamma::StagAlgebra::GZX},
  //                                                 {StagGamma::StagAlgebra::GXT,StagGamma::StagAlgebra::GXT},
  //                                                 {StagGamma::StagAlgebra::GYZ,StagGamma::StagAlgebra::GYZ},
  //                                                 {StagGamma::StagAlgebra::GYT,StagGamma::StagAlgebra::GYT},
  //                                                 {StagGamma::StagAlgebra::GZT,StagGamma::StagAlgebra::GZT},
  //                                                 {StagGamma::StagAlgebra::G5X,StagGamma::StagAlgebra::G5X},
  //                                                 {StagGamma::StagAlgebra::G5Y,StagGamma::StagAlgebra::G5Y},
  //                                                 {StagGamma::StagAlgebra::G5Z,StagGamma::StagAlgebra::G5Z},
  //                                                 {StagGamma::StagAlgebra::G5T,StagGamma::StagAlgebra::G5T},
  //                                                 {StagGamma::StagAlgebra::G5,StagGamma::StagAlgebra::G5},
  //                                                 {StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::G1},
  //                                                 {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::G1},
  //                                                 {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::G1}};

  params.boundary_phases[Tdir] = 1.0;

  /*    std::vector<StagGamma::SpinTastePair> gammas =
     {{StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::G1},
      {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::G1},
      {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::G1}};*/

  GlobalPar par;
  XmlReader reader("params/param.xml", false, "grid");
  read(reader, "parameters", par);

  std::vector<StagGamma::SpinTastePair> gammas =
      strToVec<StagGamma::SpinTastePair>(par.gammas);
  /*    std::vector<StagGamma::SpinTastePair> gammas = {
      {StagGamma::StagAlgebra::G5,StagGamma::StagAlgebra::G5},
      {StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::GX},
      {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::GY},
      {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::GZ}};*/

  std::vector<MesonFile> MF(gammas.size());

  std::vector<ComplexField> mom(1, UGrid);
  mom[0] = 1.0;
  int nEvecs = 384, ki = 0, kf = nEvecs;
  bool multifile = false;
  PackRecord record;

  std::string filename = "vec_384_odd_ks_links_fine.20.bin";
  // std::string filename = "vec_naive_odd.20.bin";

  std::vector<FermionFieldD> evec(nEvecs, UrbGrid);
  std::vector<RealD> eval(nEvecs);
  Vector<ComplexD> evalMassive(nEvecs);

  readPack(evec, eval, record, filename, ki, kf, multifile);

  for (auto i = 0; i < nEvecs; i++) {
    auto lam = ::sqrt(eval[i]);
    evalMassive[i] = ComplexD(mass, lam);
    evec[i].Checkerboard() = Odd;
  }

  int Nt = UGrid->_fdimensions[Tdir];

#if 1
#if 1
  FieldMetaData header;
  // std::string file("configs_a2a/fatKSAPBC.l4444.ildg.20");
  std::string file("configs_a2a/ks_links/fatlinks.l4444.ildg.20");
  IldgReader IR;

  IR.open(file);
  IR.readConfiguration(U_fat, header);
  IR.close();

  // file = "configs_a2a/lngKSAPBC.l4444.ildg.20";
  file = "configs_a2a/ks_links/longlinks.l4444.ildg.20";
  IR.open(file);
  IR.readConfiguration(U_long, header);
  IR.close();

  // file = "configs_a2a/lat.sample.l4444.ildg.20";
  file = "configs_a2a/lat.sample.l4444.ildg.20";
  IR.open(file);
  IR.readConfiguration(U, header);
  IR.close();

  precisionChange(U_fatf, U_fat);
  precisionChange(U_longf, U_long);

  ImprovedStaggeredFermionD stag(*UGrid, *UrbGrid, mass, c1, c2, u0, params);
  ImprovedStaggeredFermionF stag_f(*UGrid_f, *UrbGrid_f, mass, c1, c2, u0,
                                   params);
#else
  SU3::ColdConfiguration(U_fat);
  SU3::ColdConfiguration(U_fatf);
  SU3::ColdConfiguration(U_long);
  SU3::ColdConfiguration(U_longf);
  ImprovedStaggeredFermionD stag(*UGrid, *UrbGrid, mass, c1, c2, u0, params);
  ImprovedStaggeredFermionF stag_f(*UGrid_f, *UrbGrid_f, mass, c1, c2, u0,
                                   params);
#endif
  stag.ImportGaugeSimple(U_long, U_fat);
  stag_f.ImportGaugeSimple(U_longf, U_fatf);

  MdagMLinearOperator<ImprovedStaggeredFermionD, FermionFieldD> hermOp(stag);
  MdagMLinearOperator<ImprovedStaggeredFermionF, FermionFieldF> hermOp_f(
      stag_f);

#else
#if 1
  FieldMetaData header;
  std::string file("configs/lat.sample.l4444.ildg.20");
  // std::string file("configs/milc.l4448.ildg.50");
  // std::string file("configs/milc.l4448.lime.60");
  // Record       record;
  IldgReader IR;
  // ScidacReader IR;

  IR.open(file);
  IR.readConfiguration(U, header);
  // IR.readScidacFieldRecord(U_fat,record);
  IR.close();
  precisionChange(Uf, U);
#else
#if 0
  std::vector<int> seeds4({1,2,3,4});
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  SU<Nc>::HotConfiguration(RNG4,U_fat);
  precisionChange(U_fatf,U_fat);
  std::string file("./Umu.l4448.ildg.20");
  // IldgWriter _IldgWriter(UGrid->IsBoss());
  // _IldgWriter.open(file);
  // _IldgWriter.writeConfiguration(U_fat,40,std::string("dummy_ildg_LFN"),std::string("dummy_config"));
  // _IldgWriter.close();

#else
  SU3::ColdConfiguration(U);
  SU3::ColdConfiguration(Uf);
#endif
#endif
  NaiveStaggeredFermionD stag(U, *UGrid, *UrbGrid, mass, c1, u0, params);
  NaiveStaggeredFermionF stag_f(Uf, *UGrid_f, *UrbGrid_f, mass, c1, u0, params);
  MdagMLinearOperator<NaiveStaggeredFermionD, FermionFieldD> hermOp(stag);
  MdagMLinearOperator<NaiveStaggeredFermionF, FermionFieldF> hermOp_f(stag_f);
#endif

  /*      CartesianStencil<vobj,vobj,int> StencilEven(UrbGrid,1,0,{1},{1},0);
        CartesianStencil<vobj,vobj,int> StencilOdd(UrbGrid,1,1,{0},{1},0);

        StencilEntry* SE;
        SimpleCompressor<vobj> compressor;

        Coordinate ocoor(4,0), icoor(4,0), lcoor(4,0);
        src1=Zero();
        ColourVector kronecker; kronecker=Zero();
        TComplex colorKronecker; colorKronecker = 1.0;
        pokeIndex<ColourIndex>(kronecker,colorKronecker,0);
        pokeSite(kronecker,src1,lcoor);

        FermionFieldD test(UrbGrid), test2(UrbGrid);

        pickCheckerboard(Even,test,src1);

        // Stencil.HaloExchange(test,compressor);
        StencilEven.HaloExchange(test,compressor);
        // StencilOdd.HaloExchange(test,compressor);

       test2 = Cshift(test,1,1);
        int ptype;
        autoView( test_v  , test  , CpuRead);
        autoView( test2_v  , test2  , CpuRead);
        // autoView( st_v  , Stencil  , CpuRead);
        autoView( ste_v  , StencilEven  , CpuRead);
        // autoView( sto_v  , StencilOdd  , CpuRead);

        std::cout << "test CB: " << (test.Checkerboard() == Even? "Even" :
     "Odd") << std::endl;
        // std::cout << "test shifted CB: " << (test2.Checkerboard() == Even?
     "Even" : "Odd") << std::endl;
        // std::cout << "test shifted CB: " << (StencilEven._checkerboard ==
     Even? "Even" : "Odd") << std::endl; int count = 1; icoor[0] = 0; icoor[1] =
     0;

      std::cout << "Ugrid rdimensions: " << UGrid->_rdimensions << std::endl;
      std::cout << "Urbgrid rdimensions: " << UrbGrid->_rdimensions <<
     std::endl; std::cout << "UGrid osites: " << UGrid->oSites() << std::endl;
      for (int i = 0; i<UGrid->oSites();i++) {
          int oSiteCheckerboard;

          // UrbGrid->InOutCoorToLocalCoor(ocoor,icoor,lcoor);

          UGrid->oCoorFromOindex(ocoor,i);

          oSiteCheckerboard=UrbGrid->CheckerBoard(ocoor);

          int cbOSite=test2.Grid()->oIndex(ocoor);

          std::cout << "full ocoor CB " << i << ": " << (oSiteCheckerboard ==
     Even? "Even" : "Odd") << std::endl; if (oSiteCheckerboard ==
     test2.Checkerboard()) { std::cout << "index: " << cbOSite << ", coords: "
     << ocoor << std::endl; std::cout << "Shifted  field value: " <<
     test2_v[cbOSite] << std::endl;
          }

          if (oSiteCheckerboard == Odd) {
              SE=ste_v.GetEntry(ptype,0,cbOSite);
              decltype(coalescedRead(test_v[0])) out;
              if(SE->_is_local) {
                  out =
     coalescedReadPermute(test_v[SE->_offset],ptype,SE->_permute); } else { out
     = coalescedRead(ste_v.CommBuf()[SE->_offset]);
              }

             auto c = test2_v(cbOSite)()()(0);
             std::cout << "cbOSite: " << cbOSite << ", SE->_offset: " <<
     SE->_offset << std::endl; std::cout << "Shifted  field value: " <<
     *((decltype(c)::scalar_type*)&c) << std::endl; std::cout << "Shifted  field
     value: " << *((decltype(out)::scalar_type*)&out) << std::endl; } else {
           auto c = test_v(cbOSite)()()(0);
           std::cout << "Original field value: " <<
     *((decltype(c)::scalar_type*)&c) << std::endl;
          }
          count++;
      }*/
  //    exit(0);
  MixedPrecisionConjugateGradient<FermionFieldD, FermionFieldF> mCG(
      1.0e-8, 10000, 50, UrbGrid_f, hermOp_f, hermOp);
  ConjugateGradient<FermionFieldD> CG(1.0e-15, 10000);
  // BiCGSTAB<FermionFieldD> CG(1.0e-10,10000);

  LatticeComplexD KSphases(UGrid);
  Lattice<iScalar<vInteger>> x(UGrid);
  LatticeCoordinate(x, 0);
  Lattice<iScalar<vInteger>> y(UGrid);
  LatticeCoordinate(y, 1);
  Lattice<iScalar<vInteger>> z(UGrid);
  LatticeCoordinate(z, 2);
  Lattice<iScalar<vInteger>> t(UGrid);
  LatticeCoordinate(t, 3);
  Lattice<iScalar<vInteger>> lin_z(UGrid);
  lin_z = x + y;
  Lattice<iScalar<vInteger>> lin_t(UGrid);
  lin_t = x + y + z;
  // Lattice<iScalar<vInteger> > lin_y(UGrid); lin_y=t+x;
  // Lattice<iScalar<vInteger> > lin_z(UGrid); lin_z=t+x+y;

  /*  for (int mu = 0; mu < Nd; mu++) {
      KSphases = 1.0;
      if ( mu == 1 ) KSphases = where( mod(x    ,2)==(Integer)0,
    KSphases,-KSphases); if ( mu == 2 ) KSphases = where(
    mod(lin_z,2)==(Integer)0, KSphases,-KSphases); if ( mu == 3 ) KSphases =
    where( mod(lin_t,2)==(Integer)0, KSphases,-KSphases);
      // if ( mu == 0 ) KSphases = where( mod(t    ,2)==(Integer)0,
    KSphases,-KSphases);
      // if ( mu == 1 ) KSphases = where( mod(lin_y,2)==(Integer)0,
    KSphases,-KSphases);
      // if ( mu == 2 ) KSphases = where( mod(lin_z,2)==(Integer)0,
    KSphases,-KSphases); Umu    = PeekIndex<LorentzIndex>(U_fat, mu); Umu = Umu
    * KSphases; PokeIndex<LorentzIndex>(U_fat,Umu,mu);
    }*/
  std::vector<FermionFieldD> left, right;
  int doubleEvec = 2 * evec.size();
  Vector<ComplexD> mData(
      mom.size() * gammas.size() * Nt * doubleEvec * doubleEvec, ComplexD(0.0));
  A2AMatrixSet<ComplexD> mBlock(mData.data(), mom.size(), gammas.size(), Nt,
                                doubleEvec, doubleEvec);

  execute(mBlock, left, right, &evec, evalMassive, stag, mom, gammas, U,
          par.shiftParityOdd);

  A2AMatrix<ComplexD> prod, tmp, ref;
  std::vector<std::set<unsigned int>> times;
  std::vector<std::vector<unsigned int>> timeSeq;
  std::set<unsigned int> translations;
  std::vector<A2AMatrixTr<ComplexD>> lastTerm(Nt);

  prod.resize(doubleEvec, doubleEvec);
  tmp.resize(doubleEvec, doubleEvec);
  ref.resize(doubleEvec, doubleEvec);

  times.push_back(parseTimeRange("0", Nt));

  translations = parseTimeRange("0..3", Nt);

  makeTimeSeq(timeSeq, times);

  for (auto g = 0; g < gammas.size(); g++) {

    MF[g].gammaName = StagGamma::GetName(gammas[g]);
    MF[g].data.resize(Nt, 0.);
    for (unsigned int t = 0; t < Nt; ++t) {
      for (int ii = 0; ii < doubleEvec; ii++) {
        for (int jj = 0; jj < doubleEvec; jj++) {
          ref(ii, jj) = mBlock(0, g, t, ii, jj);
        }
      }
      lastTerm[t].resize(ref.rows(), ref.cols());
      thread_for(j, ref.cols(), {
        for (unsigned int i = 0; i < ref.rows(); ++i) {
          lastTerm[t](i, j) = ref(i, j);
        }
      });
    }

    for (unsigned int i = 0; i < timeSeq.size(); ++i) {

      auto &t = timeSeq[i];

      for (unsigned int tLast = 0; tLast < Nt; ++tLast) {
        MF[g].data[tLast] = 0.;
      }

      for (auto &dt : translations) {
        for (int ii = 0; ii < doubleEvec; ii++) {
          for (int jj = 0; jj < doubleEvec; jj++) {
            prod(ii, jj) = mBlock(0, g, TIME_MOD(t[0] + dt), ii, jj);
          }
        }

        for (unsigned int tLast = 0; tLast < Nt; ++tLast) {
          accTrMul(MF[g].data[TIME_MOD(tLast - dt)], prod, lastTerm[tLast]);
        }
      }
      for (unsigned int tLast = 0; tLast < Nt; ++tLast) {
        MF[g].data[tLast] /= translations.size();
      }
    }
  }

  XmlWriter WR("test_a2a.xml");
  write(WR, "MesonFile", MF);

  Grid_finalize();
}
