#include <Grid/Grid.h>
#include <StagGamma.h>
using namespace std;
using namespace Grid;

class MesonFile : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFile, std::string, gammaName,
                                  std::string, coor, std::vector<Complex>,
                                  data);
};

struct Record : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(Record, std::string, title, std::string,
                                  info);
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 1;

#if 1
  typename ImprovedStaggeredFermionD::ImplParams params;
  typedef typename ImprovedStaggeredFermionD::PropagatorField PropagatorFieldD;
  typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;
  typedef typename ImprovedStaggeredFermionF::FermionField FermionFieldF;
  typedef typename ImprovedStaggeredFermionD::ComplexField ComplexField;
  typedef typename ImprovedStaggeredFermionF::ComplexField ComplexFieldF;
#else
  typedef typename NaiveStaggeredFermionD::FermionField FermionFieldD;
  typedef typename NaiveStaggeredFermionF::FermionField FermionFieldF;
#endif
  {
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

    std::vector<MesonFile> MF;

    std::vector<std::vector<int>> coors = {{0, 0, 0, 0}};

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

    /*  std::vector<StagGamma::SpinTastePair> gammas = {
        {StagGamma::StagAlgebra::G5,StagGamma::StagAlgebra::G5},
        {StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::GX},
        {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::GY},
        {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::GZ}
      };*/
    std::vector<StagGamma::SpinTastePair> gammas = {
        {StagGamma::StagAlgebra::GX, StagGamma::StagAlgebra::G1},
        {StagGamma::StagAlgebra::GY, StagGamma::StagAlgebra::G1},
        {StagGamma::StagAlgebra::GZ, StagGamma::StagAlgebra::G1}};

    // std::vector<StagGamma::SpinTastePair> gammas =
    // {{StagGamma::StagAlgebra::GX,StagGamma::StagAlgebra::G1},
    // {StagGamma::StagAlgebra::GY,StagGamma::StagAlgebra::G1},
    // {StagGamma::StagAlgebra::GZ,StagGamma::StagAlgebra::G1}};

    params.boundary_phases[Tdir] = 1.0;
#if 1
#if 1
    FieldMetaData header;
    std::string file("configs/fatlinks.l4444.ildg.20");
    IldgReader IR;

    IR.open(file);
    IR.readConfiguration(U_fat, header);
    IR.close();

    file = "configs/longlinks.l4444.ildg.20";
    IR.open(file);
    IR.readConfiguration(U_long, header);
    IR.close();

    file = "configs/lat.sample.l4444.ildg.20";
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
#if 0
  FieldMetaData header;
  std::string file("configs/lat.sample.l4444.ildg.20");
  // std::string file("configs/milc.l4448.ildg.50");
  // std::string file("configs/milc.l4448.lime.60");
  Record       record;
  IldgReader IR;
  // ScidacReader IR;

  IR.open(file);
  IR.readConfiguration(U,header);
  // IR.readScidacFieldRecord(U,record);
  IR.close();
  precisionChange(Uf,U);
#else
#if 0
  std::vector<int> seeds4({1,2,3,4});
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  SU<Nc>::HotConfiguration(RNG4,U);
  precisionChange(Uf,U);
  std::string file("./Umu.l4448.ildg.20");
  // IldgWriter _IldgWriter(UGrid->IsBoss());
  // _IldgWriter.open(file);
  // _IldgWriter.writeConfiguration(U,40,std::string("dummy_ildg_LFN"),std::string("dummy_config"));
  // _IldgWriter.close();

#else
    SU3::ColdConfiguration(U);
    SU3::ColdConfiguration(Uf);
#endif
#endif
    NaiveStaggeredFermionD stag(U, *UGrid, *UrbGrid, mass, c1, u0, params);
    NaiveStaggeredFermionF stag_f(Uf, *UGrid_f, *UrbGrid_f, mass, c1, u0,
                                  params);
    MdagMLinearOperator<NaiveStaggeredFermionD, FermionFieldD> hermOp(stag);
    MdagMLinearOperator<NaiveStaggeredFermionF, FermionFieldF> hermOp_f(stag_f);
#endif

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
      KSphases,-KSphases); Umu    = PeekIndex<LorentzIndex>(U, mu); Umu = Umu *
      KSphases; PokeIndex<LorentzIndex>(U,Umu,mu);
      }*/

    RealD sign;

    MF.resize(gammas.size() * coors.size());
    int i = 0;
    for (auto &g : gammas) {
      // RealD sign = 1.0;

      // uint8_t g1, g2, flip = 0;
      // switch(g.first) {
      // case StagGamma::StagAlgebra::GXZ:
      // case StagGamma::StagAlgebra::GTX:
      // case StagGamma::StagAlgebra::GTY:
      // case StagGamma::StagAlgebra::GTZ:
      // case StagGamma::StagAlgebra::G5Y:
      // case StagGamma::StagAlgebra::G5T:

      //   for (auto & dir : StagGamma::gmu) {
      //     if (dir & g.first) {
      //       g1 = dir ^ StagGamma::StagAlgebra::G5;
      //       g2 = dir ^ g.first;
      //       flip = g1^g2;
      //       break;
      //     }
      //   }
      //   break;
      // }

      // switch(g.second) {
      // case StagGamma::StagAlgebra::GXZ:
      // case StagGamma::StagAlgebra::GTX:
      // case StagGamma::StagAlgebra::GTY:
      // case StagGamma::StagAlgebra::GTZ:
      //   for (auto & dir : StagGamma::gmu) {
      //     if (dir & g.second) {
      //       g1 = dir ^ StagGamma::StagAlgebra::G5;
      //       g2 = dir ^ g.second;
      //       flip = flip ^ g1 ^ g2;
      //       break;
      //     }
      //   }
      //   break;
      // }

      // for (auto & dir : StagGamma::gmu) {
      //   if (dir & flip) {
      //     sign = -sign;
      //   }
      // }
      FieldMetaData header;
      // std::string file("configs/lat.sample.l4444.ildg.20");
      // std::string file("configs/milc.l4448.ildg.50");
      for (auto &coor : coors) {
        MF[i].data = {0, 0, 0, 0};
        MF[i].gammaName = StagGamma::GetName(g);
        MF[i].coor = vecToStr(coor);

        for (int j = 0; j < 3; j++) {

          Coordinate ocoor(4, 0);
          Coordinate coor1, coor2;
          coor1 = coor;
          coor2 = coor1;

          src1 = Zero();
          src2 = Zero();
          ColourVector kronecker;
          kronecker = Zero();
          TComplex colorKronecker;
          colorKronecker = 1.0;
          pokeIndex<ColourIndex>(kronecker, colorKronecker, j);
          pokeSite(kronecker, src1, coor1);
          pokeSite(kronecker, src2, coor2);

          StagGamma st1(g.first, g.second);
          StagGamma st2(StagGamma::StagAlgebra::G1, StagGamma::StagAlgebra::G1);
          StagGamma st3(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);
          StagGamma st4(StagGamma::StagAlgebra::G5Y,
                        StagGamma::StagAlgebra::G5);
          StagGamma st5, st6;

          st1.setGaugeField(U);
          st2.setGaugeField(U);
          st3.setGaugeField(U);

          st5 = st1 * st3;
          st6 = st3 * st1;

          st5(result1, src1);

          stag.Mdag(result1, temp);
          CG(hermOp, temp, result1);

          st2(result2, src2);

          stag.Mdag(result2, temp);
          CG(hermOp, temp, result2);

          st5(result2, result2);

          PropagatorFieldD temp_CF(result1.Grid()), temp2_CF(result1.Grid());
          ComplexField meson_CF(result1.Grid());
          std::vector<TComplex> meson_T;

          meson_CF = localInnerProduct(result2, result1);
          sliceSum(meson_CF, meson_T, Tdir);

          int nt = meson_T.size();
          int shift = coor[Tdir];
          int offset;
          std::vector<Complex> corr(nt, 0);
          std::cout << " channel (";
          for (int t = 0; t < nt; t++) {
            offset = (shift + t) % nt;
            corr[t] = TensorRemove(meson_T[offset]);
            std::cout << corr[t].real() << ", ";
          }
          std::cout << std::endl;

          for (int t = 0; t < nt; t++) {
            MF[i].data[t] += corr[t];
          }
        }
        i++;
      }
    }
    XmlWriter WR("stag_gamma_output.xml");
    write(WR, "MesonFile", MF);
  }

  Grid_finalize();
}
