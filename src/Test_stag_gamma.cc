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

  typename ImprovedStaggeredFermionD::ImplParams params;
  typedef typename ImprovedStaggeredFermionD::PropagatorField PropagatorFieldD;
  typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;
  typedef typename ImprovedStaggeredFermionD::ComplexField ComplexFieldD;
  {
    // Create double precision grid layout
    GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
        GridDefaultLatt(), // Lattice size, specified by run-time flag:
                           // --grid x.y.z.t

        GridDefaultSimd(
            Nd,
            vComplexD::Nsimd()), // Specifies how to split the lattice based on
                                 // how many vectorized (SIMD) operations can be
                                 // done at the same time on the given CPU
                                 // hardware. Since vectorized instructions have
                                 // a fixed bit width (512 bits, for example),
                                 // this depends on the data type you want
                                 // to vectorize (eg, 512 bits can fit 4 double
                                 // precision complex numbers)
        GridDefaultMpi());       // lattice blocks distributed over mpi ranks,
                                 // specified by run-time flag: --mpi x.y.z.t

    // Build checkerboarded grid from full grid
    GridRedBlackCartesian *UrbGrid =
        SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
    // Create fermion field objects
    FermionFieldD src(UGrid);
    FermionFieldD quark1(UGrid);
    FermionFieldD quark2(UGrid);
    FermionFieldD temp(UGrid);

    // Create double precision gauge field objects (4 links per site)
    LatticeGaugeFieldD U(UGrid);
    LatticeGaugeFieldD U_fat(UGrid);
    LatticeGaugeFieldD U_long(UGrid);

    // Create double precision link field object (1 link per site)
    LatticeColourMatrixD Umu(UGrid);

    // Coefficients for HISQ action (with MILC factors of 2)
    RealD mass = 2 * 0.1, c1 = 2 * 1.0, c2 = 2 * 1.0, u0 = 1.0;

    // Storage for saving final results
    std::vector<MesonFile> MF;

    // List of lattice point source locations to test
    std::vector<std::vector<int>> coors = {{0, 0, 0, 0}};

    // List of spin-taste gammas to test
    std::vector<StagGamma::SpinTastePair> spinTaste = {
        {StagGamma::StagAlgebra::GX, StagGamma::StagAlgebra::G1},
        {StagGamma::StagAlgebra::GY, StagGamma::StagAlgebra::G1},
        {StagGamma::StagAlgebra::GZ, StagGamma::StagAlgebra::G1}};

    // Set boundary conditions, -1.0 for antiperiodic time
    params.boundary_phases[Tdir] = 1.0;

    std::string workPath("../");
#if 1
    // Load smeared links from file
    FieldMetaData header;
    std::string file(workPath + "configs/fatlinks.l4444.ildg.20");
    IldgReader IR;

    IR.open(file);
    IR.readConfiguration(U_fat, header);
    IR.close();

    file = workPath + "configs/longlinks.l4444.ildg.20";
    IR.open(file);
    IR.readConfiguration(U_long, header);
    IR.close();

    file = workPath + "configs/lat.sample.l4444.ildg.20";
    IR.open(file);
    IR.readConfiguration(U, header);
    IR.close();

    // Create Dirac matrix for single and double precision
    ImprovedStaggeredFermionD stag(*UGrid, *UrbGrid, mass, c1, c2, u0, params);
#else
    // Create free field
    SU3::ColdConfiguration(U_fat);
    SU3::ColdConfiguration(U_long);
    ImprovedStaggeredFermionD stag(*UGrid, *UrbGrid, mass, c1, c2, u0, params);
#endif

    // Initialize dirac matrix by importing smeared fields
    stag.ImportGaugeSimple(U_long, U_fat);

    // Instanciate hermitian operators for single and double precision
    MdagMLinearOperator<ImprovedStaggeredFermionD, FermionFieldD> hermOp(stag);

    ConjugateGradient<FermionFieldD> CG(1.0e-15, 10000);

    Coordinate ocoor(4, 0);
    MF.resize(spinTaste.size() * coors.size());

    StagGamma gammaSource, gammaSink;
    StagGamma g5g5(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);

    gammaSource.setGaugeField(U);
    gammaSink.setGaugeField(U);

    int i = 0;
    for (auto &st : spinTaste) {
      FieldMetaData header;

      for (auto &coor : coors) {
        MF[i].data = {0, 0, 0, 0};
        MF[i].gammaName = StagGamma::GetName(st);
        MF[i].coor = vecToStr(coor);

        for (int j = 0; j < 3; j++) {

          src = Zero();
          ColourVector kronecker;
          kronecker = Zero();
          TComplex colorKronecker;
          colorKronecker = 1.0;
          pokeIndex<ColourIndex>(kronecker, colorKronecker, j);

          pokeSite(kronecker, src, coor);

          gammaSource.setSpinTaste(st);
          gammaSource *= g5g5;

          quark1 = gammaSource * src;
          stag.Mdag(quark1, temp);  // temp = Mdag * quark1
          CG(hermOp, temp, quark1); // Solve hermop * quark1 = temp

          stag.Mdag(src, temp);
          CG(hermOp, temp, quark2);

          quark2 *= gammaSink;

          ComplexFieldD meson_CF(quark1.Grid());
          std::vector<TComplex> meson_T;

          meson_CF = localInnerProduct(quark2, quark1);
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
