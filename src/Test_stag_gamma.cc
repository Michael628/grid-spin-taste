#include <Grid/Grid.h>
#include <StagGamma.h>

using namespace std;
using namespace Grid;

class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, std::string, gauge, std::string,
                                  gaugeFat, std::string, gaugeLong, bool, free,
                                  std::string, gammas, int, trajectory);
};

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

  std::string paramFile = argv[1];
  XmlReader reader(paramFile, false, "grid");

  GlobalPar inputParams;
  read(reader, "parameters", inputParams);
  {
    // Create double precision grid layout
    auto latt = GridDefaultLatt(); // Lattice size, specified by run-time flag:
                                   // --grid x.y.z.t
    GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
        latt,

        GridDefaultSimd(
            Nd,
            vComplexD::Nsimd()), // Specifies how to split the lattice based
                                 // on how many vectorized (SIMD) operations
                                 // can be done at the same time on the
                                 // given CPU hardware. Since vectorized
                                 // instructions have a fixed bit width (512
                                 // bits, for example), this depends on the
                                 // data type you want to vectorize (eg, 512
                                 // bits can fit 4 double precision complex
                                 // numbers)
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
    std::vector<std::vector<int>> coors = {
        {0, 0, 0, 0},
        // {2, 0, 0, 0},
        // {1, 0, 0, 0},
        // {3, 0, 0, 0},
        // {4, 0, 0, 0},
        // {5, 0, 0, 0},
    };

    // List of spin-taste gammas to test
    std::vector<StagGamma::SpinTastePair> spinTastes =
        StagGamma::ParseSpinTaste(inputParams.gammas);

    // Set boundary conditions, -1.0 for antiperiodic time
    params.boundary_phases[Tdir] = -1.0;

    if (!inputParams.free) {
      int traj = inputParams.trajectory;
      // Load smeared links from file
      FieldMetaData header;
      std::string file(inputParams.gaugeFat + "." + std::to_string(traj));
      IldgReader IR;

      IR.open(file);
      IR.readConfiguration(U_fat, header);
      IR.close();

      file = inputParams.gaugeLong + "." + std::to_string(traj);
      IR.open(file);
      IR.readConfiguration(U_long, header);
      IR.close();

      file = inputParams.gauge + "." + std::to_string(traj);
      IR.open(file);
      IR.readConfiguration(U, header);
      IR.close();
    } else {
      SU3::ColdConfiguration(U_fat);
      SU3::ColdConfiguration(U);
      SU3::ColdConfiguration(U_long);
    }
    // Create Dirac matrix for single and double precision
    ImprovedStaggeredFermionD stag(*UGrid, *UrbGrid, mass, c1, c2, u0, params);

    // Initialize dirac matrix by importing smeared fields
    stag.ImportGaugeSimple(U_long, U_fat);

    // Instanciate hermitian operators for single and double precision
    MdagMLinearOperator<ImprovedStaggeredFermionD, FermionFieldD> hermOp(stag);

    // Instanciate conjugate gradient solver for double recision
    ConjugateGradient<FermionFieldD> CG(1.0e-15, 10000); // (tol, max_iter)

    // Initialize output vector
    MF.resize(spinTastes.size() * coors.size());

    // Instanciate gamma sources and sinks
    StagGamma gammaSource, gammaSink;
    StagGamma g5g5(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);

    // Set gauge field to be used for Cshifts
    gammaSource.setGaugeField(U);
    gammaSink.setGaugeField(U);

    // Just a complex number wrapper...
    TComplex colorKronecker;
    colorKronecker = 1.0;

    // A colour vector of complex numbers
    ColourVector kronecker;

    int i = 0;
    for (StagGamma::SpinTastePair &st : spinTastes) {

      // Update StagGamma object with new `source` spin-taste
      gammaSource.setSpinTaste(st);
      gammaSource.setGaugeField(U);
      gammaSource = gammaSource * g5g5;

      // Update StagGamma object with new `sink` spin-taste
      gammaSink.setSpinTaste(st);
      gammaSink.setGaugeField(U);
      gammaSink = gammaSink * g5g5;

      // Loop over lattice point coordinates
      for (auto &coor : coors) {

        // Initialize MesonFile for coor, spin-taste combination
        MF[i].data.resize(latt[Tdir], 0.0); // (new_size, initial_value)
        MF[i].gammaName = StagGamma::GetName(st);
        MF[i].coor = vecToStr(coor);

        // Average over point source prop for each color index
        for (int j = 0; j < 3; j++) {

          // Zero out the color vector delta
          kronecker = Zero();

          // Zero out all fields for next CG solve
          quark1 = Zero();
          quark2 = Zero();
          src = Zero();
          temp = Zero();

          // kronecker[j] = 1.0
          pokeIndex<ColourIndex>(kronecker, colorKronecker, j);

          // src[coor] = kronecker
          pokeSite(kronecker, src, coor);

          // Multiply source field by source gamma
          quark1 = gammaSource * src;

          // temp = Mdag * quark1
          stag.Mdag(quark1, temp);

          // Solve Mdag*M * quark1 = Mdag*quark1
          // Note: hermop = Mdag*M
          CG(hermOp, temp,
             quark1); // Performs CG solve, stores result in quark1

          // Repeat for `antiquark`
          // Note: no gamma applied yet
          stag.Mdag(src, temp);
          CG(hermOp, temp,
             quark2); // Performs CG solve, stores result in quark2

          // Multiply `antiquark` field by sink gamma
          quark2 = gammaSink * quark2;

          // Initialize lattice of complex numbers
          ComplexFieldD meson_CF(quark1.Grid());

          // Inner product over all internal indices (here: color),
          // no spatial sum
          meson_CF = localInnerProduct(quark2, quark1);

          // Initialize SIMD vector of complex numbers
          std::vector<TComplex> meson_T;

          // Sum over spatial indices of lattice, not time
          sliceSum(meson_CF, meson_T, Tdir);

          int nt = meson_T.size();
          int shift = coor[Tdir];
          int offset;
          std::vector<Complex> corr(nt, 0);
          std::cout << MF[i].gammaName << ", coor: " << MF[i].coor
                    << ", nt: " << nt << ", j: " << j << " channel (";

          for (int t = 0; t < nt; t++) {
            offset = (shift + t) % nt;
            corr[t] = TensorRemove(meson_T[offset]);
            std::cout << corr[t].real() << ", ";
          }
          std::cout << ")" << std::endl;

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
