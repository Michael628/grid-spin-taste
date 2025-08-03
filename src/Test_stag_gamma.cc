#include <Grid/Grid.h>
#include <StagGamma.h>

using namespace std;
using namespace Grid;

// clang-format off
class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, 
                                  std::string, gauge, 
                                  std::string, gaugeFat, 
                                  std::string, gaugeLong, 
                                  Real, mass,
                                  // bool, free,
                                  std::string, gammas, 
                                  std::string, writeFile, 
                                  std::string, trajectory);
};
// clang-format on

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
  typedef typename ImprovedStaggeredFermionD::SitePropagator::scalar_object
      SitePropagator;
  typedef typename ImprovedStaggeredFermionD::Impl_t FImpl;
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

    // Calculate spatial volume
    int spatial_volume = 1;
    for (int i = 0; i < Nd; i++) {
      if (i == Tdir) {
        continue;
      }
      spatial_volume *= latt[i];
    }

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
    FermionFieldD source(UGrid);
    FermionFieldD solution(UGrid);
    PropagatorFieldD quark1(UGrid);
    PropagatorFieldD quark2(UGrid);
    ComplexFieldD result(UGrid);
    FermionFieldD temp(UGrid);

    // Create double precision gauge field objects (4 links per site)
    LatticeGaugeFieldD U(UGrid);
    LatticeGaugeFieldD U_fat(UGrid);
    LatticeGaugeFieldD U_long(UGrid);

    // Create double precision link field object (1 link per site)
    LatticeColourMatrixD Umu(UGrid);

    // Coefficients for HISQ action (with MILC factors of 2)
    RealD mass = 2. * inputParams.mass, c1 = 2. * 1.0, c2 = 2. * 1.0, u0 = 1.0;

    // Storage for saving final results
    std::vector<MesonFile> MF;

    // List of lattice point source locations to test
    // clang-format off
    std::vector<std::vector<int>> coors = {
      {0, 0, 0, 0},
      // {0, 0, 0, 1},
      // {0, 0, 1, 0},
      // {0, 0, 1, 1},
      // {0, 1, 0, 0},
      // {0, 1, 0, 1},
      // {0, 1, 1, 0},
      // {0, 1, 1, 1},
      // {1, 0, 0, 0},
      // {1, 0, 0, 1},
      // {1, 0, 1, 0},
      // {1, 0, 1, 1},
      // {1, 1, 0, 0},
      // {1, 1, 0, 1},
      // {1, 1, 1, 0},
      // {1, 1, 1, 1},
    };
    // clang-format on

    // List of spin-taste gammas to test
    std::vector<StagGamma::SpinTastePair> spinTastes =
        StagGamma::ParseSpinTaste(inputParams.gammas);

    // Set boundary conditions, -1.0 for antiperiodic time
    // WARN: does nothing
    // params.boundary_phases[Tdir] = 1.0;

    std::string trajSuffix = !inputParams.trajectory.empty()
                                 ? std::string(".") + inputParams.trajectory
                                 : std::string("");

    // Load smeared links from file
    FieldMetaData header;
    std::string file(inputParams.gaugeFat + trajSuffix);
    IldgReader IR;

    IR.open(file);
    IR.readConfiguration(U_fat, header);
    IR.close();

    file = inputParams.gaugeLong + trajSuffix;
    IR.open(file);
    IR.readConfiguration(U_long, header);
    IR.close();

    file = inputParams.gauge + trajSuffix;
    IR.open(file);
    IR.readConfiguration(U, header);
    IR.close();

    // WARN: Free field implementation below. Does not work.
    /*
     * SU3::ColdConfiguration(U_fat);
     * SU3::ColdConfiguration(U);
     * SU3::ColdConfiguration(U_long);
     *
     * LatticeComplexD eta_mu(UGrid);
     *
     * // Create lattices of integers
     * Lattice<iScalar<vInteger>> x(UGrid);
     * Lattice<iScalar<vInteger>> y(UGrid);
     * Lattice<iScalar<vInteger>> z(UGrid);
     * Lattice<iScalar<vInteger>> t(UGrid);
     *
     * // Hold the corresponding index direction in each lattice of ints
     * LatticeCoordinate(x, Xdir);
     * LatticeCoordinate(y, Ydir);
     * LatticeCoordinate(z, Zdir);
     * LatticeCoordinate(t, Tdir);
     *
     * // for TXYZ coordinate convention
     * Lattice<iScalar<vInteger>> eta_exp_x(UGrid);
     * Lattice<iScalar<vInteger>> eta_exp_y(UGrid);
     * Lattice<iScalar<vInteger>> eta_exp_z(UGrid);
     * eta_exp_x = t;
     * eta_exp_y = t + x;
     * eta_exp_z = t + x + y;
     *
     * // for XYZT coordinate convention
     * Lattice<iScalar<vInteger>> eta_exp_y_alt(UGrid);
     * Lattice<iScalar<vInteger>> eta_exp_z_alt(UGrid);
     * Lattice<iScalar<vInteger>> eta_exp_t_alt(UGrid);
     * eta_exp_y_alt = x;
     * eta_exp_z_alt = x + y;
     * eta_exp_t_alt = x + y + z;
     *
     * for (int mu = 0; mu < Nd; mu++) {
     *   eta_mu = 1.0;
     *   // if (mu == Ydir)
     *   //   KSphases =
     *   //       where(mod(eta_exp_y_alt, 2) == (Integer)0, KSphases,
     *   //       -KSphases);
     *   // if (mu == Zdir)
     *   //   KSphases =
     *   //       where(mod(eta_exp_z_alt, 2) == (Integer)0, KSphases,
     *   //       -KSphases);
     *   // if (mu == Tdir)
     *   //   KSphases =
     *   //       where(mod(eta_exp_t_alt, 2) == (Integer)0, KSphases,
     *   //       -KSphases);
     *   if (mu == Xdir)
     *     eta_mu = where(mod(eta_exp_x, 2) == (Integer)0, eta_mu, -eta_mu);
     *   if (mu == Ydir)
     *     eta_mu = where(mod(eta_exp_y, 2) == (Integer)0, eta_mu, -eta_mu);
     *   if (mu == Zdir)
     *     eta_mu = where(mod(eta_exp_z, 2) == (Integer)0, eta_mu, -eta_mu);
     *   Umu = PeekIndex<LorentzIndex>(U_long, mu);
     *   Umu = Umu * eta_mu;
     *   PokeIndex<LorentzIndex>(U_long, Umu, mu);
     * }
     * U_fat = U_long;
     */

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

    // TODO: Make this work properly. I.e. set gauge field once and copy pointer
    // on modification
    // Set gauge field to be used for Cshifts
    // gammaSource.setGaugeField(U);
    // gammaSink.setGaugeField(U);

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

        quark1 = Zero();
        quark2 = Zero();

        // Set internal dims (3x3 color matrix) of quark fields to the identity
        SitePropagator identity;
        identity = 1.0;
        pokeSite(identity, quark1, coor);
        pokeSite(identity, quark2, coor);

        // Multiply source field by source gamma
        quark1 = gammaSource * quark1;

        // Average over point source prop for each color index
        for (int colorIndex = 0; colorIndex < Nc; colorIndex++) {

          // Pick one color index from source field
          PropToFerm<FImpl>(source, quark1, colorIndex);

          // temp = Mdag * quark1
          stag.Mdag(source, temp);

          // Mdag*M * solution = Mdag*source
          // Note: hermop = Mdag*M
          CG(hermOp, temp,
             solution); // Performs CG solve

          // Store result in quark1
          FermToProp<FImpl>(quark1, solution, colorIndex);

          // Repeat for `antiquark`
          PropToFerm<FImpl>(source, quark2, colorIndex);

          // temp = Mdag * source
          stag.Mdag(source, temp);

          // Mdag*M * solution = Mdag*source
          CG(hermOp, temp,
             solution); // Performs CG solve

          // Store result in quark2
          FermToProp<FImpl>(quark2, solution, colorIndex);
        }

        // Multiply by sink gamma and take color trace of outer product
        // |quark2><quark1|
        result = trace(gammaSink * quark2 * adj(quark1));

        // Initialize SIMD vector of complex numbers
        std::vector<TComplex> meson_T;

        sliceSum(result, meson_T, Tdir);

        int nt = meson_T.size();
        int shift = coor[Tdir];
        int offset;
        std::vector<Complex> corr(nt, 0);
        std::cout << MF[i].gammaName << ", coor: " << MF[i].coor
                  << ", nt: " << nt << " channel (";

        for (int t = 0; t < nt; t++) {
          offset = (shift + t) % nt;
          corr[t] = TensorRemove(meson_T[offset]);
          std::cout << corr[t].real() << ", ";
        }
        std::cout << ")" << std::endl;

        for (int t = 0; t < nt; t++) {
          MF[i].data[t] += corr[t];
        }
        i++;
      }
    }
    XmlWriter WR(inputParams.writeFile + ".xml");
    write(WR, "MesonFile", MF);
  }

  Grid_finalize();
}
